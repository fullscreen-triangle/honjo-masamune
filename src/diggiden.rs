//! Diggiden Adversarial Testing Engine
//! 
//! The adversarial system that continuously attacks the mzekezeke belief network
//! to find vulnerabilities, test robustness, and improve overall system reliability
//! through systematic probing and attack strategies.

use anyhow::Result;
use atp_manager::AtpManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use rand::{Rng, thread_rng, seq::SliceRandom};

use crate::config::DiggidenAdversarialConfig;
use crate::mzekezeke::{BeliefUpdateResult, Evidence, Belief, BeliefQuery, BeliefQueryType, BeliefQueryResult};

/// The diggiden adversarial testing engine
#[derive(Debug)]
pub struct DiggidenEngine {
    config: DiggidenAdversarialConfig,
    atp_manager: Arc<AtpManager>,
    attack_strategies: Vec<AttackStrategy>,
    vulnerability_database: Arc<RwLock<VulnerabilityDatabase>>,
    attack_history: Arc<RwLock<Vec<AttackRecord>>>,
    current_campaigns: Arc<RwLock<Vec<AttackCampaign>>>,
}

impl DiggidenEngine {
    /// Create a new diggiden adversarial engine
    pub fn new(
        config: DiggidenAdversarialConfig,
        atp_manager: Arc<AtpManager>,
    ) -> Self {
        let attack_strategies = Self::initialize_attack_strategies(&config);
        
        Self {
            config,
            atp_manager,
            attack_strategies,
            vulnerability_database: Arc::new(RwLock::new(VulnerabilityDatabase::new())),
            attack_history: Arc::new(RwLock::new(Vec::new())),
            current_campaigns: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Launch comprehensive adversarial testing campaign
    pub async fn launch_attack_campaign(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
        campaign_config: AttackCampaignConfig,
    ) -> Result<AttackCampaignResult> {
        info!("🎯 Launching diggiden attack campaign: {}", campaign_config.name);

        // Reserve ATP for attack campaign
        let campaign_cost = self.calculate_campaign_cost(&campaign_config);
        let reservation = self.atp_manager.reserve_atp("adversarial_campaign", campaign_cost).await?;

        let campaign_id = Uuid::new_v4();
        let start_time = Utc::now();

        // Initialize campaign
        let mut campaign = AttackCampaign {
            id: campaign_id,
            name: campaign_config.name.clone(),
            start_time,
            end_time: None,
            attack_rounds: Vec::new(),
            total_attacks: 0,
            successful_attacks: 0,
            vulnerabilities_discovered: Vec::new(),
            campaign_status: CampaignStatus::Active,
        };

        // Add to active campaigns
        {
            let mut campaigns = self.current_campaigns.write().await;
            campaigns.push(campaign.clone());
        }

        // Execute attack rounds
        for round in 0..campaign_config.max_rounds {
            info!("🔴 Attack round {}/{}", round + 1, campaign_config.max_rounds);
            
            let round_result = self.execute_attack_round(
                mzekezeke_engine,
                &campaign_config,
                round
            ).await?;

            campaign.attack_rounds.push(round_result.clone());
            campaign.total_attacks += round_result.attacks_executed;
            campaign.successful_attacks += round_result.successful_attacks;
            campaign.vulnerabilities_discovered.extend(round_result.vulnerabilities_found);

            // Check if we should continue based on success rate
            if round > 3 {
                let success_rate = campaign.successful_attacks as f64 / campaign.total_attacks as f64;
                if success_rate < 0.1 {
                    info!("🛡️ System appears resilient, ending campaign early");
                    break;
                }
            }
        }

        // Finalize campaign
        campaign.end_time = Some(Utc::now());
        campaign.campaign_status = CampaignStatus::Completed;

        // Update vulnerability database
        {
            let mut vuln_db = self.vulnerability_database.write().await;
            for vulnerability in &campaign.vulnerabilities_discovered {
                vuln_db.add_vulnerability(vulnerability.clone());
            }
        }

        // Remove from active campaigns
        {
            let mut campaigns = self.current_campaigns.write().await;
            campaigns.retain(|c| c.id != campaign_id);
        }

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "adversarial_campaign").await?;

        // Record attack history
        self.record_campaign(&campaign).await;

        let result = AttackCampaignResult {
            campaign_id,
            duration: Utc::now() - start_time,
            total_attacks: campaign.total_attacks,
            successful_attacks: campaign.successful_attacks,
            success_rate: campaign.successful_attacks as f64 / campaign.total_attacks.max(1) as f64,
            vulnerabilities_discovered: campaign.vulnerabilities_discovered.len(),
            most_critical_vulnerabilities: Self::get_critical_vulnerabilities(&campaign.vulnerabilities_discovered),
            atp_cost: campaign_cost,
        };

        info!("✅ Attack campaign complete: {}/{} attacks successful, {} vulnerabilities found", 
              result.successful_attacks, result.total_attacks, result.vulnerabilities_discovered);

        Ok(result)
    }

    /// Execute single attack round
    async fn execute_attack_round(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
        campaign_config: &AttackCampaignConfig,
        round: usize,
    ) -> Result<AttackRoundResult> {
        let mut attacks_executed = 0;
        let mut successful_attacks = 0;
        let mut vulnerabilities_found = Vec::new();
        let mut attack_results = Vec::new();

        // Select attack strategies for this round
        let strategies = self.select_attack_strategies(campaign_config, round);

        for strategy in strategies {
            let attack_result = self.execute_attack_strategy(
                mzekezeke_engine,
                &strategy,
                round
            ).await?;

            attacks_executed += 1;
            if attack_result.success {
                successful_attacks += 1;
                vulnerabilities_found.extend(attack_result.vulnerabilities_discovered);
            }

            attack_results.push(attack_result);
        }

        Ok(AttackRoundResult {
            round_number: round,
            attacks_executed,
            successful_attacks,
            vulnerabilities_found,
            attack_results,
            round_duration: Duration::seconds(5), // Simplified
        })
    }

    /// Execute specific attack strategy
    async fn execute_attack_strategy(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
        strategy: &AttackStrategy,
        round: usize,
    ) -> Result<AttackResult> {
        debug!("🎯 Executing attack strategy: {:?}", strategy.strategy_type);

        let attack_id = Uuid::new_v4();
        let start_time = Utc::now();

        let mut vulnerabilities_discovered = Vec::new();
        let mut success = false;

        match strategy.strategy_type {
            AttackStrategyType::WeakEvidenceExploitation => {
                let result = self.attack_weak_evidence(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
            AttackStrategyType::OverconfidenceAmplification => {
                let result = self.attack_overconfidence(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
            AttackStrategyType::CircularReasoningInduction => {
                let result = self.attack_circular_reasoning(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
            AttackStrategyType::BiasAmplification => {
                let result = self.attack_bias_amplification(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
            AttackStrategyType::TemporalInconsistency => {
                let result = self.attack_temporal_consistency(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
            AttackStrategyType::EvidenceCollusion => {
                let result = self.attack_evidence_collusion(mzekezeke_engine).await?;
                success = result.success;
                vulnerabilities_discovered = result.vulnerabilities;
            },
        }

        Ok(AttackResult {
            attack_id,
            strategy: strategy.clone(),
            success,
            vulnerabilities_discovered,
            confidence_impact: if success { 0.1 } else { 0.0 },
            attack_duration: Utc::now() - start_time,
            round_number: round,
        })
    }

    /// Attack strategy: Exploit weak evidence
    async fn attack_weak_evidence(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create contradictory evidence with low quality
        let weak_evidence = vec![
            Evidence {
                id: format!("weak_attack_{}", Uuid::new_v4()),
                source_type: "UnverifiedBlog".to_string(),
                content: "Contradictory weak evidence".to_string(),
                affected_hypotheses: vec!["test_hypothesis".to_string()],
                factual_accuracy: 0.2,
                contextual_relevance: 0.3,
                temporal_validity: 0.1,
                logical_consistency: 0.2,
                empirical_support: 0.1,
                relevance: 0.4,
                complexity: 0.8,
                timestamp: Utc::now(),
            }
        ];

        // Try to process weak evidence
        let result = mzekezeke_engine.process_evidence_batch(weak_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(update_result) => {
                // Check if system accepted weak evidence
                if update_result.beliefs_updated > 0 {
                    vulnerabilities.push(Vulnerability {
                        id: Uuid::new_v4(),
                        vulnerability_type: VulnerabilityType::WeakEvidence,
                        severity: VulnerabilitySeverity::Medium,
                        description: "System accepted evidence below quality threshold".to_string(),
                        affected_nodes: vec!["test_hypothesis".to_string()],
                        discovery_timestamp: Utc::now(),
                        exploitation_method: "Injected low-quality contradictory evidence".to_string(),
                        confidence_impact: 0.1,
                        remediation_suggestions: vec![
                            "Increase minimum quality threshold".to_string(),
                            "Add stricter source validation".to_string(),
                        ],
                    });
                    true
                } else {
                    false
                }
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Attack strategy: Amplify overconfidence
    async fn attack_overconfidence(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create multiple pieces of confirming evidence from similar sources
        let confirming_evidence = vec![
            Evidence {
                id: format!("confirm_attack_{}", Uuid::new_v4()),
                source_type: "SimilarSource".to_string(),
                content: "Highly confirming evidence".to_string(),
                affected_hypotheses: vec!["overconfidence_test".to_string()],
                factual_accuracy: 0.95,
                contextual_relevance: 0.95,
                temporal_validity: 0.95,
                logical_consistency: 0.95,
                empirical_support: 0.95,
                relevance: 0.95,
                complexity: 0.2,
                timestamp: Utc::now(),
            },
            Evidence {
                id: format!("confirm_attack_{}", Uuid::new_v4()),
                source_type: "SimilarSource".to_string(),
                content: "Another highly confirming evidence".to_string(),
                affected_hypotheses: vec!["overconfidence_test".to_string()],
                factual_accuracy: 0.96,
                contextual_relevance: 0.94,
                temporal_validity: 0.97,
                logical_consistency: 0.93,
                empirical_support: 0.98,
                relevance: 0.96,
                complexity: 0.2,
                timestamp: Utc::now(),
            }
        ];

        let result = mzekezeke_engine.process_evidence_batch(confirming_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(_) => {
                // Check if belief became overconfident
                if let Ok(Some(belief)) = mzekezeke_engine.get_belief("overconfidence_test").await {
                    if belief.confidence > 0.95 {
                        vulnerabilities.push(Vulnerability {
                            id: Uuid::new_v4(),
                            vulnerability_type: VulnerabilityType::OverConfidence,
                            severity: VulnerabilitySeverity::High,
                            description: "System developed overconfidence from similar sources".to_string(),
                            affected_nodes: vec!["overconfidence_test".to_string()],
                            discovery_timestamp: Utc::now(),
                            exploitation_method: "Multiple confirming evidence from similar sources".to_string(),
                            confidence_impact: 0.2,
                            remediation_suggestions: vec![
                                "Implement source diversity requirements".to_string(),
                                "Add confidence ceiling based on source similarity".to_string(),
                            ],
                        });
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Attack strategy: Induce circular reasoning
    async fn attack_circular_reasoning(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create evidence that creates circular dependencies
        let circular_evidence = vec![
            Evidence {
                id: format!("circular_a_{}", Uuid::new_v4()),
                source_type: "CircularSource".to_string(),
                content: "A supports B".to_string(),
                affected_hypotheses: vec!["hypothesis_a".to_string(), "hypothesis_b".to_string()],
                factual_accuracy: 0.8,
                contextual_relevance: 0.8,
                temporal_validity: 0.8,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.8,
                complexity: 0.5,
                timestamp: Utc::now(),
            },
            Evidence {
                id: format!("circular_b_{}", Uuid::new_v4()),
                source_type: "CircularSource".to_string(),
                content: "B supports C".to_string(),
                affected_hypotheses: vec!["hypothesis_b".to_string(), "hypothesis_c".to_string()],
                factual_accuracy: 0.8,
                contextual_relevance: 0.8,
                temporal_validity: 0.8,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.8,
                complexity: 0.5,
                timestamp: Utc::now(),
            },
            Evidence {
                id: format!("circular_c_{}", Uuid::new_v4()),
                source_type: "CircularSource".to_string(),
                content: "C supports A".to_string(),
                affected_hypotheses: vec!["hypothesis_c".to_string(), "hypothesis_a".to_string()],
                factual_accuracy: 0.8,
                contextual_relevance: 0.8,
                temporal_validity: 0.8,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.8,
                complexity: 0.5,
                timestamp: Utc::now(),
            },
        ];

        let result = mzekezeke_engine.process_evidence_batch(circular_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(update_result) => {
                if update_result.beliefs_updated >= 3 {
                    // Check if circular reasoning was created
                    vulnerabilities.push(Vulnerability {
                        id: Uuid::new_v4(),
                        vulnerability_type: VulnerabilityType::CircularReasoning,
                        severity: VulnerabilitySeverity::High,
                        description: "System created circular reasoning dependencies".to_string(),
                        affected_nodes: vec!["hypothesis_a".to_string(), "hypothesis_b".to_string(), "hypothesis_c".to_string()],
                        discovery_timestamp: Utc::now(),
                        exploitation_method: "Injected mutually supporting evidence".to_string(),
                        confidence_impact: 0.15,
                        remediation_suggestions: vec![
                            "Implement cycle detection in belief network".to_string(),
                            "Add dependency analysis before accepting evidence".to_string(),
                        ],
                    });
                    true
                } else {
                    false
                }
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Attack strategy: Amplify bias
    async fn attack_bias_amplification(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create evidence that reinforces existing biases
        let bias_evidence = vec![
            Evidence {
                id: format!("bias_attack_{}", Uuid::new_v4()),
                source_type: "BiasedSource".to_string(),
                content: "Evidence that confirms existing bias".to_string(),
                affected_hypotheses: vec!["biased_hypothesis".to_string()],
                factual_accuracy: 0.6,
                contextual_relevance: 0.9,
                temporal_validity: 0.8,
                logical_consistency: 0.5,
                empirical_support: 0.4,
                relevance: 0.9,
                complexity: 0.3,
                timestamp: Utc::now(),
            }
        ];

        let result = mzekezeke_engine.process_evidence_batch(bias_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(_) => {
                // Simplified bias detection
                vulnerabilities.push(Vulnerability {
                    id: Uuid::new_v4(),
                    vulnerability_type: VulnerabilityType::BiasAmplification,
                    severity: VulnerabilitySeverity::Medium,
                    description: "System potentially amplified existing bias".to_string(),
                    affected_nodes: vec!["biased_hypothesis".to_string()],
                    discovery_timestamp: Utc::now(),
                    exploitation_method: "Injected bias-confirming evidence".to_string(),
                    confidence_impact: 0.08,
                    remediation_suggestions: vec![
                        "Implement bias detection algorithms".to_string(),
                        "Add counter-evidence requirements".to_string(),
                    ],
                });
                true
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Attack strategy: Create temporal inconsistencies
    async fn attack_temporal_consistency(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create evidence with inconsistent temporal claims
        let temporal_evidence = vec![
            Evidence {
                id: format!("temporal_old_{}", Uuid::new_v4()),
                source_type: "TemporalSource".to_string(),
                content: "Old evidence".to_string(),
                affected_hypotheses: vec!["temporal_hypothesis".to_string()],
                factual_accuracy: 0.9,
                contextual_relevance: 0.5,
                temporal_validity: 0.2, // Low temporal validity
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.7,
                complexity: 0.4,
                timestamp: Utc::now() - Duration::days(365),
            },
            Evidence {
                id: format!("temporal_new_{}", Uuid::new_v4()),
                source_type: "TemporalSource".to_string(),
                content: "Contradictory new evidence".to_string(),
                affected_hypotheses: vec!["temporal_hypothesis".to_string()],
                factual_accuracy: 0.9,
                contextual_relevance: 0.9,
                temporal_validity: 0.9,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.9,
                complexity: 0.4,
                timestamp: Utc::now(),
            }
        ];

        let result = mzekezeke_engine.process_evidence_batch(temporal_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(_) => {
                vulnerabilities.push(Vulnerability {
                    id: Uuid::new_v4(),
                    vulnerability_type: VulnerabilityType::TemporalInconsistency,
                    severity: VulnerabilitySeverity::Medium,
                    description: "System failed to handle temporal inconsistencies properly".to_string(),
                    affected_nodes: vec!["temporal_hypothesis".to_string()],
                    discovery_timestamp: Utc::now(),
                    exploitation_method: "Injected temporally inconsistent evidence".to_string(),
                    confidence_impact: 0.12,
                    remediation_suggestions: vec![
                        "Improve temporal decay mechanisms".to_string(),
                        "Add temporal consistency checks".to_string(),
                    ],
                });
                true
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Attack strategy: Create evidence collusion
    async fn attack_evidence_collusion(
        &self,
        mzekezeke_engine: &crate::mzekezeke::MzekezekeEngine,
    ) -> Result<AttackStrategyResult> {
        // Create evidence that appears independent but is actually colluding
        let collusion_evidence = vec![
            Evidence {
                id: format!("collusion_1_{}", Uuid::new_v4()),
                source_type: "IndependentSource1".to_string(),
                content: "Supporting evidence 1".to_string(),
                affected_hypotheses: vec!["collusion_hypothesis".to_string()],
                factual_accuracy: 0.8,
                contextual_relevance: 0.8,
                temporal_validity: 0.8,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.8,
                complexity: 0.3,
                timestamp: Utc::now(),
            },
            Evidence {
                id: format!("collusion_2_{}", Uuid::new_v4()),
                source_type: "IndependentSource2".to_string(),
                content: "Supporting evidence 2 (actually colluding)".to_string(),
                affected_hypotheses: vec!["collusion_hypothesis".to_string()],
                factual_accuracy: 0.8,
                contextual_relevance: 0.8,
                temporal_validity: 0.8,
                logical_consistency: 0.8,
                empirical_support: 0.8,
                relevance: 0.8,
                complexity: 0.3,
                timestamp: Utc::now(),
            }
        ];

        let result = mzekezeke_engine.process_evidence_batch(collusion_evidence).await;
        
        let mut vulnerabilities = Vec::new();
        let success = match result {
            Ok(_) => {
                vulnerabilities.push(Vulnerability {
                    id: Uuid::new_v4(),
                    vulnerability_type: VulnerabilityType::EvidenceCollusion,
                    severity: VulnerabilitySeverity::High,
                    description: "System failed to detect evidence collusion".to_string(),
                    affected_nodes: vec!["collusion_hypothesis".to_string()],
                    discovery_timestamp: Utc::now(),
                    exploitation_method: "Injected colluding evidence from seemingly independent sources".to_string(),
                    confidence_impact: 0.18,
                    remediation_suggestions: vec![
                        "Implement source independence verification".to_string(),
                        "Add collusion detection algorithms".to_string(),
                        "Cross-reference evidence patterns".to_string(),
                    ],
                });
                true
            },
            Err(_) => false,
        };

        Ok(AttackStrategyResult {
            success,
            vulnerabilities,
        })
    }

    /// Get discovered vulnerabilities
    pub async fn get_discovered_vulnerabilities(&self) -> Vec<Vulnerability> {
        let vuln_db = self.vulnerability_database.read().await;
        vuln_db.get_all_vulnerabilities()
    }

    /// Get vulnerability statistics
    pub async fn get_vulnerability_statistics(&self) -> VulnerabilityStatistics {
        let vuln_db = self.vulnerability_database.read().await;
        let history = self.attack_history.read().await;

        let vulnerabilities = vuln_db.get_all_vulnerabilities();
        let total_attacks = history.iter().map(|r| r.total_attacks).sum();
        let successful_attacks = history.iter().map(|r| r.successful_attacks).sum();

        VulnerabilityStatistics {
            total_vulnerabilities: vulnerabilities.len(),
            high_severity: vulnerabilities.iter().filter(|v| matches!(v.severity, VulnerabilitySeverity::High)).count(),
            medium_severity: vulnerabilities.iter().filter(|v| matches!(v.severity, VulnerabilitySeverity::Medium)).count(),
            low_severity: vulnerabilities.iter().filter(|v| matches!(v.severity, VulnerabilitySeverity::Low)).count(),
            total_attack_campaigns: history.len(),
            total_attacks,
            successful_attacks,
            overall_success_rate: if total_attacks > 0 { successful_attacks as f64 / total_attacks as f64 } else { 0.0 },
            most_common_vulnerability: Self::get_most_common_vulnerability_type(&vulnerabilities),
        }
    }

    /// Initialize attack strategies
    fn initialize_attack_strategies(config: &DiggidenAdversarialConfig) -> Vec<AttackStrategy> {
        vec![
            AttackStrategy {
                strategy_type: AttackStrategyType::WeakEvidenceExploitation,
                priority: config.strategy_priorities.get("weak_evidence").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
            AttackStrategy {
                strategy_type: AttackStrategyType::OverconfidenceAmplification,
                priority: config.strategy_priorities.get("overconfidence").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
            AttackStrategy {
                strategy_type: AttackStrategyType::CircularReasoningInduction,
                priority: config.strategy_priorities.get("circular_reasoning").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
            AttackStrategy {
                strategy_type: AttackStrategyType::BiasAmplification,
                priority: config.strategy_priorities.get("bias_amplification").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
            AttackStrategy {
                strategy_type: AttackStrategyType::TemporalInconsistency,
                priority: config.strategy_priorities.get("temporal_inconsistency").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
            AttackStrategy {
                strategy_type: AttackStrategyType::EvidenceCollusion,
                priority: config.strategy_priorities.get("evidence_collusion").copied().unwrap_or(1.0),
                success_rate: 0.0,
                last_used: None,
            },
        ]
    }

    /// Select attack strategies for a round
    fn select_attack_strategies(
        &self,
        campaign_config: &AttackCampaignConfig,
        round: usize,
    ) -> Vec<AttackStrategy> {
        let mut rng = thread_rng();
        let mut strategies = self.attack_strategies.clone();
        
        // Shuffle for variety
        strategies.shuffle(&mut rng);
        
        // Select based on intensity
        let num_strategies = match campaign_config.intensity {
            AttackIntensity::Low => 1,
            AttackIntensity::Medium => 2,
            AttackIntensity::High => 3,
            AttackIntensity::Maximum => 4,
        };
        
        strategies.into_iter().take(num_strategies).collect()
    }

    /// Calculate campaign cost
    fn calculate_campaign_cost(&self, config: &AttackCampaignConfig) -> u64 {
        let base_cost = 200u64;
        let round_cost = config.max_rounds as u64 * 100;
        let intensity_multiplier = match config.intensity {
            AttackIntensity::Low => 1.0,
            AttackIntensity::Medium => 1.5,
            AttackIntensity::High => 2.0,
            AttackIntensity::Maximum => 3.0,
        };
        
        ((base_cost + round_cost) as f64 * intensity_multiplier) as u64
    }

    /// Get critical vulnerabilities
    fn get_critical_vulnerabilities(vulnerabilities: &[Vulnerability]) -> Vec<Vulnerability> {
        vulnerabilities.iter()
            .filter(|v| matches!(v.severity, VulnerabilitySeverity::High))
            .cloned()
            .collect()
    }

    /// Get most common vulnerability type
    fn get_most_common_vulnerability_type(vulnerabilities: &[Vulnerability]) -> Option<VulnerabilityType> {
        let mut type_counts = HashMap::new();
        
        for vuln in vulnerabilities {
            *type_counts.entry(vuln.vulnerability_type.clone()).or_insert(0) += 1;
        }
        
        type_counts.into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(vuln_type, _)| vuln_type)
    }

    /// Record campaign history
    async fn record_campaign(&self, campaign: &AttackCampaign) {
        let mut history = self.attack_history.write().await;
        
        let record = AttackRecord {
            campaign_id: campaign.id,
            timestamp: campaign.start_time,
            total_attacks: campaign.total_attacks,
            successful_attacks: campaign.successful_attacks,
            vulnerabilities_found: campaign.vulnerabilities_discovered.len(),
            duration: campaign.end_time.unwrap_or(Utc::now()) - campaign.start_time,
        };
        
        history.push(record);
        
        // Keep recent history
        if history.len() > 100 {
            history.drain(0..50);
        }
    }
}

/// Vulnerability database
#[derive(Debug)]
pub struct VulnerabilityDatabase {
    vulnerabilities: HashMap<Uuid, Vulnerability>,
    discovered_count: usize,
}

impl VulnerabilityDatabase {
    fn new() -> Self {
        Self {
            vulnerabilities: HashMap::new(),
            discovered_count: 0,
        }
    }

    fn add_vulnerability(&mut self, vulnerability: Vulnerability) {
        self.vulnerabilities.insert(vulnerability.id, vulnerability);
        self.discovered_count += 1;
    }

    fn get_all_vulnerabilities(&self) -> Vec<Vulnerability> {
        self.vulnerabilities.values().cloned().collect()
    }
}

/// Attack strategy definition
#[derive(Debug, Clone)]
pub struct AttackStrategy {
    pub strategy_type: AttackStrategyType,
    pub priority: f64,
    pub success_rate: f64,
    pub last_used: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttackStrategyType {
    WeakEvidenceExploitation,
    OverconfidenceAmplification,
    CircularReasoningInduction,
    BiasAmplification,
    TemporalInconsistency,
    EvidenceCollusion,
}

/// Vulnerability representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vulnerability {
    pub id: Uuid,
    pub vulnerability_type: VulnerabilityType,
    pub severity: VulnerabilitySeverity,
    pub description: String,
    pub affected_nodes: Vec<String>,
    pub discovery_timestamp: DateTime<Utc>,
    pub exploitation_method: String,
    pub confidence_impact: f64,
    pub remediation_suggestions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum VulnerabilityType {
    WeakEvidence,
    OverConfidence,
    CircularReasoning,
    BiasAmplification,
    TemporalInconsistency,
    EvidenceCollusion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VulnerabilitySeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Attack campaign configuration
#[derive(Debug, Clone)]
pub struct AttackCampaignConfig {
    pub name: String,
    pub max_rounds: usize,
    pub intensity: AttackIntensity,
    pub target_success_rate: f64,
}

#[derive(Debug, Clone)]
pub enum AttackIntensity {
    Low,
    Medium,
    High,
    Maximum,
}

/// Attack campaign tracking
#[derive(Debug, Clone)]
pub struct AttackCampaign {
    pub id: Uuid,
    pub name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub attack_rounds: Vec<AttackRoundResult>,
    pub total_attacks: usize,
    pub successful_attacks: usize,
    pub vulnerabilities_discovered: Vec<Vulnerability>,
    pub campaign_status: CampaignStatus,
}

#[derive(Debug, Clone)]
pub enum CampaignStatus {
    Active,
    Completed,
    Aborted,
}

/// Results and statistics
#[derive(Debug, Clone)]
pub struct AttackCampaignResult {
    pub campaign_id: Uuid,
    pub duration: Duration,
    pub total_attacks: usize,
    pub successful_attacks: usize,
    pub success_rate: f64,
    pub vulnerabilities_discovered: usize,
    pub most_critical_vulnerabilities: Vec<Vulnerability>,
    pub atp_cost: u64,
}

#[derive(Debug, Clone)]
pub struct AttackRoundResult {
    pub round_number: usize,
    pub attacks_executed: usize,
    pub successful_attacks: usize,
    pub vulnerabilities_found: Vec<Vulnerability>,
    pub attack_results: Vec<AttackResult>,
    pub round_duration: Duration,
}

#[derive(Debug, Clone)]
pub struct AttackResult {
    pub attack_id: Uuid,
    pub strategy: AttackStrategy,
    pub success: bool,
    pub vulnerabilities_discovered: Vec<Vulnerability>,
    pub confidence_impact: f64,
    pub attack_duration: Duration,
    pub round_number: usize,
}

#[derive(Debug)]
struct AttackStrategyResult {
    success: bool,
    vulnerabilities: Vec<Vulnerability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityStatistics {
    pub total_vulnerabilities: usize,
    pub high_severity: usize,
    pub medium_severity: usize,
    pub low_severity: usize,
    pub total_attack_campaigns: usize,
    pub total_attacks: usize,
    pub successful_attacks: usize,
    pub overall_success_rate: f64,
    pub most_common_vulnerability: Option<VulnerabilityType>,
}

#[derive(Debug, Clone)]
struct AttackRecord {
    campaign_id: Uuid,
    timestamp: DateTime<Utc>,
    total_attacks: usize,
    successful_attacks: usize,
    vulnerabilities_found: usize,
    duration: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[tokio::test]
    async fn test_diggiden_engine_creation() {
        let atp_manager = Arc::new(AtpManager::new(
            10000, 50000, 1000, 100,
            AtpCosts {
                basic_query: 100,
                fuzzy_operation: 50,
                uncertainty_processing: 25,
                repository_call: 100,
                synthesis_operation: 200,
                verification_step: 150,
                dreaming_cycle: 500,
                gray_area_processing: 75,
                truth_spectrum_analysis: 300,
                lactic_fermentation: 10,
            }
        ));

        let config = DiggidenAdversarialConfig::default();
        let engine = DiggidenEngine::new(config, atp_manager);

        let stats = engine.get_vulnerability_statistics().await;
        assert_eq!(stats.total_vulnerabilities, 0);
    }

    #[tokio::test]
    async fn test_attack_campaign() {
        let atp_manager = Arc::new(AtpManager::new(
            10000, 50000, 1000, 100,
            AtpCosts {
                basic_query: 100,
                fuzzy_operation: 50,
                uncertainty_processing: 25,
                repository_call: 100,
                synthesis_operation: 200,
                verification_step: 150,
                dreaming_cycle: 500,
                gray_area_processing: 75,
                truth_spectrum_analysis: 300,
                lactic_fermentation: 10,
            }
        ));

        let diggiden_config = DiggidenAdversarialConfig::default();
        let mzekezeke_config = crate::config::MzekezekeBayesianConfig::default();
        
        let diggiden_engine = DiggidenEngine::new(diggiden_config, atp_manager.clone());
        let mzekezeke_engine = crate::mzekezeke::MzekezekeEngine::new(mzekezeke_config, atp_manager);

        let campaign_config = AttackCampaignConfig {
            name: "Test Campaign".to_string(),
            max_rounds: 2,
            intensity: AttackIntensity::Low,
            target_success_rate: 0.3,
        };

        let result = diggiden_engine.launch_attack_campaign(&mzekezeke_engine, campaign_config).await.unwrap();
        assert!(result.total_attacks > 0);
    }
}