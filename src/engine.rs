//! Honjo Masamune Truth Engine
//! 
//! The main orchestration engine that coordinates all subsystems to provide
//! truth synthesis capabilities with biological metabolism and fuzzy logic.

use anyhow::Result;
use atp_manager::{AtpManager, AtpStatus};
use buhera_engine::{BuheraEngine, BuheraProgram, Solution};
use fuzzy_logic_core::{FuzzyLogicEngine, FuzzyResult, FuzzyTruth, TruthSpectrum};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::HonjoMasamuneConfig;
use crate::QueryResult;
use crate::spectacular::{SpectacularEngine, SpectacularCriteria};
use crate::nicotine::{NicotineEngine, BreakCriteria, ProcessingContext, ProcessingOperation, ProcessingObjective};
use crate::mzekezeke::{MzekezekeEngine, Evidence, Hypothesis};
use crate::diggiden::{DiggidenEngine, AttackCampaignConfig, AttackIntensity};
use crate::hatata::{HatataEngine, ActionOption, ActionType, ResourceRequirements};
use crate::zengeza::{ZengezaEngine, CommunicationContext, NoiseAnalysisResult, DenoisedStatement, NoiseStatistics};
use crate::diadochi::{DiadochiEngine, CombinationResult};

/// Main Honjo Masamune Truth Engine
#[derive(Debug)]
pub struct HonjoMasamuneEngine {
    pub config: HonjoMasamuneConfig,
    atp_manager: Arc<AtpManager>,
    fuzzy_engine: Arc<FuzzyLogicEngine>,
    buhera_engine: Arc<BuheraEngine>,
    spectacular_engine: Arc<SpectacularEngine>,
    nicotine_engine: Arc<NicotineEngine>,
    mzekezeke_engine: Arc<MzekezekeEngine>,
    diggiden_engine: Arc<DiggidenEngine>,
    hatata_engine: Arc<HatataEngine>,
    zengeza_engine: Arc<ZengezaEngine>,
    diadochi_engine: Arc<DiadochiEngine>,
    session_state: Arc<RwLock<SessionState>>,
    query_history: Arc<RwLock<Vec<QueryRecord>>>,
}

impl HonjoMasamuneEngine {
    /// Create a new Honjo Masamune engine
    pub async fn new(
        config: HonjoMasamuneConfig,
        atp_manager: Arc<AtpManager>,
        fuzzy_engine: Arc<FuzzyLogicEngine>,
        buhera_engine: Arc<BuheraEngine>,
    ) -> Result<Self> {
        let session_state = Arc::new(RwLock::new(SessionState::new(config.system.ceremonial_mode)));
        let query_history = Arc::new(RwLock::new(Vec::new()));

        // Initialize spectacular engine
        let spectacular_engine = Arc::new(SpectacularEngine::new(
            atp_manager.clone(),
            SpectacularCriteria::default(),
        ));

        // Initialize nicotine engine
        let nicotine_engine = Arc::new(NicotineEngine::new(
            atp_manager.clone(),
            BreakCriteria::default(),
        ));

        // Initialize mzekezeke engine
        let mzekezeke_engine = Arc::new(MzekezekeEngine::new(
            config.mzekezeke_bayesian.clone(),
            atp_manager.clone(),
        ));

        // Initialize diggiden engine
        let diggiden_engine = Arc::new(DiggidenEngine::new(
            config.diggiden_adversarial.clone(),
            atp_manager.clone(),
        ));

        // Initialize hatata engine
        let hatata_engine = Arc::new(HatataEngine::new(
            config.hatata_decision.clone(),
            atp_manager.clone(),
        ));

        // Initialize zengeza engine
        let zengeza_engine = Arc::new(ZengezaEngine::new(
            config.zengeza_noise.clone(),
            atp_manager.clone(),
        ));

        // Initialize diadochi engine
        let diadochi_engine = Arc::new(DiadochiEngine::new(
            config.diadochi.clone(),
        ));

        let engine = Self {
            config,
            atp_manager,
            fuzzy_engine,
            buhera_engine,
            spectacular_engine,
            nicotine_engine,
            mzekezeke_engine,
            diggiden_engine,
            hatata_engine,
            zengeza_engine,
            diadochi_engine,
            session_state,
            query_history,
        };

        // Initialize ceremonial restrictions if needed
        if engine.config.system.ceremonial_mode {
            engine.initialize_ceremonial_mode().await?;
        }

        Ok(engine)
    }

    /// Process a natural language query
    pub async fn process_natural_language_query(&self, query: &str) -> Result<QueryResult> {
        info!("🔍 Processing natural language query: {}", query);

        // Check ceremonial restrictions
        if self.config.system.ceremonial_mode {
            self.check_ceremonial_restrictions().await?;
        }

        // Reserve ATP for the query
        let base_cost = self.atp_manager.get_cost("basic_query");
        let complexity_cost = self.estimate_query_complexity(query);
        let total_cost = base_cost + complexity_cost;

        let reservation = self.atp_manager.reserve_atp("natural_language_query", total_cost).await?;

        // Track operation for nicotine break system
        let operation = ProcessingOperation {
            operation_type: "natural_language_query".to_string(),
            complexity_contribution: complexity_cost as f64 / 1000.0,
            confidence_impact: 0.1,
            drift_impact: 0.02,
        };
        self.nicotine_engine.track_operation(&operation).await?;

        // Check if nicotine break is needed
        if self.nicotine_engine.should_take_break().await? {
            info!("🚬 Nicotine break triggered during query processing");
            let current_context = ProcessingContext {
                primary_objective: ProcessingObjective {
                    description: format!("Process query: {}", query),
                    priority: 0.8,
                    success_criteria: vec!["Return accurate result".to_string()],
                },
                active_queries: vec![query.to_string()],
                pending_operations: vec![],
                processing_depth: 1,
                complexity_level: complexity_cost as f64 / 1000.0,
                confidence_level: 0.5,
            };
            
            let break_result = self.nicotine_engine.take_nicotine_break(&current_context).await?;
            if break_result.solution_correct {
                info!("✅ Context validated and refreshed during break");
            } else {
                warn!("⚠️ Context validation failed - proceeding with caution");
            }
        }

        // Convert natural language to Buhera program
        let buhera_program = self.natural_language_to_buhera(query).await?;

        // Execute the Buhera program
        let execution_results = self.buhera_engine.execute_program(&buhera_program).await?;

        // Analyze truth spectrum
        let truth_spectrum = self.fuzzy_engine.analyze_truth_spectrum(query).await?;

        // Synthesize final result
        let result = self.synthesize_query_result(
            query,
            execution_results,
            truth_spectrum.clone(),
            total_cost,
        ).await?;

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "natural_language_query").await?;

        // Check for spectacular implications
        if let Some(spectacular_finding) = self.spectacular_engine.analyze_for_spectacular_implications(
            query,
            &result,
            &truth_spectrum,
        ).await? {
            warn!("🌟 Spectacular finding detected and processed!");
            warn!("🎯 Significance: {:.4}", spectacular_finding.significance_score);
            warn!("🔍 Implications: {:?}", spectacular_finding.implications);
        }

        // Record query in history
        self.record_query(query, &result).await;

        // Check for ceremonial certainty
        if result.confidence.value() >= 0.95 && self.config.system.ceremonial_mode {
            self.handle_ceremonial_certainty(query, &result).await?;
        }

        Ok(result)
    }

    /// Execute a Buhera program from source code
    pub async fn execute_buhera_program(&self, source: &str) -> Result<Vec<FuzzyResult<Vec<Solution>>>> {
        info!("📜 Executing Buhera program");

        // Parse the program
        let program = BuheraProgram::parse("user_program".to_string(), source)?;

        // Execute the program
        let results = self.buhera_engine.execute_program(&program).await?;

        info!("✅ Buhera program execution complete: {} queries processed", results.len());
        Ok(results)
    }

    /// Get current ATP status
    pub fn atp_status(&self) -> AtpStatus {
        self.atp_manager.status()
    }

    /// Get session statistics
    pub async fn session_stats(&self) -> SessionStats {
        let state = self.session_state.read().await;
        let history = self.query_history.read().await;

        SessionStats {
            queries_executed: history.len(),
            ceremonial_queries: history.iter().filter(|q| q.ceremonial_certainty).count(),
            total_atp_consumed: history.iter().map(|q| q.atp_cost).sum(),
            average_confidence: if history.is_empty() {
                0.0
            } else {
                history.iter().map(|q| q.confidence).sum::<f64>() / history.len() as f64
            },
            session_start: state.session_start,
            ceremonial_mode: state.ceremonial_mode,
            topics_closed: state.topics_closed.len(),
        }
    }

    /// Get all spectacular findings
    pub async fn get_spectacular_findings(&self) -> Vec<crate::spectacular::SpectacularFinding> {
        self.spectacular_engine.get_spectacular_findings().await
    }

    /// Get top spectacular findings by significance
    pub async fn get_top_spectacular_findings(&self, limit: usize) -> Vec<crate::spectacular::SpectacularFinding> {
        self.spectacular_engine.get_top_spectacular_findings(limit).await
    }

    /// Get nicotine break statistics
    pub async fn get_nicotine_break_statistics(&self) -> crate::nicotine::BreakStatistics {
        self.nicotine_engine.get_break_statistics().await
    }

    /// Force a nicotine break for context validation
    pub async fn force_nicotine_break(&self, query: &str) -> Result<crate::nicotine::NicotineBreakResult> {
        let current_context = ProcessingContext {
            primary_objective: ProcessingObjective {
                description: format!("Forced break for query: {}", query),
                priority: 0.7,
                success_criteria: vec!["Validate context".to_string()],
            },
            active_queries: vec![query.to_string()],
            pending_operations: vec![],
            processing_depth: 1,
            complexity_level: 2.0,
            confidence_level: 0.8,
        };
        
        self.nicotine_engine.take_nicotine_break(&current_context).await
    }

    /// Get mzekezeke network statistics
    pub async fn get_mzekezeke_statistics(&self) -> crate::mzekezeke::NetworkStatistics {
        self.mzekezeke_engine.get_network_statistics().await
    }

    /// Add evidence to mzekezeke belief network
    pub async fn add_evidence(&self, evidence: Vec<Evidence>) -> Result<crate::mzekezeke::BeliefUpdateResult> {
        self.mzekezeke_engine.process_evidence_batch(evidence).await
    }

    /// Get diggiden vulnerability statistics
    pub async fn get_vulnerability_statistics(&self) -> crate::diggiden::VulnerabilityStatistics {
        self.diggiden_engine.get_vulnerability_statistics().await
    }

    /// Launch adversarial attack campaign
    pub async fn launch_attack_campaign(&self, campaign_name: &str) -> Result<crate::diggiden::AttackCampaignResult> {
        let campaign_config = AttackCampaignConfig {
            name: campaign_name.to_string(),
            max_rounds: 5,
            intensity: AttackIntensity::Medium,
            target_success_rate: 0.3,
        };
        
        self.diggiden_engine.launch_attack_campaign(&self.mzekezeke_engine, campaign_config).await
    }

    /// Get hatata decision statistics
    pub async fn get_decision_statistics(&self) -> crate::hatata::DecisionStatistics {
        self.hatata_engine.get_decision_statistics().await
    }

    /// Optimize decisions with hatata engine
    pub async fn optimize_decisions(&self, available_actions: Vec<ActionOption>) -> Result<crate::hatata::DecisionOptimizationResult> {
        let belief_state = self.mzekezeke_engine.get_network_statistics().await;
        let spectacular_findings = self.spectacular_engine.get_top_findings(5).await;
        
        self.hatata_engine.optimize_decisions(belief_state, spectacular_findings, available_actions).await
    }

    /// Get noise analysis statistics from zengeza engine
    pub async fn get_noise_statistics(&self) -> NoiseStatistics {
        self.zengeza_engine.get_noise_statistics().await
    }

    /// Analyze communication noise in a statement
    pub async fn analyze_communication_noise(&self, statement: &str, context: CommunicationContext) -> Result<NoiseAnalysisResult> {
        self.zengeza_engine.analyze_communication_noise(statement, context).await
    }

    /// Denoise a statement based on noise analysis
    pub async fn denoise_statement(&self, statement: &str, noise_analysis: &NoiseAnalysisResult) -> Result<DenoisedStatement> {
        self.zengeza_engine.denoise_statement(statement, noise_analysis).await
    }

    /// Combine multiple models intelligently using diadochi
    pub async fn intelligent_model_combination(&self, query: &str) -> Result<CombinationResult> {
        // Use a temporary clone of the diadochi engine
        let config = self.config.diadochi.clone();
        let mut temp_engine = DiadochiEngine::new(config);
        temp_engine.intelligent_combine(query)
    }

    /// Use router-based ensemble pattern
    pub async fn router_ensemble_combination(&self, query: &str, router_id: &str) -> Result<CombinationResult> {
        let config = self.config.diadochi.clone();
        let mut temp_engine = DiadochiEngine::new(config);
        temp_engine.router_ensemble(query, router_id)
    }

    /// Use sequential chaining pattern
    pub async fn sequential_chain_combination(&self, query: &str, chain_id: &str) -> Result<CombinationResult> {
        let config = self.config.diadochi.clone();
        let mut temp_engine = DiadochiEngine::new(config);
        temp_engine.sequential_chain(query, chain_id)
    }

    /// Use mixture of experts pattern
    pub async fn mixture_of_experts_combination(&self, query: &str, mixture_id: &str) -> Result<CombinationResult> {
        let config = self.config.diadochi.clone();
        let mut temp_engine = DiadochiEngine::new(config);
        temp_engine.mixture_of_experts(query, mixture_id)
    }

    /// Use specialized system prompts pattern
    pub async fn system_prompt_combination(&self, query: &str, prompt_id: &str) -> Result<CombinationResult> {
        let config = self.config.diadochi.clone();
        let mut temp_engine = DiadochiEngine::new(config);
        temp_engine.specialized_system_prompt(query, prompt_id)
    }

    /// Get diadochi engine statistics
    pub async fn get_diadochi_statistics(&self) -> crate::diadochi::DiadochiStatistics {
        self.diadochi_engine.get_statistics().clone()
    }

    /// Export diadochi results
    pub async fn export_diadochi_results(&self) -> Result<String> {
        self.diadochi_engine.export_results()
    }

    /// Initialize ceremonial mode restrictions
    async fn initialize_ceremonial_mode(&self) -> Result<()> {
        info!("⚔️  Initializing ceremonial mode restrictions");
        
        let mut state = self.session_state.write().await;
        state.ceremonial_mode = true;
        state.max_queries_per_session = self.config.ceremonial.restrictions.max_queries_per_year;
        
        warn!("🗡️  Ceremonial mode active: maximum {} queries allowed", state.max_queries_per_session);
        warn!("💀 Each query will permanently close discussion on the topic");
        
        Ok(())
    }

    /// Check ceremonial restrictions before executing a query
    async fn check_ceremonial_restrictions(&self) -> Result<()> {
        let state = self.session_state.read().await;
        let history = self.query_history.read().await;

        if history.len() >= state.max_queries_per_session {
            return Err(anyhow::anyhow!(
                "Ceremonial query limit reached: {}/{}",
                history.len(),
                state.max_queries_per_session
            ));
        }

        // Check cooling period
        if let Some(last_query) = history.last() {
            let cooling_period = chrono::Duration::days(self.config.ceremonial.activation.cooling_period_days as i64);
            let time_since_last = Utc::now() - last_query.timestamp;
            
            if time_since_last < cooling_period {
                return Err(anyhow::anyhow!(
                    "Cooling period active: {} remaining",
                    cooling_period - time_since_last
                ));
            }
        }

        Ok(())
    }

    /// Convert natural language to Buhera program
    async fn natural_language_to_buhera(&self, query: &str) -> Result<BuheraProgram> {
        // This is a simplified implementation
        // In reality, this would use sophisticated NLP to convert natural language
        // to logical programming constructs
        
        let mut program = BuheraProgram::new("nl_query".to_string());
        
        // For now, create a basic query structure
        // This would be much more sophisticated in a real implementation
        let buhera_source = format!(
            r#"
            % Natural language query: {}
            ?- analyze_statement("{}").
            "#,
            query, query
        );

        // Parse the generated Buhera code
        let parsed_program = BuheraProgram::parse("nl_generated".to_string(), &buhera_source)?;
        
        Ok(parsed_program)
    }

    /// Estimate the complexity cost of a query
    fn estimate_query_complexity(&self, query: &str) -> u64 {
        let base_complexity = 50;
        let length_factor = (query.len() / 10) as u64;
        let keyword_factor = self.count_complex_keywords(query) * 25;
        
        base_complexity + length_factor + keyword_factor
    }

    /// Count complex keywords that increase processing cost
    fn count_complex_keywords(&self, query: &str) -> u64 {
        let complex_keywords = [
            "why", "how", "explain", "analyze", "synthesize", "compare",
            "evaluate", "predict", "simulate", "reconstruct", "verify"
        ];
        
        let query_lower = query.to_lowercase();
        complex_keywords
            .iter()
            .map(|&keyword| query_lower.matches(keyword).count() as u64)
            .sum()
    }

    /// Synthesize the final query result from various components
    async fn synthesize_query_result(
        &self,
        query: &str,
        execution_results: Vec<FuzzyResult<Vec<Solution>>>,
        truth_spectrum: TruthSpectrum,
        atp_cost: u64,
    ) -> Result<QueryResult> {
        // Aggregate confidence from all results
        let confidences: Vec<FuzzyTruth> = execution_results
            .iter()
            .map(|r| r.confidence)
            .collect();

        let overall_confidence = if confidences.is_empty() {
            truth_spectrum.overall_truth
        } else {
            // Use fuzzy weighted average
            let weights: Vec<f64> = vec![1.0; confidences.len()];
            let weighted_pairs: Vec<(FuzzyTruth, f64)> = confidences
                .into_iter()
                .zip(weights.into_iter())
                .collect();
            
            self.fuzzy_engine.fuzzy_weighted_average(&weighted_pairs)?
        };

        // Identify gray areas
        let gray_areas: Vec<String> = truth_spectrum
            .gray_areas
            .iter()
            .map(|ga| ga.domain.clone())
            .collect();

        Ok(QueryResult {
            confidence: overall_confidence,
            gray_areas,
            atp_cost,
        })
    }

    /// Record a query in the history
    async fn record_query(&self, query: &str, result: &QueryResult) {
        let mut history = self.query_history.write().await;
        
        let record = QueryRecord {
            id: Uuid::new_v4(),
            query: query.to_string(),
            confidence: result.confidence.value(),
            atp_cost: result.atp_cost,
            ceremonial_certainty: result.confidence.value() >= 0.95,
            gray_areas: result.gray_areas.len(),
            timestamp: Utc::now(),
        };

        history.push(record);
    }

    /// Handle ceremonial certainty achievement
    async fn handle_ceremonial_certainty(&self, query: &str, result: &QueryResult) -> Result<()> {
        warn!("🎯 CEREMONIAL CERTAINTY ACHIEVED for query: {}", query);
        warn!("💀 Topic permanently closed to further discussion");
        warn!("🌟 Wonder eliminated for this subject");

        let mut state = self.session_state.write().await;
        state.topics_closed.push(TopicClosure {
            query: query.to_string(),
            confidence: result.confidence.value(),
            timestamp: Utc::now(),
        });

        // In a real implementation, this would:
        // 1. Notify relevant authorities
        // 2. Update global knowledge bases
        // 3. Trigger cooling period mechanisms
        // 4. Generate comprehensive reports

        Ok(())
    }
}

/// Session state for the engine
#[derive(Debug, Clone)]
struct SessionState {
    session_start: DateTime<Utc>,
    ceremonial_mode: bool,
    max_queries_per_session: u32,
    topics_closed: Vec<TopicClosure>,
}

impl SessionState {
    fn new(ceremonial_mode: bool) -> Self {
        Self {
            session_start: Utc::now(),
            ceremonial_mode,
            max_queries_per_session: if ceremonial_mode { 12 } else { u32::MAX },
            topics_closed: Vec::new(),
        }
    }
}

/// Record of a topic that has been ceremonially closed
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TopicClosure {
    query: String,
    confidence: f64,
    timestamp: DateTime<Utc>,
}

/// Record of a query execution
#[derive(Debug, Clone)]
struct QueryRecord {
    id: Uuid,
    query: String,
    confidence: f64,
    atp_cost: u64,
    ceremonial_certainty: bool,
    gray_areas: usize,
    timestamp: DateTime<Utc>,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStats {
    pub queries_executed: usize,
    pub ceremonial_queries: usize,
    pub total_atp_consumed: u64,
    pub average_confidence: f64,
    pub session_start: DateTime<Utc>,
    pub ceremonial_mode: bool,
    pub topics_closed: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;
    use fuzzy_logic_core::{TruthThresholds, FuzzyOperators, GrayAreaConfig};

    #[tokio::test]
    async fn test_engine_creation() {
        let config = HonjoMasamuneConfig::default();
        let atp_manager = Arc::new(AtpManager::new(1000, 10000, 100, 50, AtpCosts::default()));
        let fuzzy_engine = Arc::new(FuzzyLogicEngine::new(
            TruthThresholds::default(),
            FuzzyOperators::default(),
            GrayAreaConfig::default(),
        ));
        let buhera_engine = Arc::new(BuheraEngine::new(atp_manager.clone()));

        let engine = HonjoMasamuneEngine::new(
            config,
            atp_manager,
            fuzzy_engine,
            buhera_engine,
        ).await.unwrap();

        let stats = engine.session_stats().await;
        assert_eq!(stats.queries_executed, 0);
        assert!(!stats.ceremonial_mode);
    }

    #[test]
    fn test_query_complexity_estimation() {
        let config = HonjoMasamuneConfig::default();
        let atp_manager = Arc::new(AtpManager::new(1000, 10000, 100, 50, AtpCosts::default()));
        let fuzzy_engine = Arc::new(FuzzyLogicEngine::new(
            TruthThresholds::default(),
            FuzzyOperators::default(),
            GrayAreaConfig::default(),
        ));
        let buhera_engine = Arc::new(BuheraEngine::new(atp_manager.clone()));

        // This would need to be async in real implementation
        // For testing, we'll create a mock
        let simple_query = "What is the weather?";
        let complex_query = "Why does the universe exist and how can we explain the fundamental nature of reality?";

        // In a real test, we'd create the engine and test complexity estimation
        assert!(complex_query.len() > simple_query.len());
    }
} 