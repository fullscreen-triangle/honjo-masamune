//! Spectacular Module
//! 
//! Handles extraordinary implications or findings that require special attention.
//! When the system treats every piece of information the same, it can miss the most
//! significant discoveries. This module identifies and processes spectacular findings.

use anyhow::Result;
use atp_manager::AtpManager;
use fuzzy_logic_core::{FuzzyResult, FuzzyTruth, TruthSpectrum};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::QueryResult;

/// Spectacular findings processor
#[derive(Debug)]
pub struct SpectacularEngine {
    atp_manager: Arc<AtpManager>,
    detection_criteria: SpectacularCriteria,
    findings_registry: Arc<RwLock<Vec<SpectacularFinding>>>,
    processing_strategies: Vec<ProcessingStrategy>,
}

impl SpectacularEngine {
    /// Create a new spectacular engine
    pub fn new(
        atp_manager: Arc<AtpManager>,
        detection_criteria: SpectacularCriteria,
    ) -> Self {
        let processing_strategies = vec![
            ProcessingStrategy::ParadigmShift,
            ProcessingStrategy::AnomalyAmplification,
            ProcessingStrategy::ContextualElevation,
            ProcessingStrategy::ResonanceDetection,
            ProcessingStrategy::EmergentPattern,
        ];

        Self {
            atp_manager,
            detection_criteria,
            findings_registry: Arc::new(RwLock::new(Vec::new())),
            processing_strategies,
        }
    }

    /// Analyze a query result for spectacular implications
    pub async fn analyze_for_spectacular_implications(
        &self,
        query: &str,
        result: &QueryResult,
        truth_spectrum: &TruthSpectrum,
    ) -> Result<Option<SpectacularFinding>> {
        info!("✨ Analyzing query for spectacular implications: {}", query);

        // Check if finding meets spectacular criteria
        let spectacular_indicators = self.detect_spectacular_indicators(
            query,
            result,
            truth_spectrum,
        ).await?;

        if spectacular_indicators.is_empty() {
            return Ok(None);
        }

        // Reserve additional ATP for spectacular processing
        let spectacular_cost = self.calculate_spectacular_processing_cost(&spectacular_indicators);
        let reservation = self.atp_manager.reserve_atp("spectacular_processing", spectacular_cost).await?;

        // Create spectacular finding
        let finding = SpectacularFinding {
            id: Uuid::new_v4(),
            query: query.to_string(),
            original_confidence: result.confidence.value(),
            spectacular_indicators,
            significance_score: 0.0, // Will be calculated
            processing_applied: Vec::new(),
            discovery_timestamp: Utc::now(),
            implications: Vec::new(),
            resonance_patterns: Vec::new(),
        };

        // Apply spectacular processing
        let processed_finding = self.apply_spectacular_processing(finding).await?;

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "spectacular_processing").await?;

        // Register the finding
        self.register_spectacular_finding(&processed_finding).await;

        warn!("🌟 SPECTACULAR FINDING DETECTED: {}", processed_finding.query);
        warn!("💫 Significance Score: {:.4}", processed_finding.significance_score);
        warn!("🔥 Indicators: {:?}", processed_finding.spectacular_indicators);

        Ok(Some(processed_finding))
    }

    /// Detect spectacular indicators in a query result
    async fn detect_spectacular_indicators(
        &self,
        query: &str,
        result: &QueryResult,
        truth_spectrum: &TruthSpectrum,
    ) -> Result<Vec<SpectacularIndicator>> {
        let mut indicators = Vec::new();

        // High confidence with low initial expectation
        if result.confidence.value() > self.detection_criteria.confidence_threshold &&
           self.is_unexpected_result(query, result.confidence.value()) {
            indicators.push(SpectacularIndicator::UnexpectedCertainty);
        }

        // Extreme confidence values (very high or very low)
        if result.confidence.value() > 0.98 {
            indicators.push(SpectacularIndicator::ExtremeConfidence);
        }

        // Paradoxical findings (simultaneous high and low confidence in different aspects)
        if self.detect_paradoxical_pattern(truth_spectrum) {
            indicators.push(SpectacularIndicator::ParadoxicalPattern);
        }

        // Paradigm-shifting implications
        if self.detect_paradigm_shift_potential(query, result) {
            indicators.push(SpectacularIndicator::ParadigmShift);
        }

        // Novel pattern emergence
        if self.detect_novel_patterns(truth_spectrum) {
            indicators.push(SpectacularIndicator::NovelPattern);
        }

        // Cross-domain resonance
        if self.detect_cross_domain_resonance(query, result) {
            indicators.push(SpectacularIndicator::CrossDomainResonance);
        }

        // Recursive self-reference
        if self.detect_recursive_implications(query, result) {
            indicators.push(SpectacularIndicator::RecursiveImplication);
        }

        // Historical significance
        if self.assess_historical_significance(query, result) {
            indicators.push(SpectacularIndicator::HistoricalSignificance);
        }

        Ok(indicators)
    }

    /// Apply spectacular processing strategies
    async fn apply_spectacular_processing(
        &self,
        mut finding: SpectacularFinding,
    ) -> Result<SpectacularFinding> {
        info!("🔬 Applying spectacular processing strategies");

        for strategy in &self.processing_strategies {
            let enhancement = self.apply_processing_strategy(strategy, &finding).await?;
            if let Some(enhancement) = enhancement {
                finding.processing_applied.push(enhancement);
            }
        }

        // Calculate final significance score
        finding.significance_score = self.calculate_significance_score(&finding);

        // Extract implications
        finding.implications = self.extract_implications(&finding).await?;

        // Detect resonance patterns
        finding.resonance_patterns = self.detect_resonance_patterns(&finding).await?;

        Ok(finding)
    }

    /// Apply a specific processing strategy
    async fn apply_processing_strategy(
        &self,
        strategy: &ProcessingStrategy,
        finding: &SpectacularFinding,
    ) -> Result<Option<ProcessingEnhancement>> {
        match strategy {
            ProcessingStrategy::ParadigmShift => {
                if finding.spectacular_indicators.contains(&SpectacularIndicator::ParadigmShift) {
                    Ok(Some(ProcessingEnhancement {
                        strategy: strategy.clone(),
                        enhancement_factor: 2.5,
                        description: "Paradigm-shifting implications detected and amplified".to_string(),
                        applied_at: Utc::now(),
                    }))
                } else {
                    Ok(None)
                }
            },
            ProcessingStrategy::AnomalyAmplification => {
                if finding.spectacular_indicators.contains(&SpectacularIndicator::UnexpectedCertainty) {
                    Ok(Some(ProcessingEnhancement {
                        strategy: strategy.clone(),
                        enhancement_factor: 1.8,
                        description: "Anomalous patterns amplified for deeper analysis".to_string(),
                        applied_at: Utc::now(),
                    }))
                } else {
                    Ok(None)
                }
            },
            ProcessingStrategy::ContextualElevation => {
                if finding.spectacular_indicators.contains(&SpectacularIndicator::HistoricalSignificance) {
                    Ok(Some(ProcessingEnhancement {
                        strategy: strategy.clone(),
                        enhancement_factor: 1.6,
                        description: "Historical context elevated for broader perspective".to_string(),
                        applied_at: Utc::now(),
                    }))
                } else {
                    Ok(None)
                }
            },
            ProcessingStrategy::ResonanceDetection => {
                if finding.spectacular_indicators.contains(&SpectacularIndicator::CrossDomainResonance) {
                    Ok(Some(ProcessingEnhancement {
                        strategy: strategy.clone(),
                        enhancement_factor: 2.0,
                        description: "Cross-domain resonance patterns detected and enhanced".to_string(),
                        applied_at: Utc::now(),
                    }))
                } else {
                    Ok(None)
                }
            },
            ProcessingStrategy::EmergentPattern => {
                if finding.spectacular_indicators.contains(&SpectacularIndicator::NovelPattern) {
                    Ok(Some(ProcessingEnhancement {
                        strategy: strategy.clone(),
                        enhancement_factor: 1.9,
                        description: "Emergent patterns recognized and elevated".to_string(),
                        applied_at: Utc::now(),
                    }))
                } else {
                    Ok(None)
                }
            },
        }
    }

    /// Calculate the significance score for a finding
    fn calculate_significance_score(&self, finding: &SpectacularFinding) -> f64 {
        let base_score = finding.original_confidence * finding.spectacular_indicators.len() as f64;
        
        let enhancement_multiplier: f64 = finding.processing_applied
            .iter()
            .map(|p| p.enhancement_factor)
            .sum::<f64>()
            .max(1.0);

        let indicator_weight = finding.spectacular_indicators
            .iter()
            .map(|i| i.weight())
            .sum::<f64>();

        base_score * enhancement_multiplier * indicator_weight / 100.0
    }

    /// Extract implications from a spectacular finding
    async fn extract_implications(&self, finding: &SpectacularFinding) -> Result<Vec<String>> {
        let mut implications = Vec::new();

        for indicator in &finding.spectacular_indicators {
            match indicator {
                SpectacularIndicator::ParadigmShift => {
                    implications.push("Fundamental assumptions about the domain may need revision".to_string());
                    implications.push("Existing theoretical frameworks may be incomplete".to_string());
                },
                SpectacularIndicator::UnexpectedCertainty => {
                    implications.push("Previously uncertain areas may have clear answers".to_string());
                    implications.push("Conventional wisdom may be challenged".to_string());
                },
                SpectacularIndicator::ExtremeConfidence => {
                    implications.push("This finding represents near-absolute certainty".to_string());
                    implications.push("Further investigation may be unnecessary".to_string());
                },
                SpectacularIndicator::ParadoxicalPattern => {
                    implications.push("Reality may be more complex than current models suggest".to_string());
                    implications.push("Paradoxical elements may indicate deeper truths".to_string());
                },
                SpectacularIndicator::NovelPattern => {
                    implications.push("New patterns of understanding have emerged".to_string());
                    implications.push("Innovation opportunities may be present".to_string());
                },
                SpectacularIndicator::CrossDomainResonance => {
                    implications.push("Connections exist between previously separate domains".to_string());
                    implications.push("Interdisciplinary approaches may yield greater insights".to_string());
                },
                SpectacularIndicator::RecursiveImplication => {
                    implications.push("Self-referential loops may create emergent properties".to_string());
                    implications.push("Recursive structures may be fundamental".to_string());
                },
                SpectacularIndicator::HistoricalSignificance => {
                    implications.push("This finding may have lasting historical importance".to_string());
                    implications.push("Future research directions may be influenced".to_string());
                },
            }
        }

        Ok(implications)
    }

    /// Detect resonance patterns in a finding
    async fn detect_resonance_patterns(&self, finding: &SpectacularFinding) -> Result<Vec<ResonancePattern>> {
        let mut patterns = Vec::new();

        // Check for harmonic resonance
        if finding.spectacular_indicators.len() >= 3 {
            patterns.push(ResonancePattern {
                pattern_type: ResonanceType::Harmonic,
                strength: 0.8,
                description: "Multiple indicators create harmonic resonance".to_string(),
            });
        }

        // Check for amplification resonance
        if finding.processing_applied.iter().any(|p| p.enhancement_factor > 2.0) {
            patterns.push(ResonancePattern {
                pattern_type: ResonanceType::Amplification,
                strength: 0.9,
                description: "Processing strategies create amplification resonance".to_string(),
            });
        }

        // Check for temporal resonance
        if finding.spectacular_indicators.contains(&SpectacularIndicator::HistoricalSignificance) {
            patterns.push(ResonancePattern {
                pattern_type: ResonanceType::Temporal,
                strength: 0.7,
                description: "Historical significance creates temporal resonance".to_string(),
            });
        }

        Ok(patterns)
    }

    /// Register a spectacular finding
    async fn register_spectacular_finding(&self, finding: &SpectacularFinding) {
        let mut registry = self.findings_registry.write().await;
        registry.push(finding.clone());
        
        info!("📋 Spectacular finding registered: {} (score: {:.4})", 
              finding.query, finding.significance_score);
    }

    /// Get all spectacular findings
    pub async fn get_spectacular_findings(&self) -> Vec<SpectacularFinding> {
        let registry = self.findings_registry.read().await;
        registry.clone()
    }

    /// Get top spectacular findings by significance
    pub async fn get_top_spectacular_findings(&self, limit: usize) -> Vec<SpectacularFinding> {
        let mut findings = self.get_spectacular_findings().await;
        findings.sort_by(|a, b| b.significance_score.partial_cmp(&a.significance_score).unwrap());
        findings.into_iter().take(limit).collect()
    }

    /// Calculate ATP cost for spectacular processing
    fn calculate_spectacular_processing_cost(&self, indicators: &[SpectacularIndicator]) -> u64 {
        let base_cost = 500u64; // Base cost for spectacular processing
        let indicator_cost = indicators.len() as u64 * 100;
        let complexity_multiplier = if indicators.len() > 4 { 2 } else { 1 };
        
        (base_cost + indicator_cost) * complexity_multiplier
    }

    // Detection helper methods
    fn is_unexpected_result(&self, query: &str, confidence: f64) -> bool {
        // Simple heuristic: if query contains uncertainty keywords but confidence is high
        let uncertainty_keywords = ["maybe", "perhaps", "uncertain", "unclear", "unknown", "mystery"];
        let has_uncertainty = uncertainty_keywords.iter().any(|&keyword| query.to_lowercase().contains(keyword));
        
        has_uncertainty && confidence > 0.85
    }

    fn detect_paradoxical_pattern(&self, truth_spectrum: &TruthSpectrum) -> bool {
        // Check for simultaneous high and low confidence in different aspects
        truth_spectrum.confidence_distribution.iter().any(|&conf| conf > 0.9) &&
        truth_spectrum.confidence_distribution.iter().any(|&conf| conf < 0.3)
    }

    fn detect_paradigm_shift_potential(&self, query: &str, result: &QueryResult) -> bool {
        let paradigm_keywords = ["paradigm", "fundamental", "revolutionary", "breakthrough", "impossible", "contradiction"];
        let has_paradigm_language = paradigm_keywords.iter().any(|&keyword| query.to_lowercase().contains(keyword));
        
        has_paradigm_language && result.confidence.value() > 0.8
    }

    fn detect_novel_patterns(&self, truth_spectrum: &TruthSpectrum) -> bool {
        // Check for unusual confidence distributions
        let mean_confidence: f64 = truth_spectrum.confidence_distribution.iter().sum::<f64>() / truth_spectrum.confidence_distribution.len() as f64;
        let variance: f64 = truth_spectrum.confidence_distribution.iter()
            .map(|&x| (x - mean_confidence).powi(2))
            .sum::<f64>() / truth_spectrum.confidence_distribution.len() as f64;
        
        variance > 0.3 // High variance indicates novel patterns
    }

    fn detect_cross_domain_resonance(&self, query: &str, result: &QueryResult) -> bool {
        let domain_keywords = ["physics", "biology", "psychology", "philosophy", "mathematics", "chemistry", "sociology"];
        let domain_count = domain_keywords.iter().filter(|&keyword| query.to_lowercase().contains(keyword)).count();
        
        domain_count >= 2 && result.confidence.value() > 0.75
    }

    fn detect_recursive_implications(&self, query: &str, result: &QueryResult) -> bool {
        let recursive_keywords = ["self", "recursive", "circular", "loop", "itself", "meta"];
        let has_recursive_language = recursive_keywords.iter().any(|&keyword| query.to_lowercase().contains(keyword));
        
        has_recursive_language && result.confidence.value() > 0.7
    }

    fn assess_historical_significance(&self, query: &str, result: &QueryResult) -> bool {
        let significance_keywords = ["historic", "unprecedented", "first", "never", "always", "forever", "eternal"];
        let has_significance_language = significance_keywords.iter().any(|&keyword| query.to_lowercase().contains(keyword));
        
        has_significance_language && result.confidence.value() > 0.85
    }
}

/// Configuration for spectacular detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectacularCriteria {
    pub confidence_threshold: f64,
    pub significance_threshold: f64,
    pub min_indicators: usize,
    pub enable_amplification: bool,
    pub resonance_detection: bool,
}

impl Default for SpectacularCriteria {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.85,
            significance_threshold: 0.7,
            min_indicators: 2,
            enable_amplification: true,
            resonance_detection: true,
        }
    }
}

/// Types of spectacular indicators
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpectacularIndicator {
    UnexpectedCertainty,
    ExtremeConfidence,
    ParadoxicalPattern,
    ParadigmShift,
    NovelPattern,
    CrossDomainResonance,
    RecursiveImplication,
    HistoricalSignificance,
}

impl SpectacularIndicator {
    fn weight(&self) -> f64 {
        match self {
            SpectacularIndicator::ParadigmShift => 3.0,
            SpectacularIndicator::ExtremeConfidence => 2.5,
            SpectacularIndicator::HistoricalSignificance => 2.8,
            SpectacularIndicator::UnexpectedCertainty => 2.0,
            SpectacularIndicator::ParadoxicalPattern => 2.2,
            SpectacularIndicator::CrossDomainResonance => 1.8,
            SpectacularIndicator::RecursiveImplication => 1.9,
            SpectacularIndicator::NovelPattern => 1.5,
        }
    }
}

/// Processing strategies for spectacular findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingStrategy {
    ParadigmShift,
    AnomalyAmplification,
    ContextualElevation,
    ResonanceDetection,
    EmergentPattern,
}

/// A spectacular finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectacularFinding {
    pub id: Uuid,
    pub query: String,
    pub original_confidence: f64,
    pub spectacular_indicators: Vec<SpectacularIndicator>,
    pub significance_score: f64,
    pub processing_applied: Vec<ProcessingEnhancement>,
    pub discovery_timestamp: DateTime<Utc>,
    pub implications: Vec<String>,
    pub resonance_patterns: Vec<ResonancePattern>,
}

/// Processing enhancement applied to a finding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingEnhancement {
    pub strategy: ProcessingStrategy,
    pub enhancement_factor: f64,
    pub description: String,
    pub applied_at: DateTime<Utc>,
}

/// Resonance pattern detected in findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonancePattern {
    pub pattern_type: ResonanceType,
    pub strength: f64,
    pub description: String,
}

/// Types of resonance patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResonanceType {
    Harmonic,
    Amplification,
    Temporal,
    Spatial,
    Conceptual,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[tokio::test]
    async fn test_spectacular_detection() {
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

        let engine = SpectacularEngine::new(
            atp_manager,
            SpectacularCriteria::default(),
        );

        // Test detection capabilities
        assert!(engine.is_unexpected_result("This is uncertain and unclear", 0.95));
        assert!(!engine.is_unexpected_result("This is certain", 0.95));
    }

    #[tokio::test]
    async fn test_significance_calculation() {
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

        let engine = SpectacularEngine::new(
            atp_manager,
            SpectacularCriteria::default(),
        );

        let finding = SpectacularFinding {
            id: Uuid::new_v4(),
            query: "test".to_string(),
            original_confidence: 0.9,
            spectacular_indicators: vec![SpectacularIndicator::ParadigmShift],
            significance_score: 0.0,
            processing_applied: vec![],
            discovery_timestamp: Utc::now(),
            implications: vec![],
            resonance_patterns: vec![],
        };

        let score = engine.calculate_significance_score(&finding);
        assert!(score > 0.0);
    }
}