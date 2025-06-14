//! Zengeza Communication Noise Analysis Module
//! 
//! Humans never say exactly what they mean. There's always subtext, cultural context,
//! emotional undertones, and indirect implications. The zengeza module calculates
//! the "noise ratio" in human statements - the gap between literal meaning and
//! intended meaning - to help the truth engine better understand actual intent.

use anyhow::Result;
use atp_manager::AtpManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use regex::Regex;

use crate::config::ZengezaNoiseConfig;

/// The zengeza communication noise analysis engine
#[derive(Debug)]
pub struct ZengezaEngine {
    config: ZengezaNoiseConfig,
    atp_manager: Arc<AtpManager>,
    noise_analyzers: Vec<NoiseAnalyzer>,
    cultural_context_db: Arc<RwLock<CulturalContextDatabase>>,
    linguistic_patterns: Arc<RwLock<LinguisticPatternDatabase>>,
    noise_history: Arc<RwLock<Vec<NoiseAnalysisRecord>>>,
    calibration_data: Arc<RwLock<CalibrationData>>,
}

impl ZengezaEngine {
    /// Create a new zengeza noise analysis engine
    pub fn new(
        config: ZengezaNoiseConfig,
        atp_manager: Arc<AtpManager>,
    ) -> Self {
        let noise_analyzers = Self::initialize_noise_analyzers(&config);
        
        Self {
            config,
            atp_manager,
            noise_analyzers,
            cultural_context_db: Arc::new(RwLock::new(CulturalContextDatabase::new())),
            linguistic_patterns: Arc::new(RwLock::new(LinguisticPatternDatabase::new())),
            noise_history: Arc::new(RwLock::new(Vec::new())),
            calibration_data: Arc::new(RwLock::new(CalibrationData::new())),
        }
    }

    /// Analyze communication noise in a statement
    pub async fn analyze_communication_noise(&self, statement: &str, context: CommunicationContext) -> Result<NoiseAnalysisResult> {
        info!("🔊 Analyzing communication noise for statement: {}", 
              if statement.len() > 50 { &statement[..50] } else { statement });

        // Reserve ATP for noise analysis
        let analysis_cost = self.calculate_analysis_cost(statement, &context);
        let reservation = self.atp_manager.reserve_atp("noise_analysis", analysis_cost).await?;

        let analysis_id = Uuid::new_v4();
        let start_time = Utc::now();

        // Run all noise analyzers
        let mut noise_components = Vec::new();
        for analyzer in &self.noise_analyzers {
            let component = analyzer.analyze(statement, &context).await?;
            noise_components.push(component);
        }

        // Calculate overall noise ratio
        let overall_noise_ratio = self.calculate_overall_noise_ratio(&noise_components).await?;

        // Identify primary noise sources
        let primary_noise_sources = self.identify_primary_noise_sources(&noise_components);

        // Generate denoising suggestions
        let denoising_suggestions = self.generate_denoising_suggestions(&noise_components, statement).await?;

        // Calculate confidence in the analysis
        let analysis_confidence = self.calculate_analysis_confidence(&noise_components, &context);

        // Estimate intended meaning clarity
        let intended_meaning_clarity = 1.0 - overall_noise_ratio;

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "noise_analysis").await?;

        let result = NoiseAnalysisResult {
            analysis_id,
            statement: statement.to_string(),
            overall_noise_ratio,
            intended_meaning_clarity,
            noise_components,
            primary_noise_sources,
            denoising_suggestions,
            analysis_confidence,
            cultural_context_impact: context.cultural_weight,
            processing_time: Utc::now() - start_time,
            atp_cost: analysis_cost,
        };

        // Record analysis
        self.record_analysis(&result).await;

        info!("✅ Noise analysis complete: {:.3} noise ratio, {:.3} clarity", 
              result.overall_noise_ratio, result.intended_meaning_clarity);

        Ok(result)
    }

    /// Denoise a statement based on analysis
    pub async fn denoise_statement(
        &self,
        statement: &str,
        noise_analysis: &NoiseAnalysisResult,
    ) -> Result<DenoisedStatement> {
        info!("🧹 Denoising statement with {:.3} noise ratio", noise_analysis.overall_noise_ratio);

        let mut denoised_text = statement.to_string();
        let mut applied_transformations = Vec::new();

        // Apply denoising transformations based on analysis
        for suggestion in &noise_analysis.denoising_suggestions {
            match suggestion.transformation_type {
                DenoisingType::RemoveHedging => {
                    let (new_text, applied) = self.remove_hedging_language(&denoised_text);
                    if applied {
                        denoised_text = new_text;
                        applied_transformations.push(suggestion.clone());
                    }
                },
                DenoisingType::ClarifyImplications => {
                    let (new_text, applied) = self.clarify_implications(&denoised_text, &suggestion.context_clues);
                    if applied {
                        denoised_text = new_text;
                        applied_transformations.push(suggestion.clone());
                    }
                },
                DenoisingType::ResolveAmbiguity => {
                    let (new_text, applied) = self.resolve_ambiguity(&denoised_text, &suggestion.context_clues);
                    if applied {
                        denoised_text = new_text;
                        applied_transformations.push(suggestion.clone());
                    }
                },
                DenoisingType::ExtractSubtext => {
                    let (new_text, applied) = self.extract_subtext(&denoised_text, &suggestion.context_clues);
                    if applied {
                        denoised_text = new_text;
                        applied_transformations.push(suggestion.clone());
                    }
                },
                DenoisingType::NormalizeEmotion => {
                    let (new_text, applied) = self.normalize_emotional_language(&denoised_text);
                    if applied {
                        denoised_text = new_text;
                        applied_transformations.push(suggestion.clone());
                    }
                },
            }
        }

        // Calculate denoising effectiveness
        let noise_reduction = applied_transformations.len() as f64 / noise_analysis.denoising_suggestions.len() as f64;
        let estimated_clarity_improvement = noise_reduction * noise_analysis.overall_noise_ratio * 0.7; // Conservative estimate

        Ok(DenoisedStatement {
            original_statement: statement.to_string(),
            denoised_statement: denoised_text,
            applied_transformations,
            noise_reduction,
            estimated_clarity_improvement,
            confidence: noise_analysis.analysis_confidence * noise_reduction,
        })
    }

    /// Get noise analysis statistics
    pub async fn get_noise_statistics(&self) -> NoiseStatistics {
        let history = self.noise_history.read().await;
        let calibration = self.calibration_data.read().await;

        let total_analyses = history.len();
        let avg_noise_ratio = if total_analyses > 0 {
            history.iter().map(|r| r.noise_ratio).sum::<f64>() / total_analyses as f64
        } else {
            0.0
        };

        let high_noise_count = history.iter().filter(|r| r.noise_ratio > 0.7).count();
        let low_noise_count = history.iter().filter(|r| r.noise_ratio < 0.3).count();

        NoiseStatistics {
            total_analyses,
            average_noise_ratio: avg_noise_ratio,
            high_noise_statements: high_noise_count,
            low_noise_statements: low_noise_count,
            most_common_noise_types: Self::get_most_common_noise_types(&history),
            calibration_accuracy: calibration.accuracy_score,
            processing_efficiency: Self::calculate_processing_efficiency(&history),
        }
    }

    /// Initialize noise analyzers
    fn initialize_noise_analyzers(config: &ZengezaNoiseConfig) -> Vec<NoiseAnalyzer> {
        vec![
            NoiseAnalyzer::new(NoiseAnalyzerType::LinguisticHedging, config.hedging_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::CulturalImplication, config.cultural_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::EmotionalUndertone, config.emotional_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::ContextualAmbiguity, config.ambiguity_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::ImplicitAssumption, config.assumption_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::SocialSignaling, config.social_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::RhetoricalDevice, config.rhetorical_weight),
            NoiseAnalyzer::new(NoiseAnalyzerType::IndirectReference, config.indirect_weight),
        ]
    }

    /// Calculate overall noise ratio from components
    async fn calculate_overall_noise_ratio(&self, components: &[NoiseComponent]) -> Result<f64> {
        if components.is_empty() {
            return Ok(0.0);
        }

        // Weighted average of noise components
        let total_weight: f64 = components.iter().map(|c| c.weight).sum();
        let weighted_noise: f64 = components.iter()
            .map(|c| c.noise_level * c.weight)
            .sum();

        let base_ratio = if total_weight > 0.0 {
            weighted_noise / total_weight
        } else {
            0.0
        };

        // Apply interaction effects (some noise types amplify each other)
        let interaction_multiplier = self.calculate_interaction_effects(components);
        
        Ok((base_ratio * interaction_multiplier).min(1.0))
    }

    /// Calculate interaction effects between noise types
    fn calculate_interaction_effects(&self, components: &[NoiseComponent]) -> f64 {
        let mut multiplier = 1.0;
        
        // Cultural + Emotional amplification
        let cultural_noise = components.iter()
            .find(|c| matches!(c.noise_type, NoiseType::CulturalImplication))
            .map(|c| c.noise_level)
            .unwrap_or(0.0);
        let emotional_noise = components.iter()
            .find(|c| matches!(c.noise_type, NoiseType::EmotionalUndertone))
            .map(|c| c.noise_level)
            .unwrap_or(0.0);
        
        if cultural_noise > 0.5 && emotional_noise > 0.5 {
            multiplier += 0.2; // 20% amplification
        }

        // Ambiguity + Implicit assumptions compound
        let ambiguity_noise = components.iter()
            .find(|c| matches!(c.noise_type, NoiseType::ContextualAmbiguity))
            .map(|c| c.noise_level)
            .unwrap_or(0.0);
        let assumption_noise = components.iter()
            .find(|c| matches!(c.noise_type, NoiseType::ImplicitAssumption))
            .map(|c| c.noise_level)
            .unwrap_or(0.0);
        
        if ambiguity_noise > 0.6 && assumption_noise > 0.6 {
            multiplier += 0.15; // 15% amplification
        }

        multiplier
    }

    /// Identify primary noise sources
    fn identify_primary_noise_sources(&self, components: &[NoiseComponent]) -> Vec<NoiseType> {
        let mut sorted_components = components.to_vec();
        sorted_components.sort_by(|a, b| b.noise_level.partial_cmp(&a.noise_level).unwrap());
        
        sorted_components.into_iter()
            .take(3) // Top 3 noise sources
            .filter(|c| c.noise_level > 0.3) // Only significant noise
            .map(|c| c.noise_type)
            .collect()
    }

    /// Generate denoising suggestions
    async fn generate_denoising_suggestions(
        &self,
        components: &[NoiseComponent],
        statement: &str,
    ) -> Result<Vec<DenoisingSuggestion>> {
        let mut suggestions = Vec::new();

        for component in components {
            if component.noise_level > 0.4 { // Only suggest for significant noise
                let suggestion = match component.noise_type {
                    NoiseType::LinguisticHedging => DenoisingSuggestion {
                        transformation_type: DenoisingType::RemoveHedging,
                        description: "Remove hedging language (maybe, perhaps, might)".to_string(),
                        confidence: component.confidence,
                        context_clues: component.indicators.clone(),
                        expected_improvement: component.noise_level * 0.6,
                    },
                    NoiseType::CulturalImplication => DenoisingSuggestion {
                        transformation_type: DenoisingType::ClarifyImplications,
                        description: "Make cultural implications explicit".to_string(),
                        confidence: component.confidence,
                        context_clues: component.indicators.clone(),
                        expected_improvement: component.noise_level * 0.5,
                    },
                    NoiseType::ContextualAmbiguity => DenoisingSuggestion {
                        transformation_type: DenoisingType::ResolveAmbiguity,
                        description: "Clarify ambiguous references".to_string(),
                        confidence: component.confidence,
                        context_clues: component.indicators.clone(),
                        expected_improvement: component.noise_level * 0.7,
                    },
                    NoiseType::EmotionalUndertone => DenoisingSuggestion {
                        transformation_type: DenoisingType::NormalizeEmotion,
                        description: "Normalize emotional language".to_string(),
                        confidence: component.confidence,
                        context_clues: component.indicators.clone(),
                        expected_improvement: component.noise_level * 0.4,
                    },
                    _ => DenoisingSuggestion {
                        transformation_type: DenoisingType::ExtractSubtext,
                        description: "Extract implicit meaning".to_string(),
                        confidence: component.confidence,
                        context_clues: component.indicators.clone(),
                        expected_improvement: component.noise_level * 0.3,
                    },
                };
                suggestions.push(suggestion);
            }
        }

        Ok(suggestions)
    }

    /// Calculate analysis confidence
    fn calculate_analysis_confidence(&self, components: &[NoiseComponent], context: &CommunicationContext) -> f64 {
        let component_confidence: f64 = components.iter()
            .map(|c| c.confidence)
            .sum::<f64>() / components.len() as f64;

        let context_confidence = match context.communication_type {
            CommunicationType::Formal => 0.9,
            CommunicationType::Informal => 0.7,
            CommunicationType::Technical => 0.95,
            CommunicationType::Emotional => 0.6,
            CommunicationType::Cultural => 0.5,
        };

        (component_confidence + context_confidence) / 2.0
    }

    /// Denoising transformation methods
    fn remove_hedging_language(&self, text: &str) -> (String, bool) {
        let hedging_patterns = [
            r"\b(maybe|perhaps|possibly|might|could|would|should|seems?|appears?)\b",
            r"\b(I think|I believe|I guess|I suppose)\b",
            r"\b(kind of|sort of|somewhat|rather)\b",
        ];

        let mut result = text.to_string();
        let mut applied = false;

        for pattern in &hedging_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(&result) {
                    result = re.replace_all(&result, "").to_string();
                    applied = true;
                }
            }
        }

        // Clean up extra spaces
        result = result.replace("  ", " ").trim().to_string();
        
        (result, applied)
    }

    fn clarify_implications(&self, text: &str, context_clues: &[String]) -> (String, bool) {
        // Simplified implementation - in reality would use sophisticated NLP
        let mut result = text.to_string();
        let mut applied = false;

        for clue in context_clues {
            if text.contains("this") || text.contains("that") || text.contains("it") {
                result = result.replace("this", &format!("this ({})", clue));
                applied = true;
                break;
            }
        }

        (result, applied)
    }

    fn resolve_ambiguity(&self, text: &str, context_clues: &[String]) -> (String, bool) {
        // Simplified ambiguity resolution
        let mut result = text.to_string();
        let applied = false;

        // In a real implementation, this would use context to resolve pronouns,
        // unclear references, and ambiguous terms
        
        (result, applied)
    }

    fn extract_subtext(&self, text: &str, context_clues: &[String]) -> (String, bool) {
        // Extract implicit meaning and make it explicit
        let mut result = text.to_string();
        let applied = false;

        // This would analyze tone, implications, and unstated assumptions
        
        (result, applied)
    }

    fn normalize_emotional_language(&self, text: &str) -> (String, bool) {
        let emotional_patterns = [
            (r"\b(absolutely|totally|completely|utterly)\b", ""),
            (r"\b(amazing|incredible|unbelievable)\b", "notable"),
            (r"\b(terrible|awful|horrible)\b", "problematic"),
        ];

        let mut result = text.to_string();
        let mut applied = false;

        for (pattern, replacement) in &emotional_patterns {
            if let Ok(re) = Regex::new(pattern) {
                if re.is_match(&result) {
                    result = re.replace_all(&result, *replacement).to_string();
                    applied = true;
                }
            }
        }

        (result, applied)
    }

    /// Utility methods
    fn calculate_analysis_cost(&self, statement: &str, context: &CommunicationContext) -> u64 {
        let base_cost = 150u64;
        let length_cost = (statement.len() / 10) as u64;
        let complexity_cost = match context.communication_type {
            CommunicationType::Technical => 50,
            CommunicationType::Cultural => 100,
            CommunicationType::Emotional => 75,
            _ => 25,
        };
        
        base_cost + length_cost + complexity_cost
    }

    fn get_most_common_noise_types(history: &[NoiseAnalysisRecord]) -> Vec<NoiseType> {
        let mut type_counts = HashMap::new();
        
        for record in history {
            for noise_type in &record.primary_noise_sources {
                *type_counts.entry(noise_type.clone()).or_insert(0) += 1;
            }
        }
        
        let mut sorted_types: Vec<_> = type_counts.into_iter().collect();
        sorted_types.sort_by(|a, b| b.1.cmp(&a.1));
        
        sorted_types.into_iter().take(5).map(|(noise_type, _)| noise_type).collect()
    }

    fn calculate_processing_efficiency(history: &[NoiseAnalysisRecord]) -> f64 {
        if history.is_empty() {
            return 0.0;
        }
        
        let avg_processing_time = history.iter()
            .map(|r| r.processing_time.num_milliseconds())
            .sum::<i64>() / history.len() as i64;
        
        // Efficiency based on processing time (lower is better)
        1.0 / (1.0 + avg_processing_time as f64 / 1000.0)
    }

    async fn record_analysis(&self, result: &NoiseAnalysisResult) {
        let mut history = self.noise_history.write().await;
        
        let record = NoiseAnalysisRecord {
            timestamp: Utc::now(),
            analysis_id: result.analysis_id,
            noise_ratio: result.overall_noise_ratio,
            clarity: result.intended_meaning_clarity,
            primary_noise_sources: result.primary_noise_sources.clone(),
            processing_time: result.processing_time,
            atp_cost: result.atp_cost,
        };
        
        history.push(record);
        
        // Keep recent history
        if history.len() > 1000 {
            history.drain(0..500);
        }
    }
}

/// Noise analyzer for specific types of communication noise
#[derive(Debug, Clone)]
pub struct NoiseAnalyzer {
    analyzer_type: NoiseAnalyzerType,
    weight: f64,
}

impl NoiseAnalyzer {
    fn new(analyzer_type: NoiseAnalyzerType, weight: f64) -> Self {
        Self { analyzer_type, weight }
    }

    async fn analyze(&self, statement: &str, context: &CommunicationContext) -> Result<NoiseComponent> {
        let (noise_level, confidence, indicators) = match self.analyzer_type {
            NoiseAnalyzerType::LinguisticHedging => self.analyze_hedging(statement),
            NoiseAnalyzerType::CulturalImplication => self.analyze_cultural_implications(statement, context),
            NoiseAnalyzerType::EmotionalUndertone => self.analyze_emotional_undertones(statement),
            NoiseAnalyzerType::ContextualAmbiguity => self.analyze_ambiguity(statement),
            NoiseAnalyzerType::ImplicitAssumption => self.analyze_assumptions(statement),
            NoiseAnalyzerType::SocialSignaling => self.analyze_social_signals(statement),
            NoiseAnalyzerType::RhetoricalDevice => self.analyze_rhetorical_devices(statement),
            NoiseAnalyzerType::IndirectReference => self.analyze_indirect_references(statement),
        };

        Ok(NoiseComponent {
            noise_type: self.analyzer_type.into(),
            noise_level,
            confidence,
            weight: self.weight,
            indicators,
            description: self.get_description(),
        })
    }

    fn analyze_hedging(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let hedging_words = ["maybe", "perhaps", "possibly", "might", "could", "seems", "appears", "I think", "I believe"];
        let mut indicators = Vec::new();
        let mut hedging_count = 0;

        for word in &hedging_words {
            if statement.to_lowercase().contains(word) {
                hedging_count += 1;
                indicators.push(word.to_string());
            }
        }

        let word_count = statement.split_whitespace().count();
        let noise_level = (hedging_count as f64 / word_count as f64).min(1.0) * 2.0; // Amplify effect
        let confidence = 0.9; // High confidence in detecting hedging

        (noise_level, confidence, indicators)
    }

    fn analyze_cultural_implications(&self, statement: &str, context: &CommunicationContext) -> (f64, f64, Vec<String>) {
        let cultural_markers = ["obviously", "of course", "everyone knows", "it's clear that", "naturally"];
        let mut indicators = Vec::new();
        let mut marker_count = 0;

        for marker in &cultural_markers {
            if statement.to_lowercase().contains(marker) {
                marker_count += 1;
                indicators.push(marker.to_string());
            }
        }

        let base_noise = (marker_count as f64 * 0.3).min(1.0);
        let cultural_amplifier = context.cultural_weight;
        let noise_level = base_noise * cultural_amplifier;
        let confidence = 0.7; // Moderate confidence

        (noise_level, confidence, indicators)
    }

    fn analyze_emotional_undertones(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let emotional_words = ["love", "hate", "amazing", "terrible", "incredible", "awful", "fantastic", "horrible"];
        let intensity_words = ["very", "extremely", "absolutely", "totally", "completely"];
        
        let mut indicators = Vec::new();
        let mut emotional_count = 0;

        for word in emotional_words.iter().chain(intensity_words.iter()) {
            if statement.to_lowercase().contains(word) {
                emotional_count += 1;
                indicators.push(word.to_string());
            }
        }

        let word_count = statement.split_whitespace().count();
        let noise_level = (emotional_count as f64 / word_count as f64).min(1.0) * 1.5;
        let confidence = 0.8;

        (noise_level, confidence, indicators)
    }

    fn analyze_ambiguity(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let ambiguous_words = ["this", "that", "it", "they", "some", "many", "few", "several"];
        let mut indicators = Vec::new();
        let mut ambiguous_count = 0;

        for word in &ambiguous_words {
            if statement.to_lowercase().contains(word) {
                ambiguous_count += 1;
                indicators.push(word.to_string());
            }
        }

        let word_count = statement.split_whitespace().count();
        let noise_level = (ambiguous_count as f64 / word_count as f64).min(1.0) * 1.2;
        let confidence = 0.6; // Lower confidence as context matters

        (noise_level, confidence, indicators)
    }

    fn analyze_assumptions(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let assumption_markers = ["obviously", "clearly", "of course", "naturally", "as we all know"];
        let mut indicators = Vec::new();
        let mut assumption_count = 0;

        for marker in &assumption_markers {
            if statement.to_lowercase().contains(marker) {
                assumption_count += 1;
                indicators.push(marker.to_string());
            }
        }

        let noise_level = (assumption_count as f64 * 0.4).min(1.0);
        let confidence = 0.8;

        (noise_level, confidence, indicators)
    }

    fn analyze_social_signals(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let social_markers = ["we", "us", "our", "everyone", "people", "society"];
        let mut indicators = Vec::new();
        let mut social_count = 0;

        for marker in &social_markers {
            if statement.to_lowercase().contains(marker) {
                social_count += 1;
                indicators.push(marker.to_string());
            }
        }

        let word_count = statement.split_whitespace().count();
        let noise_level = (social_count as f64 / word_count as f64).min(1.0) * 0.8;
        let confidence = 0.5; // Low confidence as these can be legitimate

        (noise_level, confidence, indicators)
    }

    fn analyze_rhetorical_devices(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let rhetorical_patterns = ["isn't it?", "don't you think?", "wouldn't you agree?", "right?"];
        let mut indicators = Vec::new();
        let mut rhetorical_count = 0;

        for pattern in &rhetorical_patterns {
            if statement.to_lowercase().contains(pattern) {
                rhetorical_count += 1;
                indicators.push(pattern.to_string());
            }
        }

        let noise_level = (rhetorical_count as f64 * 0.5).min(1.0);
        let confidence = 0.9;

        (noise_level, confidence, indicators)
    }

    fn analyze_indirect_references(&self, statement: &str) -> (f64, f64, Vec<String>) {
        let indirect_markers = ["the thing", "you know", "that stuff", "what's-his-name", "whatsit"];
        let mut indicators = Vec::new();
        let mut indirect_count = 0;

        for marker in &indirect_markers {
            if statement.to_lowercase().contains(marker) {
                indirect_count += 1;
                indicators.push(marker.to_string());
            }
        }

        let noise_level = (indirect_count as f64 * 0.6).min(1.0);
        let confidence = 0.7;

        (noise_level, confidence, indicators)
    }

    fn get_description(&self) -> String {
        match self.analyzer_type {
            NoiseAnalyzerType::LinguisticHedging => "Detects hedging language that weakens statements".to_string(),
            NoiseAnalyzerType::CulturalImplication => "Identifies culturally-dependent assumptions".to_string(),
            NoiseAnalyzerType::EmotionalUndertone => "Analyzes emotional coloring of language".to_string(),
            NoiseAnalyzerType::ContextualAmbiguity => "Finds ambiguous references and pronouns".to_string(),
            NoiseAnalyzerType::ImplicitAssumption => "Detects unstated assumptions".to_string(),
            NoiseAnalyzerType::SocialSignaling => "Identifies social positioning language".to_string(),
            NoiseAnalyzerType::RhetoricalDevice => "Finds rhetorical questions and devices".to_string(),
            NoiseAnalyzerType::IndirectReference => "Detects vague and indirect references".to_string(),
        }
    }
}

/// Supporting data structures and databases
#[derive(Debug)]
pub struct CulturalContextDatabase {
    contexts: HashMap<String, CulturalContext>,
}

impl CulturalContextDatabase {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct LinguisticPatternDatabase {
    patterns: HashMap<String, LinguisticPattern>,
}

impl LinguisticPatternDatabase {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct CalibrationData {
    accuracy_score: f64,
    sample_count: usize,
}

impl CalibrationData {
    fn new() -> Self {
        Self {
            accuracy_score: 0.0,
            sample_count: 0,
        }
    }
}

/// Core data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationContext {
    pub communication_type: CommunicationType,
    pub cultural_background: String,
    pub formality_level: f64,
    pub emotional_context: String,
    pub cultural_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommunicationType {
    Formal,
    Informal,
    Technical,
    Emotional,
    Cultural,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseAnalysisResult {
    pub analysis_id: Uuid,
    pub statement: String,
    pub overall_noise_ratio: f64,
    pub intended_meaning_clarity: f64,
    pub noise_components: Vec<NoiseComponent>,
    pub primary_noise_sources: Vec<NoiseType>,
    pub denoising_suggestions: Vec<DenoisingSuggestion>,
    pub analysis_confidence: f64,
    pub cultural_context_impact: f64,
    pub processing_time: Duration,
    pub atp_cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseComponent {
    pub noise_type: NoiseType,
    pub noise_level: f64,
    pub confidence: f64,
    pub weight: f64,
    pub indicators: Vec<String>,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum NoiseType {
    LinguisticHedging,
    CulturalImplication,
    EmotionalUndertone,
    ContextualAmbiguity,
    ImplicitAssumption,
    SocialSignaling,
    RhetoricalDevice,
    IndirectReference,
}

#[derive(Debug, Clone)]
pub enum NoiseAnalyzerType {
    LinguisticHedging,
    CulturalImplication,
    EmotionalUndertone,
    ContextualAmbiguity,
    ImplicitAssumption,
    SocialSignaling,
    RhetoricalDevice,
    IndirectReference,
}

impl From<NoiseAnalyzerType> for NoiseType {
    fn from(analyzer_type: NoiseAnalyzerType) -> Self {
        match analyzer_type {
            NoiseAnalyzerType::LinguisticHedging => NoiseType::LinguisticHedging,
            NoiseAnalyzerType::CulturalImplication => NoiseType::CulturalImplication,
            NoiseAnalyzerType::EmotionalUndertone => NoiseType::EmotionalUndertone,
            NoiseAnalyzerType::ContextualAmbiguity => NoiseType::ContextualAmbiguity,
            NoiseAnalyzerType::ImplicitAssumption => NoiseType::ImplicitAssumption,
            NoiseAnalyzerType::SocialSignaling => NoiseType::SocialSignaling,
            NoiseAnalyzerType::RhetoricalDevice => NoiseType::RhetoricalDevice,
            NoiseAnalyzerType::IndirectReference => NoiseType::IndirectReference,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoisingSuggestion {
    pub transformation_type: DenoisingType,
    pub description: String,
    pub confidence: f64,
    pub context_clues: Vec<String>,
    pub expected_improvement: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DenoisingType {
    RemoveHedging,
    ClarifyImplications,
    ResolveAmbiguity,
    ExtractSubtext,
    NormalizeEmotion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenoisedStatement {
    pub original_statement: String,
    pub denoised_statement: String,
    pub applied_transformations: Vec<DenoisingSuggestion>,
    pub noise_reduction: f64,
    pub estimated_clarity_improvement: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseStatistics {
    pub total_analyses: usize,
    pub average_noise_ratio: f64,
    pub high_noise_statements: usize,
    pub low_noise_statements: usize,
    pub most_common_noise_types: Vec<NoiseType>,
    pub calibration_accuracy: f64,
    pub processing_efficiency: f64,
}

// Internal data structures
#[derive(Debug, Clone)]
struct CulturalContext {
    region: String,
    communication_style: String,
    implicit_assumptions: Vec<String>,
}

#[derive(Debug, Clone)]
struct LinguisticPattern {
    pattern: String,
    noise_contribution: f64,
    context_dependency: f64,
}

#[derive(Debug, Clone)]
struct NoiseAnalysisRecord {
    timestamp: DateTime<Utc>,
    analysis_id: Uuid,
    noise_ratio: f64,
    clarity: f64,
    primary_noise_sources: Vec<NoiseType>,
    processing_time: Duration,
    atp_cost: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[tokio::test]
    async fn test_zengeza_engine_creation() {
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

        let config = ZengezaNoiseConfig::default();
        let engine = ZengezaEngine::new(config, atp_manager);

        let stats = engine.get_noise_statistics().await;
        assert_eq!(stats.total_analyses, 0);
    }

    #[tokio::test]
    async fn test_noise_analysis() {
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

        let config = ZengezaNoiseConfig::default();
        let engine = ZengezaEngine::new(config, atp_manager);

        let context = CommunicationContext {
            communication_type: CommunicationType::Informal,
            cultural_background: "Western".to_string(),
            formality_level: 0.3,
            emotional_context: "Neutral".to_string(),
            cultural_weight: 0.5,
        };

        let statement = "I think maybe this could possibly be a good idea, you know?";
        let result = engine.analyze_communication_noise(statement, context).await.unwrap();

        assert!(result.overall_noise_ratio > 0.0);
        assert!(result.intended_meaning_clarity < 1.0);
        assert!(!result.noise_components.is_empty());
    }
} 