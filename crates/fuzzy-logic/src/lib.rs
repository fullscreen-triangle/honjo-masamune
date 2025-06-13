//! Fuzzy Logic Core for Honjo Masamune Truth Engine
//! 
//! This crate implements the fuzzy logic system that handles the fundamental truth
//! that "no human message is 100% true". It provides fuzzy truth values, operators,
//! gray area detection, and uncertainty propagation.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;
use chrono::{DateTime, Utc};

pub mod operators;
pub mod truth_spectrum;
pub mod gray_areas;
pub mod uncertainty;
pub mod hedges;

pub use operators::*;
pub use truth_spectrum::*;
pub use gray_areas::*;
pub use uncertainty::*;
pub use hedges::*;

/// Fuzzy truth value representing certainty on a scale from 0.0 to 1.0
/// 0.0 = completely false, 1.0 = completely true
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct FuzzyTruth(f64);

impl FuzzyTruth {
    /// Create a new fuzzy truth value
    pub fn new(value: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&value) {
            return Err(anyhow::anyhow!("Fuzzy truth value must be between 0.0 and 1.0, got {}", value));
        }
        Ok(FuzzyTruth(value))
    }

    /// Create a fuzzy truth value without validation (for internal use)
    pub fn new_unchecked(value: f64) -> Self {
        FuzzyTruth(value.clamp(0.0, 1.0))
    }

    /// Get the raw value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Check if this is in the gray area (ambiguous truth)
    pub fn is_gray_area(&self, range: [f64; 2]) -> bool {
        self.0 >= range[0] && self.0 <= range[1]
    }

    /// Convert to truth membership category
    pub fn to_membership(&self, thresholds: &TruthThresholds) -> TruthMembership {
        match self.0 {
            x if x >= thresholds.certain => TruthMembership::Certain,
            x if x >= thresholds.probable => TruthMembership::Probable,
            x if x >= thresholds.possible => TruthMembership::Possible,
            x if x >= thresholds.unlikely => TruthMembership::Unlikely,
            x if x >= thresholds.improbable => TruthMembership::Improbable,
            _ => TruthMembership::False,
        }
    }
}

impl fmt::Display for FuzzyTruth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.3}", self.0)
    }
}

impl From<f64> for FuzzyTruth {
    fn from(value: f64) -> Self {
        FuzzyTruth::new_unchecked(value)
    }
}

impl From<FuzzyTruth> for f64 {
    fn from(truth: FuzzyTruth) -> Self {
        truth.0
    }
}

/// Truth membership categories based on fuzzy truth values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruthMembership {
    Certain,     // 0.95-1.0: Ceremonial certainty level
    Probable,    // 0.75-0.95: High confidence
    Possible,    // 0.5-0.75: Gray area threshold
    Unlikely,    // 0.25-0.5: Low confidence
    Improbable,  // 0.05-0.25: Very low confidence
    False,       // 0.0-0.05: Essentially false
}

impl TruthMembership {
    /// Check if this membership level is sufficient for ceremonial use
    pub fn is_ceremonial_ready(&self) -> bool {
        matches!(self, TruthMembership::Certain)
    }

    /// Check if this membership level requires human judgment
    pub fn requires_human_judgment(&self) -> bool {
        matches!(self, TruthMembership::Possible | TruthMembership::Unlikely)
    }

    /// Get the typical confidence range for this membership
    pub fn confidence_range(&self) -> (f64, f64) {
        match self {
            TruthMembership::Certain => (0.95, 1.0),
            TruthMembership::Probable => (0.75, 0.95),
            TruthMembership::Possible => (0.5, 0.75),
            TruthMembership::Unlikely => (0.25, 0.5),
            TruthMembership::Improbable => (0.05, 0.25),
            TruthMembership::False => (0.0, 0.05),
        }
    }
}

impl fmt::Display for TruthMembership {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TruthMembership::Certain => write!(f, "Certain"),
            TruthMembership::Probable => write!(f, "Probable"),
            TruthMembership::Possible => write!(f, "Possible"),
            TruthMembership::Unlikely => write!(f, "Unlikely"),
            TruthMembership::Improbable => write!(f, "Improbable"),
            TruthMembership::False => write!(f, "False"),
        }
    }
}

/// Truth thresholds configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthThresholds {
    pub certain: f64,
    pub probable: f64,
    pub possible: f64,
    pub unlikely: f64,
    pub improbable: f64,
}

impl Default for TruthThresholds {
    fn default() -> Self {
        Self {
            certain: 0.95,
            probable: 0.75,
            possible: 0.50,
            unlikely: 0.25,
            improbable: 0.05,
        }
    }
}

/// A fuzzy result that includes the value, confidence, and uncertainty information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyResult<T> {
    pub id: Uuid,
    pub value: T,
    pub confidence: FuzzyTruth,
    pub uncertainty_sources: Vec<UncertaintySource>,
    pub gray_areas: Vec<GrayArea>,
    pub confidence_intervals: ConfidenceInterval,
    pub timestamp: DateTime<Utc>,
}

impl<T> FuzzyResult<T> {
    /// Create a new fuzzy result
    pub fn new(
        value: T,
        confidence: FuzzyTruth,
        uncertainty_sources: Vec<UncertaintySource>,
        gray_areas: Vec<GrayArea>,
        confidence_intervals: ConfidenceInterval,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            value,
            confidence,
            uncertainty_sources,
            gray_areas,
            confidence_intervals,
            timestamp: Utc::now(),
        }
    }

    /// Check if this result is in a gray area
    pub fn is_gray_area(&self, range: [f64; 2]) -> bool {
        self.confidence.is_gray_area(range)
    }

    /// Get the truth membership for this result
    pub fn membership(&self, thresholds: &TruthThresholds) -> TruthMembership {
        self.confidence.to_membership(thresholds)
    }

    /// Check if this result requires human judgment
    pub fn requires_human_judgment(&self, thresholds: &TruthThresholds) -> bool {
        self.membership(thresholds).requires_human_judgment()
    }

    /// Map the value while preserving fuzzy information
    pub fn map<U, F>(self, f: F) -> FuzzyResult<U>
    where
        F: FnOnce(T) -> U,
    {
        FuzzyResult {
            id: self.id,
            value: f(self.value),
            confidence: self.confidence,
            uncertainty_sources: self.uncertainty_sources,
            gray_areas: self.gray_areas,
            confidence_intervals: self.confidence_intervals,
            timestamp: self.timestamp,
        }
    }
}

/// Confidence interval for fuzzy results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64, // e.g., 0.95 for 95% confidence
}

impl ConfidenceInterval {
    /// Create a new confidence interval
    pub fn new(lower_bound: f64, upper_bound: f64, confidence_level: f64) -> Self {
        Self {
            lower_bound: lower_bound.clamp(0.0, 1.0),
            upper_bound: upper_bound.clamp(0.0, 1.0),
            confidence_level: confidence_level.clamp(0.0, 1.0),
        }
    }

    /// Get the width of the confidence interval
    pub fn width(&self) -> f64 {
        self.upper_bound - self.lower_bound
    }

    /// Check if a value falls within this interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower_bound && value <= self.upper_bound
    }
}

/// Fuzzy logic engine that orchestrates all fuzzy operations
#[derive(Debug, Clone)]
pub struct FuzzyLogicEngine {
    pub thresholds: TruthThresholds,
    pub operators: FuzzyOperators,
    pub gray_area_config: GrayAreaConfig,
}

impl FuzzyLogicEngine {
    /// Create a new fuzzy logic engine
    pub fn new(
        thresholds: TruthThresholds,
        operators: FuzzyOperators,
        gray_area_config: GrayAreaConfig,
    ) -> Self {
        Self {
            thresholds,
            operators,
            gray_area_config,
        }
    }

    /// Analyze the truth spectrum of a statement
    pub async fn analyze_truth_spectrum(&self, statement: &str) -> Result<TruthSpectrum> {
        // This would integrate with various analysis modules
        // For now, we'll create a basic implementation
        let factual_accuracy = self.assess_factual_accuracy(statement).await?;
        let contextual_relevance = self.assess_contextual_relevance(statement).await?;
        let temporal_validity = self.assess_temporal_validity(statement).await?;
        let source_credibility = self.assess_source_credibility(statement).await?;

        let overall_truth = self.fuzzy_weighted_average(&[
            (factual_accuracy, 0.4),
            (contextual_relevance, 0.25),
            (temporal_validity, 0.2),
            (source_credibility, 0.15),
        ])?;

        let membership = overall_truth.to_membership(&self.thresholds);
        let gray_areas = self.identify_gray_areas(&[
            ("factual_accuracy", factual_accuracy),
            ("contextual_relevance", contextual_relevance),
            ("temporal_validity", temporal_validity),
            ("source_credibility", source_credibility),
        ])?;

        Ok(TruthSpectrum {
            id: Uuid::new_v4(),
            overall_truth,
            membership,
            components: vec![
                ("factual_accuracy".to_string(), factual_accuracy),
                ("contextual_relevance".to_string(), contextual_relevance),
                ("temporal_validity".to_string(), temporal_validity),
                ("source_credibility".to_string(), source_credibility),
            ],
            uncertainty_factors: vec![], // Would be populated by analysis
            gray_areas,
            timestamp: Utc::now(),
        })
    }

    /// Weighted average of fuzzy truth values
    pub fn fuzzy_weighted_average(&self, values: &[(FuzzyTruth, f64)]) -> Result<FuzzyTruth> {
        if values.is_empty() {
            return Err(anyhow::anyhow!("Cannot compute weighted average of empty values"));
        }

        let total_weight: f64 = values.iter().map(|(_, weight)| weight).sum();
        if total_weight == 0.0 {
            return Err(anyhow::anyhow!("Total weight cannot be zero"));
        }

        let weighted_sum: f64 = values
            .iter()
            .map(|(truth, weight)| truth.value() * weight)
            .sum();

        Ok(FuzzyTruth::new_unchecked(weighted_sum / total_weight))
    }

    /// Identify gray areas in a set of truth values
    pub fn identify_gray_areas(&self, values: &[(&str, FuzzyTruth)]) -> Result<Vec<GrayArea>> {
        let mut gray_areas = Vec::new();

        for (domain, truth) in values {
            if truth.is_gray_area(self.gray_area_config.detection_range) {
                let transition_type = self.determine_transition_type(*truth);
                
                gray_areas.push(GrayArea {
                    id: Uuid::new_v4(),
                    domain: domain.to_string(),
                    confidence_range: (
                        truth.value() - 0.1,
                        truth.value() + 0.1,
                    ),
                    transition_type,
                    ambiguity_factors: vec![], // Would be populated by deeper analysis
                    requires_human_judgment: truth.value() < self.gray_area_config.human_judgment_threshold,
                    philosophical_implications: format!(
                        "Gray area detected in {}: truth value {} represents transition between certainty and uncertainty",
                        domain, truth
                    ),
                    timestamp: Utc::now(),
                });
            }
        }

        Ok(gray_areas)
    }

    /// Determine the type of truth transition
    fn determine_transition_type(&self, confidence: FuzzyTruth) -> TransitionType {
        match confidence.value() {
            x if x >= 0.6 && x <= 0.7 => TransitionType::CertaintyToUncertainty,
            x if x >= 0.5 && x <= 0.6 => TransitionType::PossibleToUnlikely,
            x if x >= 0.4 && x <= 0.5 => TransitionType::UncertaintyToImprobability,
            _ => TransitionType::Unknown,
        }
    }

    // Placeholder methods for truth assessment - these would integrate with other systems
    async fn assess_factual_accuracy(&self, _statement: &str) -> Result<FuzzyTruth> {
        // This would integrate with fact-checking systems
        Ok(FuzzyTruth::new_unchecked(0.8))
    }

    async fn assess_contextual_relevance(&self, _statement: &str) -> Result<FuzzyTruth> {
        // This would analyze context and relevance
        Ok(FuzzyTruth::new_unchecked(0.7))
    }

    async fn assess_temporal_validity(&self, _statement: &str) -> Result<FuzzyTruth> {
        // This would check if the statement is still valid over time
        Ok(FuzzyTruth::new_unchecked(0.9))
    }

    async fn assess_source_credibility(&self, _statement: &str) -> Result<FuzzyTruth> {
        // This would evaluate the credibility of the source
        Ok(FuzzyTruth::new_unchecked(0.6))
    }
}

impl Default for FuzzyLogicEngine {
    fn default() -> Self {
        Self::new(
            TruthThresholds::default(),
            FuzzyOperators::default(),
            GrayAreaConfig::default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_truth_creation() {
        assert!(FuzzyTruth::new(0.5).is_ok());
        assert!(FuzzyTruth::new(-0.1).is_err());
        assert!(FuzzyTruth::new(1.1).is_err());
    }

    #[test]
    fn test_truth_membership() {
        let thresholds = TruthThresholds::default();
        let certain = FuzzyTruth::new_unchecked(0.96);
        let probable = FuzzyTruth::new_unchecked(0.8);
        let possible = FuzzyTruth::new_unchecked(0.6);

        assert_eq!(certain.to_membership(&thresholds), TruthMembership::Certain);
        assert_eq!(probable.to_membership(&thresholds), TruthMembership::Probable);
        assert_eq!(possible.to_membership(&thresholds), TruthMembership::Possible);
    }

    #[test]
    fn test_gray_area_detection() {
        let truth = FuzzyTruth::new_unchecked(0.55);
        assert!(truth.is_gray_area([0.4, 0.7]));
        assert!(!truth.is_gray_area([0.8, 0.9]));
    }

    #[tokio::test]
    async fn test_fuzzy_weighted_average() {
        let engine = FuzzyLogicEngine::default();
        let values = [
            (FuzzyTruth::new_unchecked(0.8), 0.4),
            (FuzzyTruth::new_unchecked(0.6), 0.6),
        ];

        let result = engine.fuzzy_weighted_average(&values).unwrap();
        assert!((result.value() - 0.68).abs() < 0.01);
    }
} 