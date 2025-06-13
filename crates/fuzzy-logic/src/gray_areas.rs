//! Gray Area Detection and Processing
//! 
//! This module handles the detection and processing of gray areas in truth
//! analysis - regions where truth values are ambiguous and may require
//! human judgment or additional processing.

use crate::FuzzyTruth;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Gray area in truth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayArea {
    pub id: Uuid,
    pub domain: String,
    pub confidence_range: (f64, f64),
    pub transition_type: TransitionType,
    pub ambiguity_factors: Vec<AmbiguityFactor>,
    pub requires_human_judgment: bool,
    pub philosophical_implications: String,
    pub timestamp: DateTime<Utc>,
}

/// Types of truth transitions in gray areas
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionType {
    CertaintyToUncertainty,
    PossibleToUnlikely,
    UncertaintyToImprobability,
    Unknown,
}

/// Factor contributing to ambiguity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityFactor {
    pub source: String,
    pub impact: FuzzyTruth,
    pub description: String,
    pub resolution_strategy: Option<String>,
}

impl GrayArea {
    /// Create a new gray area
    pub fn new(
        domain: String,
        confidence_range: (f64, f64),
        transition_type: TransitionType,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            domain,
            confidence_range,
            transition_type,
            ambiguity_factors: Vec::new(),
            requires_human_judgment: true,
            philosophical_implications: String::new(),
            timestamp: Utc::now(),
        }
    }

    /// Check if a confidence value falls within this gray area
    pub fn contains(&self, confidence: f64) -> bool {
        confidence >= self.confidence_range.0 && confidence <= self.confidence_range.1
    }

    /// Get the width of the gray area
    pub fn width(&self) -> f64 {
        self.confidence_range.1 - self.confidence_range.0
    }

    /// Add an ambiguity factor
    pub fn add_ambiguity_factor(&mut self, factor: AmbiguityFactor) {
        self.ambiguity_factors.push(factor);
    }
}

impl TransitionType {
    /// Get a description of the transition type
    pub fn description(&self) -> &'static str {
        match self {
            TransitionType::CertaintyToUncertainty => "Transition from certainty to uncertainty",
            TransitionType::PossibleToUnlikely => "Transition from possible to unlikely",
            TransitionType::UncertaintyToImprobability => "Transition from uncertainty to improbability",
            TransitionType::Unknown => "Unknown transition type",
        }
    }
} 