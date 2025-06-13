//! Uncertainty Handling and Propagation
//! 
//! This module provides mechanisms for tracking, propagating, and managing
//! uncertainty throughout fuzzy logic operations.

use crate::FuzzyTruth;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Source of uncertainty in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintySource {
    pub id: Uuid,
    pub source_type: UncertaintyType,
    pub magnitude: FuzzyTruth,
    pub description: String,
    pub propagation_factor: f64,
    pub timestamp: DateTime<Utc>,
}

/// Types of uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UncertaintyType {
    DataIncomplete,
    SourceUnreliable,
    TemporalDecay,
    ConflictingEvidence,
    ModelLimitation,
    HumanSubjectivity,
    MeasurementError,
    Unknown,
}

impl UncertaintySource {
    /// Create a new uncertainty source
    pub fn new(
        source_type: UncertaintyType,
        magnitude: FuzzyTruth,
        description: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            source_type,
            magnitude,
            description,
            propagation_factor: 1.0,
            timestamp: Utc::now(),
        }
    }

    /// Calculate the impact of this uncertainty on a confidence value
    pub fn impact_on_confidence(&self, base_confidence: FuzzyTruth) -> FuzzyTruth {
        let reduction = self.magnitude.value() * self.propagation_factor;
        let new_confidence = (base_confidence.value() - reduction).max(0.0);
        FuzzyTruth::new_unchecked(new_confidence)
    }
}

impl UncertaintyType {
    /// Get a description of the uncertainty type
    pub fn description(&self) -> &'static str {
        match self {
            UncertaintyType::DataIncomplete => "Incomplete or missing data",
            UncertaintyType::SourceUnreliable => "Unreliable or questionable source",
            UncertaintyType::TemporalDecay => "Information becoming outdated over time",
            UncertaintyType::ConflictingEvidence => "Contradictory evidence from multiple sources",
            UncertaintyType::ModelLimitation => "Limitations in the analysis model",
            UncertaintyType::HumanSubjectivity => "Human bias or subjective interpretation",
            UncertaintyType::MeasurementError => "Errors in measurement or observation",
            UncertaintyType::Unknown => "Unknown source of uncertainty",
        }
    }

    /// Get the typical propagation factor for this uncertainty type
    pub fn default_propagation_factor(&self) -> f64 {
        match self {
            UncertaintyType::DataIncomplete => 0.8,
            UncertaintyType::SourceUnreliable => 0.9,
            UncertaintyType::TemporalDecay => 0.6,
            UncertaintyType::ConflictingEvidence => 1.0,
            UncertaintyType::ModelLimitation => 0.7,
            UncertaintyType::HumanSubjectivity => 0.5,
            UncertaintyType::MeasurementError => 0.8,
            UncertaintyType::Unknown => 1.0,
        }
    }
} 