//! Truth Spectrum Analysis
//! 
//! This module provides comprehensive truth analysis across multiple dimensions
//! including factual accuracy, contextual relevance, temporal validity, and more.

use crate::{FuzzyTruth, GrayArea};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Comprehensive truth spectrum analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthSpectrum {
    pub id: Uuid,
    pub overall_truth: FuzzyTruth,
    pub membership: crate::TruthMembership,
    pub components: Vec<(String, FuzzyTruth)>,
    pub uncertainty_factors: Vec<UncertaintyFactor>,
    pub gray_areas: Vec<GrayArea>,
    pub timestamp: DateTime<Utc>,
}

/// Uncertainty factor in truth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyFactor {
    pub source: String,
    pub impact: FuzzyTruth,
    pub description: String,
}

/// Gray area configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayAreaConfig {
    pub detection_range: [f64; 2],
    pub processing_overhead: f64,
    pub human_judgment_threshold: f64,
}

impl Default for GrayAreaConfig {
    fn default() -> Self {
        Self {
            detection_range: [0.4, 0.7],
            processing_overhead: 1.5,
            human_judgment_threshold: 0.5,
        }
    }
} 