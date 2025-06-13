//! Linguistic Hedges for Fuzzy Logic
//! 
//! This module implements linguistic hedges that modify fuzzy truth values,
//! such as "very", "somewhat", "extremely", etc.

use crate::FuzzyTruth;
use serde::{Deserialize, Serialize};

/// Configuration for linguistic hedges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticHedges {
    pub very_concentration: f64,
    pub somewhat_dilation: f64,
}

impl Default for LinguisticHedges {
    fn default() -> Self {
        Self {
            very_concentration: 2.0,
            somewhat_dilation: 0.5,
        }
    }
}

/// Apply "very" hedge (concentration)
pub fn very(truth: FuzzyTruth, power: f64) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(truth.value().powf(power))
}

/// Apply "somewhat" hedge (dilation)
pub fn somewhat(truth: FuzzyTruth, power: f64) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(truth.value().powf(power))
}

/// Apply "extremely" hedge (strong concentration)
pub fn extremely(truth: FuzzyTruth) -> FuzzyTruth {
    very(truth, 3.0)
}

/// Apply "slightly" hedge (weak dilation)
pub fn slightly(truth: FuzzyTruth) -> FuzzyTruth {
    somewhat(truth, 0.3)
}

/// Apply "more or less" hedge
pub fn more_or_less(truth: FuzzyTruth) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(truth.value().sqrt())
}

/// Apply "not very" hedge
pub fn not_very(truth: FuzzyTruth) -> FuzzyTruth {
    let very_truth = very(truth, 2.0);
    FuzzyTruth::new_unchecked(1.0 - very_truth.value())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_very_hedge() {
        let truth = FuzzyTruth::new_unchecked(0.8);
        let very_truth = very(truth, 2.0);
        assert!((very_truth.value() - 0.64).abs() < 0.001);
    }

    #[test]
    fn test_somewhat_hedge() {
        let truth = FuzzyTruth::new_unchecked(0.64);
        let somewhat_truth = somewhat(truth, 0.5);
        assert!((somewhat_truth.value() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_extremely_hedge() {
        let truth = FuzzyTruth::new_unchecked(0.8);
        let extremely_truth = extremely(truth);
        assert!((extremely_truth.value() - 0.512).abs() < 0.001);
    }
} 