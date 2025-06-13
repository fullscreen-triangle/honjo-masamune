//! Fuzzy Logic Operators
//! 
//! This module implements the core fuzzy logic operators including T-norms,
//! T-conorms, implications, and other fuzzy operations used throughout
//! the Honjo Masamune truth engine.

use crate::FuzzyTruth;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for fuzzy operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyOperators {
    pub t_norm: TNormType,
    pub t_conorm: TConormType,
    pub implication: ImplicationType,
}

impl Default for FuzzyOperators {
    fn default() -> Self {
        Self {
            t_norm: TNormType::Minimum,
            t_conorm: TConormType::Maximum,
            implication: ImplicationType::KleeneDienes,
        }
    }
}

/// Types of T-norms (fuzzy AND operations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TNormType {
    Minimum,        // min(a, b)
    Product,        // a * b
    LukasiewiczAnd, // max(0, a + b - 1)
    DrasticProduct, // Special case T-norm
}

/// Types of T-conorms (fuzzy OR operations)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TConormType {
    Maximum,        // max(a, b)
    ProbabilisticSum, // a + b - a*b
    LukasiewiczOr,  // min(1, a + b)
    DrasticSum,     // Special case T-conorm
}

/// Types of fuzzy implications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplicationType {
    KleeneDienes,   // max(1-a, b)
    Lukasiewicz,    // min(1, 1-a+b)
    Godel,          // if a <= b then 1 else b
    Goguen,         // if a <= b then 1 else b/a
}

/// Fuzzy AND operation (T-norm)
pub fn fuzzy_and(a: FuzzyTruth, b: FuzzyTruth, t_norm: &TNormType) -> FuzzyTruth {
    let result = match t_norm {
        TNormType::Minimum => a.value().min(b.value()),
        TNormType::Product => a.value() * b.value(),
        TNormType::LukasiewiczAnd => (a.value() + b.value() - 1.0).max(0.0),
        TNormType::DrasticProduct => {
            if a.value() == 1.0 {
                b.value()
            } else if b.value() == 1.0 {
                a.value()
            } else {
                0.0
            }
        }
    };
    FuzzyTruth::new_unchecked(result)
}

/// Fuzzy OR operation (T-conorm)
pub fn fuzzy_or(a: FuzzyTruth, b: FuzzyTruth, t_conorm: &TConormType) -> FuzzyTruth {
    let result = match t_conorm {
        TConormType::Maximum => a.value().max(b.value()),
        TConormType::ProbabilisticSum => a.value() + b.value() - a.value() * b.value(),
        TConormType::LukasiewiczOr => (a.value() + b.value()).min(1.0),
        TConormType::DrasticSum => {
            if a.value() == 0.0 {
                b.value()
            } else if b.value() == 0.0 {
                a.value()
            } else {
                1.0
            }
        }
    };
    FuzzyTruth::new_unchecked(result)
}

/// Fuzzy NOT operation (standard complement)
pub fn fuzzy_not(a: FuzzyTruth) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(1.0 - a.value())
}

/// Fuzzy implication
pub fn fuzzy_implies(a: FuzzyTruth, b: FuzzyTruth, implication: &ImplicationType) -> FuzzyTruth {
    let result = match implication {
        ImplicationType::KleeneDienes => (1.0 - a.value()).max(b.value()),
        ImplicationType::Lukasiewicz => (1.0 - a.value() + b.value()).min(1.0),
        ImplicationType::Godel => {
            if a.value() <= b.value() {
                1.0
            } else {
                b.value()
            }
        }
        ImplicationType::Goguen => {
            if a.value() <= b.value() {
                1.0
            } else if a.value() == 0.0 {
                1.0
            } else {
                b.value() / a.value()
            }
        }
    };
    FuzzyTruth::new_unchecked(result)
}

/// Fuzzy equivalence (biconditional)
pub fn fuzzy_equivalent(a: FuzzyTruth, b: FuzzyTruth, operators: &FuzzyOperators) -> FuzzyTruth {
    // (a → b) ∧ (b → a)
    let a_implies_b = fuzzy_implies(a, b, &operators.implication);
    let b_implies_a = fuzzy_implies(b, a, &operators.implication);
    fuzzy_and(a_implies_b, b_implies_a, &operators.t_norm)
}

/// Fuzzy exclusive OR
pub fn fuzzy_xor(a: FuzzyTruth, b: FuzzyTruth, operators: &FuzzyOperators) -> FuzzyTruth {
    // (a ∨ b) ∧ ¬(a ∧ b)
    let a_or_b = fuzzy_or(a, b, &operators.t_conorm);
    let a_and_b = fuzzy_and(a, b, &operators.t_norm);
    let not_a_and_b = fuzzy_not(a_and_b);
    fuzzy_and(a_or_b, not_a_and_b, &operators.t_norm)
}

/// Fuzzy aggregation using ordered weighted averaging (OWA)
pub fn fuzzy_owa(values: &[FuzzyTruth], weights: &[f64]) -> Result<FuzzyTruth> {
    if values.len() != weights.len() {
        return Err(anyhow::anyhow!("Values and weights must have the same length"));
    }

    if values.is_empty() {
        return Err(anyhow::anyhow!("Cannot aggregate empty values"));
    }

    // Sort values in descending order
    let mut sorted_values: Vec<f64> = values.iter().map(|v| v.value()).collect();
    sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

    // Compute weighted sum
    let weighted_sum: f64 = sorted_values
        .iter()
        .zip(weights.iter())
        .map(|(value, weight)| value * weight)
        .sum();

    let weight_sum: f64 = weights.iter().sum();
    if weight_sum == 0.0 {
        return Err(anyhow::anyhow!("Sum of weights cannot be zero"));
    }

    Ok(FuzzyTruth::new_unchecked(weighted_sum / weight_sum))
}

/// Fuzzy power mean aggregation
pub fn fuzzy_power_mean(values: &[FuzzyTruth], power: f64) -> Result<FuzzyTruth> {
    if values.is_empty() {
        return Err(anyhow::anyhow!("Cannot aggregate empty values"));
    }

    if power == 0.0 {
        // Geometric mean
        let product: f64 = values.iter().map(|v| v.value()).product();
        let result = product.powf(1.0 / values.len() as f64);
        return Ok(FuzzyTruth::new_unchecked(result));
    }

    let sum: f64 = values.iter().map(|v| v.value().powf(power)).sum();
    let mean = sum / values.len() as f64;
    let result = mean.powf(1.0 / power);

    Ok(FuzzyTruth::new_unchecked(result))
}

/// Fuzzy median aggregation
pub fn fuzzy_median(values: &[FuzzyTruth]) -> Result<FuzzyTruth> {
    if values.is_empty() {
        return Err(anyhow::anyhow!("Cannot compute median of empty values"));
    }

    let mut sorted_values: Vec<f64> = values.iter().map(|v| v.value()).collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = if sorted_values.len() % 2 == 0 {
        let mid = sorted_values.len() / 2;
        (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };

    Ok(FuzzyTruth::new_unchecked(median))
}

/// Fuzzy consensus operator (finds the value with maximum agreement)
pub fn fuzzy_consensus(values: &[FuzzyTruth], tolerance: f64) -> Result<FuzzyTruth> {
    if values.is_empty() {
        return Err(anyhow::anyhow!("Cannot find consensus of empty values"));
    }

    let mut best_value = values[0];
    let mut max_agreement = 0;

    for candidate in values {
        let agreement = values
            .iter()
            .filter(|v| (v.value() - candidate.value()).abs() <= tolerance)
            .count();

        if agreement > max_agreement {
            max_agreement = agreement;
            best_value = *candidate;
        }
    }

    Ok(best_value)
}

/// Fuzzy distance between two truth values
pub fn fuzzy_distance(a: FuzzyTruth, b: FuzzyTruth) -> f64 {
    (a.value() - b.value()).abs()
}

/// Fuzzy similarity between two truth values
pub fn fuzzy_similarity(a: FuzzyTruth, b: FuzzyTruth) -> FuzzyTruth {
    let distance = fuzzy_distance(a, b);
    FuzzyTruth::new_unchecked(1.0 - distance)
}

/// Fuzzy concentration (very operator)
pub fn fuzzy_concentration(a: FuzzyTruth, power: f64) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(a.value().powf(power))
}

/// Fuzzy dilation (somewhat operator)
pub fn fuzzy_dilation(a: FuzzyTruth, power: f64) -> FuzzyTruth {
    FuzzyTruth::new_unchecked(a.value().powf(power))
}

/// Fuzzy contrast intensification
pub fn fuzzy_contrast_intensification(a: FuzzyTruth) -> FuzzyTruth {
    let value = if a.value() <= 0.5 {
        2.0 * a.value().powi(2)
    } else {
        1.0 - 2.0 * (1.0 - a.value()).powi(2)
    };
    FuzzyTruth::new_unchecked(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_and_minimum() {
        let a = FuzzyTruth::new_unchecked(0.7);
        let b = FuzzyTruth::new_unchecked(0.3);
        let result = fuzzy_and(a, b, &TNormType::Minimum);
        assert_eq!(result.value(), 0.3);
    }

    #[test]
    fn test_fuzzy_or_maximum() {
        let a = FuzzyTruth::new_unchecked(0.7);
        let b = FuzzyTruth::new_unchecked(0.3);
        let result = fuzzy_or(a, b, &TConormType::Maximum);
        assert_eq!(result.value(), 0.7);
    }

    #[test]
    fn test_fuzzy_not() {
        let a = FuzzyTruth::new_unchecked(0.3);
        let result = fuzzy_not(a);
        assert_eq!(result.value(), 0.7);
    }

    #[test]
    fn test_fuzzy_implies_kleene_dienes() {
        let a = FuzzyTruth::new_unchecked(0.8);
        let b = FuzzyTruth::new_unchecked(0.6);
        let result = fuzzy_implies(a, b, &ImplicationType::KleeneDienes);
        assert_eq!(result.value(), 0.6); // max(1-0.8, 0.6) = max(0.2, 0.6) = 0.6
    }

    #[test]
    fn test_fuzzy_median() {
        let values = [
            FuzzyTruth::new_unchecked(0.1),
            FuzzyTruth::new_unchecked(0.5),
            FuzzyTruth::new_unchecked(0.9),
        ];
        let result = fuzzy_median(&values).unwrap();
        assert_eq!(result.value(), 0.5);
    }

    #[test]
    fn test_fuzzy_distance() {
        let a = FuzzyTruth::new_unchecked(0.7);
        let b = FuzzyTruth::new_unchecked(0.3);
        let distance = fuzzy_distance(a, b);
        assert_eq!(distance, 0.4);
    }

    #[test]
    fn test_fuzzy_concentration() {
        let a = FuzzyTruth::new_unchecked(0.8);
        let result = fuzzy_concentration(a, 2.0); // very(0.8) = 0.8^2 = 0.64
        assert!((result.value() - 0.64).abs() < 0.001);
    }
} 