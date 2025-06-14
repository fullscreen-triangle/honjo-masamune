//! Conversational AI Interface
//! 
//! Provides natural language interaction for truth synthesis,
//! automatic assumption validation, and plain language explanations.

use crate::zero_config::{AiConfig, InterpretedRequest};
use anyhow::Result;
use fuzzy_logic_core::FuzzyResult;
use serde::{Deserialize, Serialize};
use tracing::{info, debug};

#[derive(Debug)]
pub struct ConversationalInterface {
    config: AiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionAnalysis {
    pub intent: QueryIntent,
    pub entities: Vec<String>,
    pub confidence_required: f64,
    pub validation_needed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryIntent {
    TruthSynthesis,
    DataAnalysis,
    PatternDiscovery,
    FactVerification,
    Comparison,
    Explanation,
}

impl ConversationalInterface {
    pub async fn new(config: AiConfig) -> Result<Self> {
        info!("Initializing conversational AI interface");
        debug!("AI config: {:?}", config);
        
        Ok(Self { config })
    }
    
    /// Interpret natural language question into structured request
    pub async fn interpret_question(&self, question: &str) -> Result<InterpretedRequest> {
        info!("Interpreting natural language question: {}", question);
        
        // Analyze the question
        let analysis = self.analyze_question(question).await?;
        
        // Convert to structured query
        let structured_query = self.convert_to_structured_query(question, &analysis).await?;
        
        // Determine validation requirements
        let requires_validation = self.determine_validation_needs(&analysis);
        
        let request = InterpretedRequest {
            structured_query,
            requires_validation,
            confidence_threshold: analysis.confidence_required,
        };
        
        debug!("Interpreted request: {:?}", request);
        Ok(request)
    }
    
    /// Explain technical results in natural language
    pub async fn explain_result(&self, result: &FuzzyResult<String>, original_question: &str) -> Result<String> {
        info!("Explaining result for question: {}", original_question);
        
        let explanation = match self.config.explanation_detail_level.as_str() {
            "simple" => self.create_simple_explanation(result, original_question).await?,
            "detailed" => self.create_detailed_explanation(result, original_question).await?,
            "technical" => self.create_technical_explanation(result, original_question).await?,
            _ => self.create_detailed_explanation(result, original_question).await?,
        };
        
        Ok(explanation)
    }
    
    /// Validate assumptions automatically and explain decisions
    pub async fn validate_and_explain_assumptions(&self, question: &str) -> Result<String> {
        if !self.config.auto_assumption_validation {
            return Ok("Assumption validation is disabled.".to_string());
        }
        
        info!("Validating assumptions for question: {}", question);
        
        let assumptions = self.identify_assumptions(question).await?;
        let validation_results = self.validate_assumptions(&assumptions).await?;
        
        let explanation = self.explain_assumption_validation(&assumptions, &validation_results).await?;
        
        Ok(explanation)
    }
    
    async fn analyze_question(&self, question: &str) -> Result<QuestionAnalysis> {
        // Simple question analysis - in real implementation would use NLP
        let question_lower = question.to_lowercase();
        
        // Determine intent
        let intent = if question_lower.contains("true") || question_lower.contains("fact") {
            QueryIntent::FactVerification
        } else if question_lower.contains("compare") || question_lower.contains("versus") {
            QueryIntent::Comparison
        } else if question_lower.contains("pattern") || question_lower.contains("trend") {
            QueryIntent::PatternDiscovery
        } else if question_lower.contains("analyze") || question_lower.contains("analysis") {
            QueryIntent::DataAnalysis
        } else if question_lower.contains("explain") || question_lower.contains("why") {
            QueryIntent::Explanation
        } else {
            QueryIntent::TruthSynthesis
        };
        
        // Extract entities (simplified)
        let entities = self.extract_entities(&question_lower);
        
        // Determine confidence requirements
        let confidence_required = if question_lower.contains("certain") || question_lower.contains("sure") {
            0.9
        } else if question_lower.contains("likely") || question_lower.contains("probable") {
            0.7
        } else {
            0.6
        };
        
        // Determine if validation is needed
        let validation_needed = question_lower.contains("verify") || 
                               question_lower.contains("confirm") ||
                               question_lower.contains("validate");
        
        Ok(QuestionAnalysis {
            intent,
            entities,
            confidence_required,
            validation_needed,
        })
    }
    
    fn extract_entities(&self, question: &str) -> Vec<String> {
        // Simplified entity extraction
        let words: Vec<&str> = question.split_whitespace().collect();
        let mut entities = Vec::new();
        
        // Look for capitalized words (potential entities)
        for word in words {
            if word.len() > 2 && word.chars().next().unwrap().is_uppercase() {
                entities.push(word.to_string());
            }
        }
        
        // Look for quoted phrases
        if let Some(start) = question.find('"') {
            if let Some(end) = question[start + 1..].find('"') {
                let quoted = &question[start + 1..start + 1 + end];
                entities.push(quoted.to_string());
            }
        }
        
        entities
    }
    
    async fn convert_to_structured_query(&self, question: &str, analysis: &QuestionAnalysis) -> Result<String> {
        // Convert natural language to structured query
        let result = match analysis.intent {
            QueryIntent::TruthSynthesis => {
                format!("SYNTHESIZE_TRUTH: {}", question)
            },
            QueryIntent::DataAnalysis => {
                format!("ANALYZE_DATA: {} ENTITIES: {:?}", question, analysis.entities)
            },
            QueryIntent::PatternDiscovery => {
                format!("DISCOVER_PATTERNS: {} CONFIDENCE: {}", question, analysis.confidence_required)
            },
            QueryIntent::FactVerification => {
                format!("VERIFY_FACT: {} ENTITIES: {:?}", question, analysis.entities)
            },
            QueryIntent::Comparison => {
                format!("COMPARE: {} ENTITIES: {:?}", question, analysis.entities)
            },
            QueryIntent::Explanation => {
                format!("EXPLAIN: {}", question)
            },
        };
        Ok(result)
    }
    
    fn determine_validation_needs(&self, analysis: &QuestionAnalysis) -> bool {
        analysis.validation_needed || 
        analysis.confidence_required > 0.8 ||
        matches!(analysis.intent, QueryIntent::FactVerification | QueryIntent::TruthSynthesis)
    }
    
    async fn create_simple_explanation(&self, result: &FuzzyResult<String>, _question: &str) -> Result<String> {
        let confidence_percent = (result.confidence.value() * 100.0) as u32;
        
        Ok(format!(
            "Based on my analysis, I'm {}% confident that: {}\n\n{}",
            confidence_percent,
            result.value,
            if confidence_percent >= 80 {
                "This is a high-confidence result."
            } else if confidence_percent >= 60 {
                "This is a moderate-confidence result."
            } else {
                "This is a low-confidence result that may require additional validation."
            }
        ))
    }
    
    async fn create_detailed_explanation(&self, result: &FuzzyResult<String>, question: &str) -> Result<String> {
        let confidence_percent = (result.confidence.value() * 100.0) as u32;
        let confidence_interval = &result.confidence_interval;
        
        let mut explanation = format!(
            "## Analysis Results for: \"{}\"\n\n",
            question
        );
        
        explanation.push_str(&format!(
            "**Finding:** {}\n\n",
            result.value
        ));
        
        explanation.push_str(&format!(
            "**Confidence Level:** {}% (Range: {:.1}% - {:.1}%)\n\n",
            confidence_percent,
            confidence_interval.lower_bound() * 100.0,
            confidence_interval.upper_bound() * 100.0
        ));
        
        if !result.uncertainty_sources.is_empty() {
            explanation.push_str("**Uncertainty Sources:**\n");
            for source in &result.uncertainty_sources {
                explanation.push_str(&format!("- {}\n", source));
            }
            explanation.push('\n');
        }
        
        if !result.gray_areas.is_empty() {
            explanation.push_str("**Areas Requiring Human Judgment:**\n");
            for area in &result.gray_areas {
                explanation.push_str(&format!("- {}\n", area));
            }
            explanation.push('\n');
        }
        
        explanation.push_str(&self.get_confidence_interpretation(confidence_percent));
        
        Ok(explanation)
    }
    
    async fn create_technical_explanation(&self, result: &FuzzyResult<String>, question: &str) -> Result<String> {
        let mut explanation = format!(
            "## Technical Analysis Report\n\n**Query:** {}\n\n",
            question
        );
        
        explanation.push_str(&format!(
            "**Result:** {}\n\n",
            result.value
        ));
        
        explanation.push_str(&format!(
            "**Truth Value:** {:.6}\n",
            result.confidence.value()
        ));
        
        let ci = &result.confidence_interval;
        explanation.push_str(&format!(
            "**Confidence Interval:** [{:.6}, {:.6}] at {:.1}% confidence\n\n",
            ci.lower_bound(),
            ci.upper_bound(),
            ci.confidence_level() * 100.0
        ));
        
        if !result.uncertainty_sources.is_empty() {
            explanation.push_str("**Uncertainty Analysis:**\n");
            for (i, source) in result.uncertainty_sources.iter().enumerate() {
                explanation.push_str(&format!("{}. {}\n", i + 1, source));
            }
            explanation.push('\n');
        }
        
        if !result.gray_areas.is_empty() {
            explanation.push_str("**Gray Area Analysis:**\n");
            for (i, area) in result.gray_areas.iter().enumerate() {
                explanation.push_str(&format!("{}. {}\n", i + 1, area));
            }
            explanation.push('\n');
        }
        
        Ok(explanation)
    }
    
    fn get_confidence_interpretation(&self, confidence_percent: u32) -> String {
        match confidence_percent {
            90..=100 => "**Interpretation:** Very high confidence. This result is highly reliable and can be used for decision-making with minimal additional validation.".to_string(),
            80..=89 => "**Interpretation:** High confidence. This result is reliable but may benefit from additional validation for critical decisions.".to_string(),
            70..=79 => "**Interpretation:** Moderate confidence. This result provides useful insights but should be validated with additional sources for important decisions.".to_string(),
            60..=69 => "**Interpretation:** Low-moderate confidence. This result suggests a direction but requires additional validation before use in decision-making.".to_string(),
            50..=59 => "**Interpretation:** Low confidence. This result is uncertain and requires significant additional validation.".to_string(),
            _ => "**Interpretation:** Very low confidence. This result is highly uncertain and should not be used for decision-making without extensive additional validation.".to_string(),
        }
    }
    
    async fn identify_assumptions(&self, question: &str) -> Result<Vec<String>> {
        // Simplified assumption identification
        let mut assumptions = Vec::new();
        
        let question_lower = question.to_lowercase();
        
        if question_lower.contains("all") || question_lower.contains("every") {
            assumptions.push("Universal quantification assumed".to_string());
        }
        
        if question_lower.contains("always") || question_lower.contains("never") {
            assumptions.push("Temporal absoluteness assumed".to_string());
        }
        
        if question_lower.contains("because") || question_lower.contains("causes") {
            assumptions.push("Causal relationship assumed".to_string());
        }
        
        if question_lower.contains("should") || question_lower.contains("must") {
            assumptions.push("Normative judgment assumed".to_string());
        }
        
        Ok(assumptions)
    }
    
    async fn validate_assumptions(&self, assumptions: &[String]) -> Result<Vec<bool>> {
        // Simplified assumption validation
        let mut results = Vec::new();
        
        for assumption in assumptions {
            // In real implementation, would perform sophisticated validation
            let is_valid = !assumption.contains("Universal") && !assumption.contains("absoluteness");
            results.push(is_valid);
        }
        
        Ok(results)
    }
    
    async fn explain_assumption_validation(&self, assumptions: &[String], results: &[bool]) -> Result<String> {
        let mut explanation = String::from("## Assumption Validation\n\n");
        
        for (assumption, &is_valid) in assumptions.iter().zip(results.iter()) {
            explanation.push_str(&format!(
                "**{}:** {}\n",
                assumption,
                if is_valid {
                    "✅ Valid - This assumption is reasonable for the analysis."
                } else {
                    "⚠️ Questionable - This assumption may limit the validity of results."
                }
            ));
        }
        
        if results.iter().any(|&r| !r) {
            explanation.push_str("\n**Recommendation:** Consider rephrasing your question to avoid problematic assumptions for more reliable results.\n");
        }
        
        Ok(explanation)
    }
} 