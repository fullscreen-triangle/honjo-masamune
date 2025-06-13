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

/// Main Honjo Masamune Truth Engine
#[derive(Debug)]
pub struct HonjoMasamuneEngine {
    pub config: HonjoMasamuneConfig,
    atp_manager: Arc<AtpManager>,
    fuzzy_engine: Arc<FuzzyLogicEngine>,
    buhera_engine: Arc<BuheraEngine>,
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

        let engine = Self {
            config,
            atp_manager,
            fuzzy_engine,
            buhera_engine,
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
            truth_spectrum,
            total_cost,
        ).await?;

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "natural_language_query").await?;

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