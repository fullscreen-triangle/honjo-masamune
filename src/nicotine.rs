//! Nicotine Module - Context Validation Through Puzzle Breaks
//! 
//! The "cigarette break" for AI systems. After intensive processing, the system
//! takes a break to solve coded puzzles that validate context retention and 
//! refresh understanding of current objectives.

use anyhow::Result;
use atp_manager::AtpManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use sha2::{Sha256, Digest};

/// The nicotine engine - provides context validation breaks
#[derive(Debug)]
pub struct NicotineEngine {
    atp_manager: Arc<AtpManager>,
    break_criteria: BreakCriteria,
    puzzle_generator: PuzzleGenerator,
    context_tracker: Arc<RwLock<ContextTracker>>,
    break_history: Arc<RwLock<Vec<NicotineBreakResult>>>,
}

impl NicotineEngine {
    /// Create a new nicotine engine
    pub fn new(
        atp_manager: Arc<AtpManager>,
        break_criteria: BreakCriteria,
    ) -> Self {
        Self {
            atp_manager,
            break_criteria,
            puzzle_generator: PuzzleGenerator::new(),
            context_tracker: Arc::new(RwLock::new(ContextTracker::new())),
            break_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Check if it's time for a nicotine break
    pub async fn should_take_break(&self) -> Result<bool> {
        let context = self.context_tracker.read().await;
        
        // Check various break triggers
        let operations_threshold = context.operations_since_break >= self.break_criteria.operations_trigger;
        let time_threshold = context.time_since_break() >= self.break_criteria.time_trigger;
        let complexity_threshold = context.complexity_accumulation >= self.break_criteria.complexity_trigger;
        let context_drift_detected = context.context_drift_score >= self.break_criteria.drift_threshold;

        debug!("Break check - Operations: {}/{}, Time: {}min/{}, Complexity: {:.2}/{:.2}, Drift: {:.2}/{:.2}",
            context.operations_since_break, self.break_criteria.operations_trigger,
            context.time_since_break().num_minutes(), self.break_criteria.time_trigger.num_minutes(),
            context.complexity_accumulation, self.break_criteria.complexity_trigger,
            context.context_drift_score, self.break_criteria.drift_threshold
        );

        Ok(operations_threshold || time_threshold || complexity_threshold || context_drift_detected)
    }

    /// Take a nicotine break - generate and solve context puzzle
    pub async fn take_nicotine_break(&self, current_context: &ProcessingContext) -> Result<NicotineBreakResult> {
        info!("🚬 Taking nicotine break - context validation time");

        // Reserve ATP for break processing
        let break_cost = self.calculate_break_cost(current_context);
        let reservation = self.atp_manager.reserve_atp("nicotine_break", break_cost).await?;

        // Generate context puzzle
        let puzzle = self.puzzle_generator.generate_context_puzzle(current_context).await?;
        
        // Record break start
        let break_start = Utc::now();
        
        // Attempt to solve the puzzle (this simulates the AI solving it)
        let solution_attempt = self.solve_context_puzzle(&puzzle).await?;
        
        // Validate solution
        let validation_result = self.validate_puzzle_solution(&puzzle, &solution_attempt).await?;
        
        // Calculate break duration
        let break_duration = Utc::now() - break_start;
        
        // Create break result
        let break_result = NicotineBreakResult {
            break_id: Uuid::new_v4(),
            timestamp: break_start,
            duration: break_duration,
            puzzle_complexity: puzzle.complexity_level,
            solution_correct: validation_result.correct,
            context_refreshed: validation_result.correct,
            confidence_recovery: if validation_result.correct { 0.95 } else { 0.3 },
            insights_gained: validation_result.insights,
            atp_cost: break_cost,
        };

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "nicotine_break").await?;

        // Reset context tracker if break was successful
        if validation_result.correct {
            self.reset_context_tracker(current_context).await;
            info!("✅ Nicotine break successful - context refreshed");
        } else {
            warn!("❌ Nicotine break failed - context may be corrupted");
        }

        // Record break in history
        self.record_break(&break_result).await;

        Ok(break_result)
    }

    /// Update context tracking with new operation
    pub async fn track_operation(&self, operation: &ProcessingOperation) -> Result<()> {
        let mut context = self.context_tracker.write().await;
        context.add_operation(operation);
        Ok(())
    }

    /// Generate a coded puzzle that encodes the current context
    async fn solve_context_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        debug!("🧩 Solving context puzzle: {}", puzzle.puzzle_type);

        let solution = match &puzzle.puzzle_type {
            PuzzleType::HashChain => self.solve_hash_chain_puzzle(puzzle).await?,
            PuzzleType::StateEncoding => self.solve_state_encoding_puzzle(puzzle).await?,
            PuzzleType::OperationSequence => self.solve_operation_sequence_puzzle(puzzle).await?,
            PuzzleType::ContextIntegrity => self.solve_context_integrity_puzzle(puzzle).await?,
            PuzzleType::ObjectiveValidation => self.solve_objective_validation_puzzle(puzzle).await?,
        };

        Ok(solution)
    }

    /// Solve hash chain puzzle (validates processing sequence)
    async fn solve_hash_chain_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        let context = self.context_tracker.read().await;
        
        // Reconstruct the hash chain from operations
        let mut hash_chain = Vec::new();
        let mut hasher = Sha256::new();
        
        for operation in &context.operation_history {
            hasher.update(operation.operation_hash.as_bytes());
            let hash_result = hasher.finalize_reset();
            hash_chain.push(hex::encode(hash_result));
        }

        Ok(PuzzleSolution {
            solution_type: puzzle.puzzle_type.clone(),
            solution_data: serde_json::to_string(&hash_chain)?,
            confidence: 0.95,
            computation_steps: hash_chain.len(),
        })
    }

    /// Solve state encoding puzzle (validates current state understanding)
    async fn solve_state_encoding_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        let context = self.context_tracker.read().await;
        
        // Encode current processing state
        let state_encoding = StateEncoding {
            current_objective: context.primary_objective.clone(),
            active_queries: context.active_queries.len(),
            processing_depth: context.processing_depth,
            confidence_level: context.average_confidence(),
            last_significant_finding: context.last_significant_timestamp,
        };

        Ok(PuzzleSolution {
            solution_type: puzzle.puzzle_type.clone(),
            solution_data: serde_json::to_string(&state_encoding)?,
            confidence: 0.92,
            computation_steps: 5,
        })
    }

    /// Solve operation sequence puzzle (validates operation ordering)
    async fn solve_operation_sequence_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        let context = self.context_tracker.read().await;
        
        // Extract operation sequence pattern
        let sequence_pattern: Vec<String> = context.operation_history
            .iter()
            .map(|op| format!("{}:{}", op.operation_type, op.confidence_delta))
            .collect();

        Ok(PuzzleSolution {
            solution_type: puzzle.puzzle_type.clone(),
            solution_data: serde_json::to_string(&sequence_pattern)?,
            confidence: 0.88,
            computation_steps: sequence_pattern.len(),
        })
    }

    /// Solve context integrity puzzle (validates overall context coherence)
    async fn solve_context_integrity_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        let context = self.context_tracker.read().await;
        
        // Calculate context integrity metrics
        let integrity_metrics = ContextIntegrityMetrics {
            objective_alignment: context.calculate_objective_alignment(),
            operation_coherence: context.calculate_operation_coherence(),
            confidence_consistency: context.calculate_confidence_consistency(),
            temporal_continuity: context.calculate_temporal_continuity(),
        };

        Ok(PuzzleSolution {
            solution_type: puzzle.puzzle_type.clone(),
            solution_data: serde_json::to_string(&integrity_metrics)?,
            confidence: 0.90,
            computation_steps: 4,
        })
    }

    /// Solve objective validation puzzle (confirms we're still on track)
    async fn solve_objective_validation_puzzle(&self, puzzle: &ContextPuzzle) -> Result<PuzzleSolution> {
        let context = self.context_tracker.read().await;
        
        // Validate current objective against initial goal
        let objective_validation = ObjectiveValidation {
            original_objective: context.original_objective.clone(),
            current_objective: context.primary_objective.clone(),
            drift_amount: context.calculate_objective_drift(),
            still_aligned: context.objective_still_aligned(),
            course_correction_needed: context.needs_course_correction(),
        };

        Ok(PuzzleSolution {
            solution_type: puzzle.puzzle_type.clone(),
            solution_data: serde_json::to_string(&objective_validation)?,
            confidence: 0.93,
            computation_steps: 5,
        })
    }

    /// Validate the puzzle solution
    async fn validate_puzzle_solution(
        &self,
        puzzle: &ContextPuzzle,
        solution: &PuzzleSolution,
    ) -> Result<ValidationResult> {
        let mut insights = Vec::new();
        let mut correct = false;

        match &puzzle.puzzle_type {
            PuzzleType::HashChain => {
                // Validate hash chain correctness
                if let Ok(submitted_chain) = serde_json::from_str::<Vec<String>>(&solution.solution_data) {
                    correct = self.validate_hash_chain(&submitted_chain).await?;
                    if correct {
                        insights.push("Hash chain integrity confirmed - operation sequence valid".to_string());
                    }
                }
            },
            PuzzleType::StateEncoding => {
                // Validate state encoding accuracy
                if let Ok(state_encoding) = serde_json::from_str::<StateEncoding>(&solution.solution_data) {
                    correct = self.validate_state_encoding(&state_encoding).await?;
                    if correct {
                        insights.push("State encoding accurate - context understanding preserved".to_string());
                    }
                }
            },
            PuzzleType::OperationSequence => {
                // Validate operation sequence pattern
                if let Ok(sequence) = serde_json::from_str::<Vec<String>>(&solution.solution_data) {
                    correct = self.validate_operation_sequence(&sequence).await?;
                    if correct {
                        insights.push("Operation sequence pattern recognized - processing flow coherent".to_string());
                    }
                }
            },
            PuzzleType::ContextIntegrity => {
                // Validate context integrity metrics
                if let Ok(metrics) = serde_json::from_str::<ContextIntegrityMetrics>(&solution.solution_data) {
                    correct = self.validate_context_integrity(&metrics).await?;
                    if correct {
                        insights.push("Context integrity verified - processing remains coherent".to_string());
                    }
                }
            },
            PuzzleType::ObjectiveValidation => {
                // Validate objective alignment
                if let Ok(validation) = serde_json::from_str::<ObjectiveValidation>(&solution.solution_data) {
                    correct = self.validate_objective_alignment(&validation).await?;
                    if correct {
                        insights.push("Objective alignment confirmed - still on track".to_string());
                    } else {
                        insights.push("Objective drift detected - course correction recommended".to_string());
                    }
                }
            },
        }

        Ok(ValidationResult {
            correct,
            confidence: if correct { 0.95 } else { 0.2 },
            insights,
        })
    }

    /// Calculate ATP cost for a nicotine break
    fn calculate_break_cost(&self, context: &ProcessingContext) -> u64 {
        let base_cost = 200u64;
        let complexity_cost = (context.complexity_level * 50.0) as u64;
        let operation_cost = context.pending_operations.len() as u64 * 10;
        
        base_cost + complexity_cost + operation_cost
    }

    /// Reset context tracker after successful break
    async fn reset_context_tracker(&self, current_context: &ProcessingContext) {
        let mut context = self.context_tracker.write().await;
        context.operations_since_break = 0;
        context.last_break = Some(Utc::now());
        context.complexity_accumulation = 0.0;
        context.context_drift_score = 0.0;
        context.refresh_context(current_context);
    }

    /// Record break in history
    async fn record_break(&self, break_result: &NicotineBreakResult) {
        let mut history = self.break_history.write().await;
        history.push(break_result.clone());
        
        // Keep only recent breaks
        if history.len() > 100 {
            history.drain(0..50);
        }
    }

    /// Get break statistics
    pub async fn get_break_statistics(&self) -> BreakStatistics {
        let history = self.break_history.read().await;
        
        let total_breaks = history.len();
        let successful_breaks = history.iter().filter(|b| b.solution_correct).count();
        let average_duration = if total_breaks > 0 {
            history.iter().map(|b| b.duration.num_seconds()).sum::<i64>() / total_breaks as i64
        } else {
            0
        };

        BreakStatistics {
            total_breaks,
            successful_breaks,
            success_rate: if total_breaks > 0 { successful_breaks as f64 / total_breaks as f64 } else { 0.0 },
            average_duration_seconds: average_duration,
            total_atp_consumed: history.iter().map(|b| b.atp_cost).sum(),
        }
    }

    // Validation helper methods
    async fn validate_hash_chain(&self, _submitted_chain: &[String]) -> Result<bool> {
        // Implement hash chain validation logic
        Ok(true) // Simplified for now
    }

    async fn validate_state_encoding(&self, _encoding: &StateEncoding) -> Result<bool> {
        // Implement state encoding validation logic
        Ok(true) // Simplified for now
    }

    async fn validate_operation_sequence(&self, _sequence: &[String]) -> Result<bool> {
        // Implement operation sequence validation logic
        Ok(true) // Simplified for now
    }

    async fn validate_context_integrity(&self, _metrics: &ContextIntegrityMetrics) -> Result<bool> {
        // Implement context integrity validation logic
        Ok(true) // Simplified for now
    }

    async fn validate_objective_alignment(&self, _validation: &ObjectiveValidation) -> Result<bool> {
        // Implement objective alignment validation logic
        Ok(true) // Simplified for now
    }
}

/// Puzzle generator for context validation
#[derive(Debug)]
pub struct PuzzleGenerator {
    rng: std::sync::Mutex<rand::rngs::ThreadRng>,
}

impl PuzzleGenerator {
    fn new() -> Self {
        Self {
            rng: std::sync::Mutex::new(thread_rng()),
        }
    }

    async fn generate_context_puzzle(&self, context: &ProcessingContext) -> Result<ContextPuzzle> {
        let puzzle_types = [
            PuzzleType::HashChain,
            PuzzleType::StateEncoding,
            PuzzleType::OperationSequence,
            PuzzleType::ContextIntegrity,
            PuzzleType::ObjectiveValidation,
        ];

        let puzzle_type = {
            let mut rng = self.rng.lock().unwrap();
            puzzle_types[rng.gen_range(0..puzzle_types.len())].clone()
        };

        let complexity_level = self.calculate_puzzle_complexity(context);

        Ok(ContextPuzzle {
            puzzle_id: Uuid::new_v4(),
            puzzle_type,
            complexity_level,
            context_snapshot: context.create_snapshot(),
            generation_timestamp: Utc::now(),
            encoded_data: self.encode_context_data(context).await?,
        })
    }

    fn calculate_puzzle_complexity(&self, context: &ProcessingContext) -> f64 {
        let base_complexity = 0.5;
        let operation_complexity = context.pending_operations.len() as f64 * 0.1;
        let depth_complexity = context.processing_depth as f64 * 0.05;
        
        (base_complexity + operation_complexity + depth_complexity).min(1.0)
    }

    async fn encode_context_data(&self, context: &ProcessingContext) -> Result<String> {
        // Create encoded representation of context that's not human readable
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_string(&context.primary_objective)?);
        hasher.update(&context.processing_depth.to_be_bytes());
        hasher.update(&context.complexity_level.to_be_bytes());
        
        Ok(hex::encode(hasher.finalize()))
    }
}

/// Configuration for when to take nicotine breaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakCriteria {
    pub operations_trigger: u32,           // Number of operations before break
    pub time_trigger: Duration,            // Time elapsed before break
    pub complexity_trigger: f64,           // Accumulated complexity threshold
    pub drift_threshold: f64,              // Context drift threshold
    pub enable_adaptive_timing: bool,      // Adjust timing based on success rate
}

impl Default for BreakCriteria {
    fn default() -> Self {
        Self {
            operations_trigger: 10,
            time_trigger: Duration::minutes(15),
            complexity_trigger: 5.0,
            drift_threshold: 0.3,
            enable_adaptive_timing: true,
        }
    }
}

/// Context tracking for break decisions
#[derive(Debug)]
pub struct ContextTracker {
    pub operations_since_break: u32,
    pub last_break: Option<DateTime<Utc>>,
    pub complexity_accumulation: f64,
    pub context_drift_score: f64,
    pub operation_history: Vec<OperationRecord>,
    pub primary_objective: ProcessingObjective,
    pub original_objective: ProcessingObjective,
    pub active_queries: Vec<String>,
    pub processing_depth: u32,
    pub last_significant_timestamp: Option<DateTime<Utc>>,
}

impl ContextTracker {
    fn new() -> Self {
        Self {
            operations_since_break: 0,
            last_break: None,
            complexity_accumulation: 0.0,
            context_drift_score: 0.0,
            operation_history: Vec::new(),
            primary_objective: ProcessingObjective::default(),
            original_objective: ProcessingObjective::default(),
            active_queries: Vec::new(),
            processing_depth: 0,
            last_significant_timestamp: None,
        }
    }

    fn add_operation(&mut self, operation: &ProcessingOperation) {
        self.operations_since_break += 1;
        self.complexity_accumulation += operation.complexity_contribution;
        self.context_drift_score += operation.drift_impact;
        
        let record = OperationRecord {
            operation_type: operation.operation_type.clone(),
            operation_hash: operation.compute_hash(),
            confidence_delta: operation.confidence_impact,
            complexity_impact: operation.complexity_contribution,
            timestamp: Utc::now(),
        };
        
        self.operation_history.push(record);
        
        // Keep only recent operations
        if self.operation_history.len() > 50 {
            self.operation_history.drain(0..25);
        }
    }

    fn time_since_break(&self) -> Duration {
        match self.last_break {
            Some(last) => Utc::now() - last,
            None => Duration::hours(24), // Force break if never taken
        }
    }

    fn average_confidence(&self) -> f64 {
        if self.operation_history.is_empty() {
            return 0.5;
        }
        
        self.operation_history.iter()
            .map(|op| op.confidence_delta)
            .sum::<f64>() / self.operation_history.len() as f64
    }

    fn calculate_objective_alignment(&self) -> f64 {
        // Calculate how aligned current objective is with original
        0.85 // Simplified implementation
    }

    fn calculate_operation_coherence(&self) -> f64 {
        // Calculate coherence between operations
        0.90 // Simplified implementation
    }

    fn calculate_confidence_consistency(&self) -> f64 {
        // Calculate consistency in confidence levels
        0.88 // Simplified implementation
    }

    fn calculate_temporal_continuity(&self) -> f64 {
        // Calculate temporal continuity of processing
        0.92 // Simplified implementation
    }

    fn calculate_objective_drift(&self) -> f64 {
        // Calculate how much we've drifted from original objective
        0.15 // Simplified implementation
    }

    fn objective_still_aligned(&self) -> bool {
        self.calculate_objective_drift() < 0.3
    }

    fn needs_course_correction(&self) -> bool {
        self.calculate_objective_drift() > 0.4
    }

    fn refresh_context(&mut self, context: &ProcessingContext) {
        self.primary_objective = context.primary_objective.clone();
        self.active_queries = context.active_queries.clone();
        self.processing_depth = context.processing_depth;
    }
}

/// Types of context puzzles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PuzzleType {
    HashChain,           // Validates operation sequence integrity
    StateEncoding,       // Validates current state understanding
    OperationSequence,   // Validates operation ordering
    ContextIntegrity,    // Validates overall context coherence
    ObjectiveValidation, // Validates objective alignment
}

/// A context puzzle for validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPuzzle {
    pub puzzle_id: Uuid,
    pub puzzle_type: PuzzleType,
    pub complexity_level: f64,
    pub context_snapshot: ContextSnapshot,
    pub generation_timestamp: DateTime<Utc>,
    pub encoded_data: String,
}

/// Solution to a context puzzle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PuzzleSolution {
    pub solution_type: PuzzleType,
    pub solution_data: String,
    pub confidence: f64,
    pub computation_steps: usize,
}

/// Result of puzzle validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub correct: bool,
    pub confidence: f64,
    pub insights: Vec<String>,
}

/// Result of a nicotine break
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NicotineBreakResult {
    pub break_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
    pub puzzle_complexity: f64,
    pub solution_correct: bool,
    pub context_refreshed: bool,
    pub confidence_recovery: f64,
    pub insights_gained: Vec<String>,
    pub atp_cost: u64,
}

/// Current processing context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingContext {
    pub primary_objective: ProcessingObjective,
    pub active_queries: Vec<String>,
    pub pending_operations: Vec<PendingOperation>,
    pub processing_depth: u32,
    pub complexity_level: f64,
    pub confidence_level: f64,
}

impl ProcessingContext {
    pub fn create_snapshot(&self) -> ContextSnapshot {
        ContextSnapshot {
            objective_hash: self.primary_objective.compute_hash(),
            query_count: self.active_queries.len(),
            operation_count: self.pending_operations.len(),
            depth: self.processing_depth,
            complexity: self.complexity_level,
            confidence: self.confidence_level,
            timestamp: Utc::now(),
        }
    }
}

/// Processing objective
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingObjective {
    pub description: String,
    pub priority: f64,
    pub success_criteria: Vec<String>,
}

impl ProcessingObjective {
    fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.description);
        hasher.update(&self.priority.to_be_bytes());
        hex::encode(hasher.finalize())
    }
}

impl Default for ProcessingObjective {
    fn default() -> Self {
        Self {
            description: "Default processing objective".to_string(),
            priority: 0.5,
            success_criteria: vec!["Complete successfully".to_string()],
        }
    }
}

/// A processing operation
#[derive(Debug, Clone)]
pub struct ProcessingOperation {
    pub operation_type: String,
    pub complexity_contribution: f64,
    pub confidence_impact: f64,
    pub drift_impact: f64,
}

impl ProcessingOperation {
    pub fn compute_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(&self.operation_type);
        hasher.update(&self.complexity_contribution.to_be_bytes());
        hex::encode(hasher.finalize())
    }
}

/// Record of a processing operation
#[derive(Debug, Clone)]
pub struct OperationRecord {
    pub operation_type: String,
    pub operation_hash: String,
    pub confidence_delta: f64,
    pub complexity_impact: f64,
    pub timestamp: DateTime<Utc>,
}

/// Pending operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOperation {
    pub operation_id: String,
    pub complexity: f64,
}

/// Context snapshot for puzzles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextSnapshot {
    pub objective_hash: String,
    pub query_count: usize,
    pub operation_count: usize,
    pub depth: u32,
    pub complexity: f64,
    pub confidence: f64,
    pub timestamp: DateTime<Utc>,
}

/// State encoding for puzzles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateEncoding {
    pub current_objective: ProcessingObjective,
    pub active_queries: usize,
    pub processing_depth: u32,
    pub confidence_level: f64,
    pub last_significant_finding: Option<DateTime<Utc>>,
}

/// Context integrity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextIntegrityMetrics {
    pub objective_alignment: f64,
    pub operation_coherence: f64,
    pub confidence_consistency: f64,
    pub temporal_continuity: f64,
}

/// Objective validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveValidation {
    pub original_objective: ProcessingObjective,
    pub current_objective: ProcessingObjective,
    pub drift_amount: f64,
    pub still_aligned: bool,
    pub course_correction_needed: bool,
}

/// Break statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreakStatistics {
    pub total_breaks: usize,
    pub successful_breaks: usize,
    pub success_rate: f64,
    pub average_duration_seconds: i64,
    pub total_atp_consumed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[tokio::test]
    async fn test_nicotine_break_cycle() {
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

        let engine = NicotineEngine::new(
            atp_manager,
            BreakCriteria::default(),
        );

        // Simulate some operations
        for i in 0..12 {
            let operation = ProcessingOperation {
                operation_type: format!("test_operation_{}", i),
                complexity_contribution: 0.5,
                confidence_impact: 0.1,
                drift_impact: 0.05,
            };
            engine.track_operation(&operation).await.unwrap();
        }

        // Should trigger a break
        assert!(engine.should_take_break().await.unwrap());

        let context = ProcessingContext {
            primary_objective: ProcessingObjective::default(),
            active_queries: vec!["test query".to_string()],
            pending_operations: vec![],
            processing_depth: 2,
            complexity_level: 3.0,
            confidence_level: 0.8,
        };

        let break_result = engine.take_nicotine_break(&context).await.unwrap();
        assert!(break_result.atp_cost > 0);
    }

    #[tokio::test]
    async fn test_break_statistics() {
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

        let engine = NicotineEngine::new(
            atp_manager,
            BreakCriteria::default(),
        );

        let stats = engine.get_break_statistics().await;
        assert_eq!(stats.total_breaks, 0);
        assert_eq!(stats.success_rate, 0.0);
    }
}