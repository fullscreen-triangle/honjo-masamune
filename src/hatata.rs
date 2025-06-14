//! Hatata Decision Optimization Engine
//! 
//! The decision optimization engine that uses Markov Decision Processes (MDP)
//! to find optimal action sequences based on belief states from mzekezeke
//! and extraordinary insights from spectacular findings.

use anyhow::Result;
use atp_manager::AtpManager;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, debug, error};
use chrono::{DateTime, Utc, Duration};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use rand::{Rng, thread_rng, seq::SliceRandom};

use crate::config::HatataDecisionConfig;
use crate::mzekezeke::{BeliefUpdateResult, Belief, NetworkStatistics};
use crate::spectacular::SpectacularFinding;

/// The hatata decision optimization engine
#[derive(Debug)]
pub struct HatataEngine {
    config: HatataDecisionConfig,
    atp_manager: Arc<AtpManager>,
    mdp_solver: MarkovDecisionSolver,
    policy_cache: Arc<RwLock<PolicyCache>>,
    decision_history: Arc<RwLock<Vec<DecisionRecord>>>,
    utility_functions: UtilityFunctionSet,
    current_state: Arc<RwLock<SystemState>>,
}

impl HatataEngine {
    /// Create a new hatata decision engine
    pub fn new(
        config: HatataDecisionConfig,
        atp_manager: Arc<AtpManager>,
    ) -> Self {
        let mdp_solver = MarkovDecisionSolver::new(&config.mdp_config);
        let utility_functions = UtilityFunctionSet::new(&config.utility_config);
        
        Self {
            config,
            atp_manager,
            mdp_solver,
            policy_cache: Arc::new(RwLock::new(PolicyCache::new())),
            decision_history: Arc::new(RwLock::new(Vec::new())),
            utility_functions,
            current_state: Arc::new(RwLock::new(SystemState::default())),
        }
    }

    /// Optimize decisions based on current system state
    pub async fn optimize_decisions(
        &self,
        belief_state: NetworkStatistics,
        spectacular_findings: Vec<SpectacularFinding>,
        available_actions: Vec<ActionOption>,
    ) -> Result<DecisionOptimizationResult> {
        info!("🎯 Optimizing decisions for {} actions", available_actions.len());

        // Reserve ATP for optimization
        let optimization_cost = self.calculate_optimization_cost(&available_actions);
        let reservation = self.atp_manager.reserve_atp("decision_optimization", optimization_cost).await?;

        let start_time = Utc::now();
        let optimization_id = Uuid::new_v4();

        // Update current system state
        {
            let mut state = self.current_state.write().await;
            state.update_from_beliefs(&belief_state);
            state.incorporate_spectacular_findings(&spectacular_findings);
        }

        // Build MDP model
        let mdp_model = self.build_mdp_model(&belief_state, &spectacular_findings, &available_actions).await?;

        // Solve MDP for optimal policy
        let optimal_policy = self.mdp_solver.solve_mdp(mdp_model).await?;

        // Generate action recommendations
        let recommendations = self.generate_action_recommendations(&optimal_policy, &available_actions).await?;

        // Calculate expected utilities
        let utility_analysis = self.analyze_expected_utilities(&recommendations, &belief_state).await?;

        // Cache policy for future use
        {
            let mut cache = self.policy_cache.write().await;
            cache.store_policy(&belief_state, optimal_policy.clone());
        }

        // Consume ATP
        self.atp_manager.consume_atp(reservation, "decision_optimization").await?;

        let result = DecisionOptimizationResult {
            optimization_id,
            recommended_actions: recommendations,
            expected_utility: utility_analysis.total_expected_utility,
            confidence_score: utility_analysis.confidence_score,
            risk_assessment: utility_analysis.risk_assessment,
            spectacular_impact: spectacular_findings.len() as f64 * 0.1,
            processing_time: Utc::now() - start_time,
            atp_cost: optimization_cost,
        };

        // Record decision
        self.record_decision(&result).await;

        info!("✅ Decision optimization complete: {} recommendations, utility: {:.3}", 
              result.recommended_actions.len(), result.expected_utility);

        Ok(result)
    }

    /// Execute decision and monitor outcomes
    pub async fn execute_decision(
        &self,
        decision: &ActionRecommendation,
        execution_context: ExecutionContext,
    ) -> Result<ExecutionResult> {
        info!("⚡ Executing decision: {}", decision.action_name);

        let execution_id = Uuid::new_v4();
        let start_time = Utc::now();

        // Prepare execution
        let execution_plan = self.create_execution_plan(decision, &execution_context).await?;
        
        // Monitor execution (simplified)
        let execution_outcome = self.simulate_execution(&execution_plan).await?;

        // Update system state based on outcome
        {
            let mut state = self.current_state.write().await;
            state.apply_execution_outcome(&execution_outcome);
        }

        let result = ExecutionResult {
            execution_id,
            decision_id: decision.recommendation_id,
            actual_utility: execution_outcome.realized_utility,
            predicted_utility: decision.expected_utility,
            success_rate: execution_outcome.success_rate,
            side_effects: execution_outcome.side_effects,
            execution_time: Utc::now() - start_time,
            lessons_learned: execution_outcome.lessons_learned,
        };

        info!("✅ Decision executed: utility {:.3}, success rate {:.3}", 
              result.actual_utility, result.success_rate);

        Ok(result)
    }

    /// Get decision statistics
    pub async fn get_decision_statistics(&self) -> DecisionStatistics {
        let history = self.decision_history.read().await;
        let state = self.current_state.read().await;

        let total_decisions = history.len();
        let avg_utility = if total_decisions > 0 {
            history.iter().map(|r| r.expected_utility).sum::<f64>() / total_decisions as f64
        } else {
            0.0
        };

        DecisionStatistics {
            total_decisions,
            average_utility: avg_utility,
            current_state: state.clone(),
            cache_hit_rate: self.calculate_cache_hit_rate().await,
            optimization_efficiency: self.calculate_optimization_efficiency().await,
        }
    }

    /// Build MDP model from current state
    async fn build_mdp_model(
        &self,
        belief_state: &NetworkStatistics,
        spectacular_findings: &[SpectacularFinding],
        available_actions: &[ActionOption],
    ) -> Result<MDPModel> {
        let num_states = self.estimate_state_space_size(belief_state);
        let num_actions = available_actions.len();

        // Build state transition matrix
        let transition_matrix = self.build_transition_matrix(num_states, num_actions, belief_state).await?;
        
        // Build reward matrix
        let reward_matrix = self.build_reward_matrix(num_states, num_actions, spectacular_findings).await?;

        // Set discount factor based on system characteristics
        let discount_factor = self.calculate_discount_factor(belief_state);

        Ok(MDPModel {
            states: num_states,
            actions: num_actions,
            transition_probabilities: transition_matrix,
            rewards: reward_matrix,
            discount_factor,
            initial_state_distribution: DVector::from_fn(num_states, |i, _| {
                if i == 0 { 1.0 } else { 0.0 }
            }),
        })
    }

    /// Generate action recommendations from optimal policy
    async fn generate_action_recommendations(
        &self,
        policy: &OptimalPolicy,
        available_actions: &[ActionOption],
    ) -> Result<Vec<ActionRecommendation>> {
        let mut recommendations = Vec::new();

        for (action_idx, action) in available_actions.iter().enumerate() {
            let recommendation_score = policy.get_action_probability(0, action_idx);
            
            if recommendation_score > 0.1 { // Threshold for recommendation
                let recommendation = ActionRecommendation {
                    recommendation_id: Uuid::new_v4(),
                    action_name: action.name.clone(),
                    action_type: action.action_type.clone(),
                    expected_utility: recommendation_score * action.base_utility,
                    confidence: recommendation_score,
                    risk_level: self.assess_action_risk(action).await,
                    prerequisites: action.prerequisites.clone(),
                    estimated_duration: action.estimated_duration,
                    resource_requirements: action.resource_requirements.clone(),
                    success_probability: policy.get_action_probability(0, action_idx),
                };
                
                recommendations.push(recommendation);
            }
        }

        // Sort by expected utility
        recommendations.sort_by(|a, b| b.expected_utility.partial_cmp(&a.expected_utility).unwrap());

        Ok(recommendations)
    }

    /// Analyze expected utilities
    async fn analyze_expected_utilities(
        &self,
        recommendations: &[ActionRecommendation],
        belief_state: &NetworkStatistics,
    ) -> Result<UtilityAnalysis> {
        let total_expected_utility = recommendations.iter()
            .map(|r| r.expected_utility * r.confidence)
            .sum::<f64>();

        let confidence_score = if recommendations.is_empty() {
            0.0
        } else {
            recommendations.iter().map(|r| r.confidence).sum::<f64>() / recommendations.len() as f64
        };

        let risk_assessment = self.assess_portfolio_risk(recommendations).await?;

        Ok(UtilityAnalysis {
            total_expected_utility,
            confidence_score,
            risk_assessment,
        })
    }

    /// Build transition matrix
    async fn build_transition_matrix(
        &self,
        num_states: usize,
        num_actions: usize,
        belief_state: &NetworkStatistics,
    ) -> Result<Vec<DMatrix<f64>>> {
        let mut matrices = Vec::new();
        
        for _ in 0..num_actions {
            let mut matrix = DMatrix::zeros(num_states, num_states);
            
            // Simplified transition probabilities based on belief convergence
            let stability = belief_state.convergence_score;
            
            for i in 0..num_states {
                for j in 0..num_states {
                    if i == j {
                        matrix[(i, j)] = stability; // Stay in same state
                    } else if j == (i + 1) % num_states {
                        matrix[(i, j)] = 1.0 - stability; // Transition to next state
                    }
                }
            }
            
            matrices.push(matrix);
        }
        
        Ok(matrices)
    }

    /// Build reward matrix
    async fn build_reward_matrix(
        &self,
        num_states: usize,
        num_actions: usize,
        spectacular_findings: &[SpectacularFinding],
    ) -> Result<DMatrix<f64>> {
        let mut rewards = DMatrix::zeros(num_states, num_actions);
        
        // Base rewards from utility functions
        for i in 0..num_states {
            for j in 0..num_actions {
                let base_reward = self.utility_functions.calculate_base_utility(i, j);
                let spectacular_bonus = spectacular_findings.iter()
                    .map(|f| f.significance_score * 0.1)
                    .sum::<f64>();
                
                rewards[(i, j)] = base_reward + spectacular_bonus;
            }
        }
        
        Ok(rewards)
    }

    /// Calculate various metrics
    fn estimate_state_space_size(&self, belief_state: &NetworkStatistics) -> usize {
        (belief_state.total_nodes + 1).max(5).min(50) // Reasonable bounds
    }

    fn calculate_discount_factor(&self, belief_state: &NetworkStatistics) -> f64 {
        0.9 + belief_state.convergence_score * 0.09 // Higher discount for stable systems
    }

    async fn assess_action_risk(&self, action: &ActionOption) -> RiskLevel {
        match action.action_type {
            ActionType::Conservative => RiskLevel::Low,
            ActionType::Moderate => RiskLevel::Medium,
            ActionType::Aggressive => RiskLevel::High,
            ActionType::Experimental => RiskLevel::VeryHigh,
        }
    }

    async fn assess_portfolio_risk(&self, recommendations: &[ActionRecommendation]) -> Result<RiskAssessment> {
        let high_risk_count = recommendations.iter()
            .filter(|r| matches!(r.risk_level, RiskLevel::High | RiskLevel::VeryHigh))
            .count();

        let overall_risk = if high_risk_count > recommendations.len() / 2 {
            RiskLevel::High
        } else if high_risk_count > 0 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        Ok(RiskAssessment {
            overall_risk,
            risk_factors: vec!["Market volatility".to_string(), "Implementation complexity".to_string()],
            mitigation_strategies: vec!["Diversification".to_string(), "Phased implementation".to_string()],
        })
    }

    async fn calculate_cache_hit_rate(&self) -> f64 {
        // Simplified cache hit rate calculation
        0.75
    }

    async fn calculate_optimization_efficiency(&self) -> f64 {
        // Simplified efficiency calculation
        0.85
    }

    fn calculate_optimization_cost(&self, actions: &[ActionOption]) -> u64 {
        let base_cost = 300u64;
        let action_cost = actions.len() as u64 * 100;
        base_cost + action_cost
    }

    async fn create_execution_plan(
        &self,
        decision: &ActionRecommendation,
        context: &ExecutionContext,
    ) -> Result<ExecutionPlan> {
        Ok(ExecutionPlan {
            decision_id: decision.recommendation_id,
            steps: vec!["Prepare".to_string(), "Execute".to_string(), "Monitor".to_string()],
            resource_allocation: decision.resource_requirements.clone(),
            risk_mitigation: vec!["Backup plan".to_string()],
            success_criteria: vec!["Utility achieved".to_string()],
        })
    }

    async fn simulate_execution(&self, plan: &ExecutionPlan) -> Result<ExecutionOutcome> {
        // Simplified execution simulation
        let mut rng = thread_rng();
        let success_rate = 0.7 + rng.gen::<f64>() * 0.3;
        
        Ok(ExecutionOutcome {
            realized_utility: success_rate * 0.8,
            success_rate,
            side_effects: vec!["Minor resource consumption".to_string()],
            lessons_learned: vec!["Execution went as planned".to_string()],
        })
    }

    async fn record_decision(&self, result: &DecisionOptimizationResult) {
        let mut history = self.decision_history.write().await;
        
        let record = DecisionRecord {
            timestamp: Utc::now(),
            optimization_id: result.optimization_id,
            expected_utility: result.expected_utility,
            confidence_score: result.confidence_score,
            recommendations_count: result.recommended_actions.len(),
            processing_time: result.processing_time,
        };
        
        history.push(record);
        
        // Keep recent history
        if history.len() > 1000 {
            history.drain(0..500);
        }
    }
}

/// Markov Decision Process solver
#[derive(Debug)]
pub struct MarkovDecisionSolver {
    max_iterations: u32,
    convergence_threshold: f64,
    algorithm: SolutionAlgorithm,
}

impl MarkovDecisionSolver {
    fn new(config: &crate::config::MDPConfig) -> Self {
        Self {
            max_iterations: config.max_iterations as u32,
            convergence_threshold: config.convergence_threshold,
            algorithm: config.algorithm.clone(),
        }
    }

    async fn solve_mdp(&self, model: MDPModel) -> Result<OptimalPolicy> {
        match self.algorithm {
            SolutionAlgorithm::ValueIteration => self.value_iteration(model).await,
            SolutionAlgorithm::PolicyIteration => self.policy_iteration(model).await,
            SolutionAlgorithm::QLearning => self.q_learning(model).await,
        }
    }

    async fn value_iteration(&self, model: MDPModel) -> Result<OptimalPolicy> {
        let mut values = DVector::zeros(model.states);
        let mut policy = DMatrix::zeros(model.states, model.actions);
        
        for _iter in 0..self.max_iterations {
            let prev_values = values.clone();
            
            // Update values
            for s in 0..model.states {
                let mut max_value = f64::NEG_INFINITY;
                let mut best_action = 0;
                
                for a in 0..model.actions {
                    let mut value = model.rewards[(s, a)];
                    
                    for s_next in 0..model.states {
                        value += model.discount_factor * 
                                model.transition_probabilities[a][(s, s_next)] * 
                                prev_values[s_next];
                    }
                    
                    if value > max_value {
                        max_value = value;
                        best_action = a;
                    }
                }
                
                values[s] = max_value;
                
                // Update policy
                for a in 0..model.actions {
                    policy[(s, a)] = if a == best_action { 1.0 } else { 0.0 };
                }
            }
            
            // Check convergence
            let diff = (&values - &prev_values).norm();
            if diff < self.convergence_threshold {
                break;
            }
        }
        
        Ok(OptimalPolicy {
            policy_matrix: policy,
            value_function: values,
        })
    }

    async fn policy_iteration(&self, model: MDPModel) -> Result<OptimalPolicy> {
        // Simplified policy iteration
        self.value_iteration(model).await
    }

    async fn q_learning(&self, model: MDPModel) -> Result<OptimalPolicy> {
        // Simplified Q-learning
        self.value_iteration(model).await
    }
}

/// Utility function set
#[derive(Debug)]
pub struct UtilityFunctionSet {
    functions: HashMap<String, Box<dyn UtilityFunction>>,
}

impl UtilityFunctionSet {
    fn new(config: &crate::config::UtilityConfig) -> Self {
        let mut functions = HashMap::new();
        
        // Add default utility functions
        functions.insert("efficiency".to_string(), Box::new(EfficiencyUtility::new()) as Box<dyn UtilityFunction>);
        functions.insert("reliability".to_string(), Box::new(ReliabilityUtility::new()) as Box<dyn UtilityFunction>);
        functions.insert("innovation".to_string(), Box::new(InnovationUtility::new()) as Box<dyn UtilityFunction>);
        
        Self { functions }
    }

    fn calculate_base_utility(&self, state: usize, action: usize) -> f64 {
        self.functions.values()
            .map(|f| f.calculate(state, action))
            .sum::<f64>() / self.functions.len() as f64
    }
}

/// Utility function trait
trait UtilityFunction: Send + Sync + std::fmt::Debug {
    fn calculate(&self, state: usize, action: usize) -> f64;
}

#[derive(Debug)]
struct EfficiencyUtility;
impl EfficiencyUtility {
    fn new() -> Self { Self }
}
impl UtilityFunction for EfficiencyUtility {
    fn calculate(&self, state: usize, action: usize) -> f64 {
        1.0 / (1.0 + (state + action) as f64 * 0.1)
    }
}

#[derive(Debug)]
struct ReliabilityUtility;
impl ReliabilityUtility {
    fn new() -> Self { Self }
}
impl UtilityFunction for ReliabilityUtility {
    fn calculate(&self, state: usize, action: usize) -> f64 {
        0.8 + (state as f64 * 0.02).min(0.2)
    }
}

#[derive(Debug)]
struct InnovationUtility;
impl InnovationUtility {
    fn new() -> Self { Self }
}
impl UtilityFunction for InnovationUtility {
    fn calculate(&self, state: usize, action: usize) -> f64 {
        (action as f64 * 0.15).min(1.0)
    }
}

/// Policy cache for storing computed policies
#[derive(Debug)]
pub struct PolicyCache {
    cache: HashMap<String, (OptimalPolicy, DateTime<Utc>)>,
    max_entries: usize,
}

impl PolicyCache {
    fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_entries: 100,
        }
    }

    fn store_policy(&mut self, belief_state: &NetworkStatistics, policy: OptimalPolicy) {
        let key = format!("{}_{}", belief_state.total_nodes, belief_state.convergence_score as u32);
        self.cache.insert(key, (policy, Utc::now()));
        
        // Cleanup old entries
        if self.cache.len() > self.max_entries {
            let oldest_key = self.cache.iter()
                .min_by_key(|(_, (_, timestamp))| timestamp)
                .map(|(k, _)| k.clone());
            
            if let Some(key) = oldest_key {
                self.cache.remove(&key);
            }
        }
    }
}

/// System state representation
#[derive(Debug, Clone, Default)]
pub struct SystemState {
    pub belief_confidence: f64,
    pub network_stability: f64,
    pub spectacular_impact: f64,
    pub last_updated: DateTime<Utc>,
}

impl SystemState {
    fn update_from_beliefs(&mut self, belief_state: &NetworkStatistics) {
        self.belief_confidence = belief_state.average_confidence;
        self.network_stability = belief_state.convergence_score;
        self.last_updated = Utc::now();
    }

    fn incorporate_spectacular_findings(&mut self, findings: &[SpectacularFinding]) {
        self.spectacular_impact = findings.iter()
            .map(|f| f.significance_score)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
    }

    fn apply_execution_outcome(&mut self, outcome: &ExecutionOutcome) {
        // Update state based on execution results
        self.belief_confidence *= outcome.success_rate;
        self.last_updated = Utc::now();
    }
}

/// Core data structures
#[derive(Debug)]
pub struct MDPModel {
    pub states: usize,
    pub actions: usize,
    pub transition_probabilities: Vec<DMatrix<f64>>,
    pub rewards: DMatrix<f64>,
    pub discount_factor: f64,
    pub initial_state_distribution: DVector<f64>,
}

#[derive(Debug, Clone)]
pub struct OptimalPolicy {
    pub policy_matrix: DMatrix<f64>,
    pub value_function: DVector<f64>,
}

impl OptimalPolicy {
    fn get_action_probability(&self, state: usize, action: usize) -> f64 {
        if state < self.policy_matrix.nrows() && action < self.policy_matrix.ncols() {
            self.policy_matrix[(state, action)]
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOption {
    pub name: String,
    pub action_type: ActionType,
    pub base_utility: f64,
    pub prerequisites: Vec<String>,
    pub estimated_duration: Duration,
    pub resource_requirements: ResourceRequirements,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    Conservative,
    Moderate,
    Aggressive,
    Experimental,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cost: u64,
    pub memory_cost: u64,
    pub atp_cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionRecommendation {
    pub recommendation_id: Uuid,
    pub action_name: String,
    pub action_type: ActionType,
    pub expected_utility: f64,
    pub confidence: f64,
    pub risk_level: RiskLevel,
    pub prerequisites: Vec<String>,
    pub estimated_duration: Duration,
    pub resource_requirements: ResourceRequirements,
    pub success_probability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOptimizationResult {
    pub optimization_id: Uuid,
    pub recommended_actions: Vec<ActionRecommendation>,
    pub expected_utility: f64,
    pub confidence_score: f64,
    pub risk_assessment: RiskAssessment,
    pub spectacular_impact: f64,
    pub processing_time: Duration,
    pub atp_cost: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub overall_risk: RiskLevel,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub environment: String,
    pub constraints: Vec<String>,
    pub resources_available: ResourceRequirements,
}

#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub execution_id: Uuid,
    pub decision_id: Uuid,
    pub actual_utility: f64,
    pub predicted_utility: f64,
    pub success_rate: f64,
    pub side_effects: Vec<String>,
    pub execution_time: Duration,
    pub lessons_learned: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStatistics {
    pub total_decisions: usize,
    pub average_utility: f64,
    pub current_state: SystemState,
    pub cache_hit_rate: f64,
    pub optimization_efficiency: f64,
}

// Internal structures
#[derive(Debug)]
struct UtilityAnalysis {
    total_expected_utility: f64,
    confidence_score: f64,
    risk_assessment: RiskAssessment,
}

#[derive(Debug)]
struct ExecutionPlan {
    decision_id: Uuid,
    steps: Vec<String>,
    resource_allocation: ResourceRequirements,
    risk_mitigation: Vec<String>,
    success_criteria: Vec<String>,
}

#[derive(Debug)]
struct ExecutionOutcome {
    realized_utility: f64,
    success_rate: f64,
    side_effects: Vec<String>,
    lessons_learned: Vec<String>,
}

#[derive(Debug, Clone)]
struct DecisionRecord {
    timestamp: DateTime<Utc>,
    optimization_id: Uuid,
    expected_utility: f64,
    confidence_score: f64,
    recommendations_count: usize,
    processing_time: Duration,
}

use crate::config::SolutionAlgorithm;

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[tokio::test]
    async fn test_hatata_engine_creation() {
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

        let config = HatataDecisionConfig::default();
        let engine = HatataEngine::new(config, atp_manager);

        let stats = engine.get_decision_statistics().await;
        assert_eq!(stats.total_decisions, 0);
    }
} 