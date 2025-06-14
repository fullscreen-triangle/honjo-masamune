use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ndarray::Array2;
use ordered_float::OrderedFloat;
use petgraph::graph::{DiGraph, NodeIndex};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// The mzekezeke Bayesian belief network - the core machine learning workhorse
/// This is the tangible objective function that the system optimizes
#[derive(Debug, Clone)]
pub struct MzekezekeBayesianCore {
    /// Python runtime for machine learning operations
    python_runtime: Arc<RwLock<PyObject>>,
    
    /// Belief network graph structure
    belief_network: Arc<RwLock<DiGraph<BeliefNode, BeliefEdge>>>,
    
    /// Node lookup by evidence ID
    node_lookup: Arc<DashMap<Uuid, NodeIndex>>,
    
    /// Temporal decay parameters
    decay_config: TemporalDecayConfig,
    
    /// Network optimization state
    optimization_state: Arc<RwLock<OptimizationState>>,
}

/// Individual belief node with temporal decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefNode {
    /// Unique identifier
    pub id: Uuid,
    
    /// Evidence content and metadata
    pub evidence: Evidence,
    
    /// Current belief probability [0.0, 1.0]
    pub belief_probability: OrderedFloat<f64>,
    
    /// Confidence in this belief [0.0, 1.0]
    pub confidence: OrderedFloat<f64>,
    
    /// Temporal decay state
    pub decay_state: TemporalDecayState,
    
    /// Prior probability before evidence
    pub prior_probability: OrderedFloat<f64>,
    
    /// Likelihood given evidence
    pub likelihood: OrderedFloat<f64>,
    
    /// Posterior probability after Bayesian update
    pub posterior_probability: OrderedFloat<f64>,
}

/// Evidence with multi-dimensional truth assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Evidence identifier
    pub id: Uuid,
    
    /// Raw evidence content
    pub content: String,
    
    /// Source of evidence
    pub source: EvidenceSource,
    
    /// Truth dimensions (not just true/false)
    pub truth_dimensions: TruthDimensions,
    
    /// Timestamp when evidence was created
    pub created_at: DateTime<Utc>,
    
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Evidence type classification
    pub evidence_type: EvidenceType,
}

/// Multi-dimensional truth assessment beyond binary true/false
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthDimensions {
    /// Factual accuracy [0.0, 1.0]
    pub factual_accuracy: OrderedFloat<f64>,
    
    /// Contextual relevance [0.0, 1.0]
    pub contextual_relevance: OrderedFloat<f64>,
    
    /// Temporal validity [0.0, 1.0]
    pub temporal_validity: OrderedFloat<f64>,
    
    /// Source credibility [0.0, 1.0]
    pub source_credibility: OrderedFloat<f64>,
    
    /// Logical consistency [0.0, 1.0]
    pub logical_consistency: OrderedFloat<f64>,
    
    /// Empirical support [0.0, 1.0]
    pub empirical_support: OrderedFloat<f64>,
}

/// Temporal decay state - information and evidence decay over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDecayState {
    /// Initial strength when evidence was fresh
    pub initial_strength: OrderedFloat<f64>,
    
    /// Current strength after decay
    pub current_strength: OrderedFloat<f64>,
    
    /// Decay rate per time unit
    pub decay_rate: OrderedFloat<f64>,
    
    /// Half-life of the information
    pub half_life: chrono::Duration,
    
    /// Time since last refresh
    pub time_since_refresh: chrono::Duration,
    
    /// Decay function type
    pub decay_function: DecayFunction,
}

/// Types of decay functions for different information types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Exponential decay: strength * e^(-decay_rate * time)
    Exponential,
    
    /// Linear decay: strength - (decay_rate * time)
    Linear,
    
    /// Power law decay: strength * time^(-decay_rate)
    PowerLaw,
    
    /// Logarithmic decay: strength * (1 - log(1 + decay_rate * time))
    Logarithmic,
    
    /// Step function decay: strength until threshold, then drop
    StepFunction { threshold: chrono::Duration },
}

/// Edge representing causal or correlational relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefEdge {
    /// Relationship strength [-1.0, 1.0]
    pub strength: OrderedFloat<f64>,
    
    /// Relationship type
    pub relationship_type: RelationshipType,
    
    /// Confidence in this relationship
    pub confidence: OrderedFloat<f64>,
    
    /// Conditional probability P(target|source)
    pub conditional_probability: OrderedFloat<f64>,
}

/// Types of relationships between beliefs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Causal relationship (A causes B)
    Causal,
    
    /// Correlational relationship (A correlates with B)
    Correlational,
    
    /// Contradictory relationship (A contradicts B)
    Contradictory,
    
    /// Supporting relationship (A supports B)
    Supporting,
    
    /// Temporal sequence (A precedes B)
    Temporal,
}

/// Evidence source classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceSource {
    /// Human expert testimony
    HumanExpert { expertise_level: f64 },
    
    /// Scientific publication
    ScientificPublication { impact_factor: f64, peer_reviewed: bool },
    
    /// Sensor data
    SensorData { accuracy: f64, calibration_date: DateTime<Utc> },
    
    /// Repository analysis result
    RepositoryAnalysis { repository_name: String, confidence: f64 },
    
    /// Historical record
    HistoricalRecord { age: chrono::Duration, authenticity: f64 },
    
    /// Synthetic/generated data
    Synthetic { generation_method: String, validation_score: f64 },
}

/// Evidence type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EvidenceType {
    /// Direct observation
    DirectObservation,
    
    /// Indirect inference
    IndirectInference,
    
    /// Statistical correlation
    StatisticalCorrelation,
    
    /// Expert opinion
    ExpertOpinion,
    
    /// Experimental result
    ExperimentalResult,
    
    /// Theoretical prediction
    TheoreticalPrediction,
}

/// Temporal decay configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDecayConfig {
    /// Default decay rate for different evidence types
    pub default_decay_rates: HashMap<EvidenceType, f64>,
    
    /// Minimum strength threshold before evidence is discarded
    pub minimum_strength_threshold: OrderedFloat<f64>,
    
    /// Refresh interval for updating decay states
    pub refresh_interval: chrono::Duration,
    
    /// Enable adaptive decay rates based on evidence performance
    pub adaptive_decay: bool,
}

/// Network optimization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationState {
    /// Current iteration number
    pub iteration: u64,
    
    /// Network likelihood score
    pub likelihood_score: OrderedFloat<f64>,
    
    /// Convergence threshold
    pub convergence_threshold: OrderedFloat<f64>,
    
    /// Has the network converged?
    pub converged: bool,
    
    /// Optimization algorithm state
    pub algorithm_state: OptimizationAlgorithm,
}

/// Optimization algorithms for belief network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Variational Bayes
    VariationalBayes {
        learning_rate: f64,
        momentum: f64,
    },
    
    /// Markov Chain Monte Carlo
    MCMC {
        chain_length: u64,
        burn_in: u64,
        thinning: u64,
    },
    
    /// Expectation Maximization
    ExpectationMaximization {
        max_iterations: u64,
        tolerance: f64,
    },
    
    /// Belief Propagation
    BeliefPropagation {
        max_iterations: u64,
        damping_factor: f64,
    },
}

impl MzekezekeBayesianCore {
    /// Initialize the mzekezeke Bayesian core with Python runtime
    pub async fn new(decay_config: TemporalDecayConfig) -> Result<Self> {
        info!("Initializing mzekezeke Bayesian belief network core");
        
        // Initialize Python runtime
        let python_runtime = Arc::new(RwLock::new(
            Self::initialize_python_runtime().await?
        ));
        
        // Initialize empty belief network
        let belief_network = Arc::new(RwLock::new(DiGraph::new()));
        let node_lookup = Arc::new(DashMap::new());
        
        // Initialize optimization state
        let optimization_state = Arc::new(RwLock::new(OptimizationState {
            iteration: 0,
            likelihood_score: OrderedFloat(0.0),
            convergence_threshold: OrderedFloat(1e-6),
            converged: false,
            algorithm_state: OptimizationAlgorithm::VariationalBayes {
                learning_rate: 0.01,
                momentum: 0.9,
            },
        }));
        
        Ok(Self {
            python_runtime,
            belief_network,
            node_lookup,
            decay_config,
            optimization_state,
        })
    }
    
    /// Initialize Python runtime with mzekezeke ML libraries
    async fn initialize_python_runtime() -> Result<PyObject> {
        Python::with_gil(|py| {
            // Create mzekezeke Python module - the ML workhorse
            let mzekezeke_code = r#"
import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
import networkx as nx
from typing import Dict, List, Tuple, Optional
import time

class MzekezekeBayesianEngine:
    """
    The mzekezeke machine learning workhorse for Bayesian belief networks
    with temporal decay and multi-dimensional truth assessment.
    
    This is the core that actually learns and makes predictions.
    """
    
    def __init__(self):
        self.network = nx.DiGraph()
        self.belief_states = {}
        self.decay_functions = {}
        self.optimization_history = []
        
    def add_evidence_node(self, node_id: str, evidence_data: Dict) -> None:
        """Add a new evidence node to the belief network."""
        self.network.add_node(node_id, **evidence_data)
        self.belief_states[node_id] = {
            'prior': evidence_data.get('prior_probability', 0.5),
            'likelihood': evidence_data.get('likelihood', 0.5),
            'posterior': evidence_data.get('prior_probability', 0.5),
            'decay_state': evidence_data.get('decay_state', 1.0)
        }
    
    def add_belief_edge(self, source: str, target: str, edge_data: Dict) -> None:
        """Add a causal/correlational edge between belief nodes."""
        self.network.add_edge(source, target, **edge_data)
    
    def apply_temporal_decay(self, current_time: float) -> None:
        """Apply temporal decay to all nodes - information decays over time."""
        for node_id in self.network.nodes():
            node_data = self.network.nodes[node_id]
            decay_config = node_data.get('decay_config', {})
            
            # Calculate time elapsed
            created_time = node_data.get('created_at', current_time)
            time_elapsed = current_time - created_time
            
            # Apply decay function
            decay_type = decay_config.get('decay_function', 'exponential')
            decay_rate = decay_config.get('decay_rate', 0.1)
            initial_strength = decay_config.get('initial_strength', 1.0)
            
            if decay_type == 'exponential':
                current_strength = initial_strength * np.exp(-decay_rate * time_elapsed)
            elif decay_type == 'linear':
                current_strength = max(0.0, initial_strength - decay_rate * time_elapsed)
            elif decay_type == 'power_law':
                current_strength = initial_strength * np.power(time_elapsed + 1, -decay_rate)
            elif decay_type == 'logarithmic':
                current_strength = initial_strength * (1 - np.log(1 + decay_rate * time_elapsed))
            else:
                current_strength = initial_strength
            
            # Update belief state
            self.belief_states[node_id]['decay_state'] = max(0.0, current_strength)
    
    def bayesian_update(self, node_id: str, new_evidence: Dict) -> float:
        """Perform Bayesian update for a specific node."""
        if node_id not in self.belief_states:
            return 0.0
        
        state = self.belief_states[node_id]
        
        # Extract evidence dimensions
        likelihood = new_evidence.get('likelihood', 0.5)
        prior = state['prior']
        
        # Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)
        # P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        evidence_prob = likelihood * prior + (1 - likelihood) * (1 - prior)
        
        if evidence_prob > 0:
            posterior = (likelihood * prior) / evidence_prob
        else:
            posterior = prior
        
        # Apply decay factor - information decays over time
        decay_factor = state['decay_state']
        adjusted_posterior = posterior * decay_factor + prior * (1 - decay_factor)
        
        # Update state
        state['posterior'] = adjusted_posterior
        state['likelihood'] = likelihood
        
        return adjusted_posterior
    
    def optimize_network(self, max_iterations: int = 1000, tolerance: float = 1e-6) -> Dict:
        """
        Optimize the entire belief network using variational inference.
        This is where the actual learning happens - the tangible objective function.
        """
        iteration = 0
        prev_likelihood = -np.inf
        
        while iteration < max_iterations:
            total_likelihood = 0.0
            
            # Iterate through all nodes
            for node_id in self.network.nodes():
                # Collect evidence from neighboring nodes
                neighbors = list(self.network.predecessors(node_id))
                neighbor_beliefs = [self.belief_states[n]['posterior'] for n in neighbors if n in self.belief_states]
                
                if neighbor_beliefs:
                    # Aggregate neighbor beliefs (weighted by edge strengths)
                    weighted_beliefs = []
                    total_weight = 0.0
                    
                    for neighbor in neighbors:
                        if neighbor in self.belief_states:
                            edge_data = self.network.edges[neighbor, node_id]
                            weight = edge_data.get('strength', 1.0)
                            belief = self.belief_states[neighbor]['posterior']
                            weighted_beliefs.append(weight * belief)
                            total_weight += abs(weight)
                    
                    if total_weight > 0:
                        aggregated_belief = sum(weighted_beliefs) / total_weight
                    else:
                        aggregated_belief = np.mean(neighbor_beliefs)
                    
                    # Update this node's belief
                    new_evidence = {'likelihood': aggregated_belief}
                    self.bayesian_update(node_id, new_evidence)
                
                # Add to total likelihood (log space for numerical stability)
                posterior = self.belief_states[node_id]['posterior']
                total_likelihood += np.log(max(1e-10, posterior))
            
            # Check convergence
            if abs(total_likelihood - prev_likelihood) < tolerance:
                break
            
            prev_likelihood = total_likelihood
            iteration += 1
        
        return {
            'converged': iteration < max_iterations,
            'iterations': iteration,
            'final_likelihood': total_likelihood,
            'belief_states': self.belief_states.copy()
        }
    
    def predict_belief(self, evidence_content: str, evidence_features: Dict) -> float:
        """
        Make a prediction about belief probability for new evidence.
        This is the core prediction capability.
        """
        # Simple similarity-based prediction for now
        # In a full implementation, this would use sophisticated ML models
        
        max_similarity = 0.0
        best_match_belief = 0.5
        
        for node_id in self.network.nodes():
            node_data = self.network.nodes[node_id]
            node_content = node_data.get('content', '')
            
            # Simple text similarity (could be enhanced with embeddings)
            similarity = self._calculate_similarity(evidence_content, node_content)
            
            if similarity > max_similarity:
                max_similarity = similarity
                if node_id in self.belief_states:
                    best_match_belief = self.belief_states[node_id]['posterior']
        
        # Adjust prediction based on evidence features
        feature_adjustment = self._calculate_feature_adjustment(evidence_features)
        predicted_belief = best_match_belief * feature_adjustment
        
        return max(0.0, min(1.0, predicted_belief))
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_feature_adjustment(self, features: Dict) -> float:
        """Calculate adjustment factor based on evidence features."""
        # Weight different truth dimensions
        factual_accuracy = features.get('factual_accuracy', 0.5)
        source_credibility = features.get('source_credibility', 0.5)
        temporal_validity = features.get('temporal_validity', 0.5)
        
        # Weighted combination
        adjustment = (0.4 * factual_accuracy + 
                     0.3 * source_credibility + 
                     0.3 * temporal_validity)
        
        return adjustment
    
    def get_network_state(self) -> Dict:
        """Get current state of the entire network."""
        return {
            'nodes': dict(self.network.nodes(data=True)),
            'edges': dict(self.network.edges(data=True)),
            'belief_states': self.belief_states.copy(),
            'network_metrics': {
                'num_nodes': self.network.number_of_nodes(),
                'num_edges': self.network.number_of_edges(),
                'density': nx.density(self.network) if self.network.number_of_nodes() > 0 else 0.0,
                'is_connected': nx.is_weakly_connected(self.network) if self.network.number_of_nodes() > 0 else False
            }
        }

# Global mzekezeke instance - the ML workhorse
mzekezeke_engine = MzekezekeBayesianEngine()
"#;
            
            // Execute mzekezeke Python code
            py.run(mzekezeke_code, None, None)?;
            
            // Get reference to the mzekezeke engine
            let globals = py.eval("globals()", None, None)?;
            let mzekezeke = globals.get_item("mzekezeke_engine")?;
            
            Ok(mzekezeke.to_object(py))
        })
    }
    
    /// Add new evidence to the belief network
    pub async fn add_evidence(&self, evidence: Evidence) -> Result<Uuid> {
        info!("Adding evidence to mzekezeke belief network: {}", evidence.id);
        
        // Create belief node
        let belief_node = BeliefNode {
            id: evidence.id,
            evidence: evidence.clone(),
            belief_probability: OrderedFloat(0.5), // Start with neutral belief
            confidence: OrderedFloat(0.5),
            decay_state: TemporalDecayState {
                initial_strength: OrderedFloat(1.0),
                current_strength: OrderedFloat(1.0),
                decay_rate: OrderedFloat(self.get_decay_rate_for_evidence_type(&evidence.evidence_type)),
                half_life: chrono::Duration::hours(24), // Default 24 hour half-life
                time_since_refresh: chrono::Duration::zero(),
                decay_function: DecayFunction::Exponential,
            },
            prior_probability: OrderedFloat(0.5),
            likelihood: OrderedFloat(self.calculate_likelihood(&evidence)),
            posterior_probability: OrderedFloat(0.5),
        };
        
        // Add to graph
        let mut network = self.belief_network.write().await;
        let node_index = network.add_node(belief_node);
        self.node_lookup.insert(evidence.id, node_index);
        
        // Add to Python mzekezeke engine
        self.add_evidence_to_python(&evidence).await?;
        
        Ok(evidence.id)
    }
    
    /// Add evidence to Python mzekezeke engine
    async fn add_evidence_to_python(&self, evidence: &Evidence) -> Result<()> {
        let python_runtime = self.python_runtime.read().await;
        
        Python::with_gil(|py| {
            let mzekezeke = python_runtime.as_ref(py);
            
            // Prepare evidence data for Python
            let evidence_data = PyDict::new(py);
            evidence_data.set_item("node_id", evidence.id.to_string())?;
            evidence_data.set_item("content", &evidence.content)?;
            evidence_data.set_item("created_at", evidence.created_at.timestamp())?;
            evidence_data.set_item("prior_probability", 0.5)?;
            evidence_data.set_item("likelihood", self.calculate_likelihood(evidence))?;
            
            // Add decay configuration
            let decay_config = PyDict::new(py);
            decay_config.set_item("decay_function", "exponential")?;
            decay_config.set_item("decay_rate", self.get_decay_rate_for_evidence_type(&evidence.evidence_type))?;
            decay_config.set_item("initial_strength", 1.0)?;
            evidence_data.set_item("decay_config", decay_config)?;
            
            // Call Python method
            mzekezeke.call_method1("add_evidence_node", (evidence.id.to_string(), evidence_data))?;
            
            Ok(())
        })
    }
    
    /// Calculate likelihood based on evidence truth dimensions
    fn calculate_likelihood(&self, evidence: &Evidence) -> f64 {
        let dims = &evidence.truth_dimensions;
        
        // Weighted combination of truth dimensions
        let weights = [0.25, 0.20, 0.15, 0.20, 0.10, 0.10]; // Must sum to 1.0
        let values = [
            dims.factual_accuracy.into_inner(),
            dims.contextual_relevance.into_inner(),
            dims.temporal_validity.into_inner(),
            dims.source_credibility.into_inner(),
            dims.logical_consistency.into_inner(),
            dims.empirical_support.into_inner(),
        ];
        
        weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum()
    }
    
    /// Get decay rate for evidence type
    fn get_decay_rate_for_evidence_type(&self, evidence_type: &EvidenceType) -> f64 {
        self.decay_config.default_decay_rates
            .get(evidence_type)
            .copied()
            .unwrap_or(0.1) // Default decay rate
    }
    
    /// Optimize the entire belief network - this is the core learning process
    pub async fn optimize_network(&self) -> Result<OptimizationResult> {
        info!("Optimizing mzekezeke belief network - running the tangible objective function");
        
        // Apply temporal decay first
        self.apply_temporal_decay().await?;
        
        // Run Python optimization - this is where the actual learning happens
        let python_runtime = self.python_runtime.read().await;
        let optimization_result = Python::with_gil(|py| {
            let mzekezeke = python_runtime.as_ref(py);
            let result = mzekezeke.call_method1("optimize_network", (1000, 1e-6))?;
            
            // Extract results
            let result_dict = result.downcast::<PyDict>()?;
            let converged = result_dict.get_item("converged")?.extract::<bool>()?;
            let iterations = result_dict.get_item("iterations")?.extract::<u64>()?;
            let final_likelihood = result_dict.get_item("final_likelihood")?.extract::<f64>()?;
            
            Ok::<OptimizationResult, PyErr>(OptimizationResult {
                converged,
                iterations,
                final_likelihood: OrderedFloat(final_likelihood),
                network_state: NetworkState {
                    num_nodes: 0,
                    num_edges: 0,
                    total_likelihood: OrderedFloat(final_likelihood),
                    convergence_status: converged,
                },
            })
        })?;
        
        // Update optimization state
        let mut opt_state = self.optimization_state.write().await;
        opt_state.iteration += optimization_result.iterations;
        opt_state.likelihood_score = optimization_result.final_likelihood;
        opt_state.converged = optimization_result.converged;
        
        Ok(optimization_result)
    }
    
    /// Apply temporal decay to all evidence nodes
    pub async fn apply_temporal_decay(&self) -> Result<()> {
        debug!("Applying temporal decay to all evidence nodes");
        
        let current_time = Utc::now();
        
        // Apply decay to Python engine
        let python_runtime = self.python_runtime.read().await;
        Python::with_gil(|py| {
            let mzekezeke = python_runtime.as_ref(py);
            mzekezeke.call_method1("apply_temporal_decay", (current_time.timestamp(),))?;
            Ok::<(), PyErr>(())
        })?;
        
        Ok(())
    }
    
    /// Make a prediction using the trained network
    pub async fn predict_belief(&self, evidence_content: &str, truth_dimensions: &TruthDimensions) -> Result<f64> {
        info!("Making belief prediction with mzekezeke for: {}", evidence_content);
        
        let python_runtime = self.python_runtime.read().await;
        let prediction = Python::with_gil(|py| {
            let mzekezeke = python_runtime.as_ref(py);
            
            // Prepare evidence features
            let features = PyDict::new(py);
            features.set_item("factual_accuracy", truth_dimensions.factual_accuracy.into_inner())?;
            features.set_item("source_credibility", truth_dimensions.source_credibility.into_inner())?;
            features.set_item("temporal_validity", truth_dimensions.temporal_validity.into_inner())?;
            
            // Call prediction method
            let result = mzekezeke.call_method1("predict_belief", (evidence_content, features))?;
            let prediction = result.extract::<f64>()?;
            
            Ok::<f64, PyErr>(prediction)
        })?;
        
        Ok(prediction)
    }
    
    /// Get belief probability for specific evidence
    pub async fn get_belief_probability(&self, evidence_id: Uuid) -> Result<f64> {
        let network = self.belief_network.read().await;
        
        if let Some(&node_idx) = self.node_lookup.get(&evidence_id) {
            if let Some(node) = network.node_weight(node_idx) {
                return Ok(node.posterior_probability.into_inner() * 
                         node.decay_state.current_strength.into_inner());
            }
        }
        
        Err(anyhow!("Evidence not found: {}", evidence_id))
    }
}

/// Result of network optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub converged: bool,
    pub iterations: u64,
    pub final_likelihood: OrderedFloat<f64>,
    pub network_state: NetworkState,
}

/// Current state of the belief network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    pub num_nodes: usize,
    pub num_edges: usize,
    pub total_likelihood: OrderedFloat<f64>,
    pub convergence_status: bool,
}

impl Default for TemporalDecayConfig {
    fn default() -> Self {
        let mut default_decay_rates = HashMap::new();
        default_decay_rates.insert(EvidenceType::DirectObservation, 0.05);
        default_decay_rates.insert(EvidenceType::IndirectInference, 0.1);
        default_decay_rates.insert(EvidenceType::StatisticalCorrelation, 0.15);
        default_decay_rates.insert(EvidenceType::ExpertOpinion, 0.2);
        default_decay_rates.insert(EvidenceType::ExperimentalResult, 0.08);
        default_decay_rates.insert(EvidenceType::TheoreticalPrediction, 0.25);
        
        Self {
            default_decay_rates,
            minimum_strength_threshold: OrderedFloat(0.01),
            refresh_interval: chrono::Duration::hours(1),
            adaptive_decay: true,
        }
    }
} 