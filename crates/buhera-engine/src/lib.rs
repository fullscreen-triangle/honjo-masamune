//! Buhera Logical Programming Engine
//! 
//! Named after Buhera district in Zimbabwe, this crate implements a logical
//! programming language designed specifically for truth synthesis and reasoning
//! within the Honjo Masamune truth engine.

use anyhow::Result;
use atp_manager::{AtpManager, AtpReservation};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use fuzzy_logic_core::{FuzzyTruth, FuzzyResult, TruthMembership};
use indexmap::IndexMap;
use parking_lot::RwLock;
use pest::Parser;
use pest_derive::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

pub mod ast;
pub mod interpreter;
pub mod knowledge_base;
pub mod unification;
pub mod resolution;
pub mod fuzzy_inference;

pub use ast::*;
pub use interpreter::*;
pub use knowledge_base::*;
pub use unification::*;
pub use resolution::*;
pub use fuzzy_inference::*;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct BuheraParser;

/// Buhera program representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuheraProgram {
    pub id: Uuid,
    pub name: String,
    pub statements: Vec<Statement>,
    pub metadata: ProgramMetadata,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
}

impl BuheraProgram {
    /// Create a new Buhera program
    pub fn new(name: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            name,
            statements: Vec::new(),
            metadata: ProgramMetadata::default(),
            created_at: now,
            modified_at: now,
        }
    }

    /// Parse a Buhera program from source code
    pub fn parse(name: String, source: &str) -> Result<Self> {
        let pairs = BuheraParser::parse(Rule::program, source)?;
        let mut program = Self::new(name);
        
        for pair in pairs {
            match pair.as_rule() {
                Rule::statement => {
                    let statement = Statement::from_pest(pair)?;
                    program.statements.push(statement);
                }
                Rule::EOI => break,
                _ => {}
            }
        }
        
        Ok(program)
    }

    /// Add a statement to the program
    pub fn add_statement(&mut self, statement: Statement) {
        self.statements.push(statement);
        self.modified_at = Utc::now();
    }

    /// Get all facts in the program
    pub fn facts(&self) -> Vec<&Fact> {
        self.statements
            .iter()
            .filter_map(|s| match s {
                Statement::Fact(fact) => Some(fact),
                _ => None,
            })
            .collect()
    }

    /// Get all rules in the program
    pub fn rules(&self) -> Vec<&Rule> {
        self.statements
            .iter()
            .filter_map(|s| match s {
                Statement::Rule(rule) => Some(rule),
                _ => None,
            })
            .collect()
    }

    /// Get all queries in the program
    pub fn queries(&self) -> Vec<&Query> {
        self.statements
            .iter()
            .filter_map(|s| match s {
                Statement::Query(query) => Some(query),
                _ => None,
            })
            .collect()
    }
}

/// Program metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgramMetadata {
    pub author: Option<String>,
    pub version: String,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub ceremonial_mode: bool,
    pub atp_budget: Option<u64>,
    pub confidence_threshold: FuzzyTruth,
}

impl Default for ProgramMetadata {
    fn default() -> Self {
        Self {
            author: None,
            version: "1.0.0".to_string(),
            description: None,
            tags: Vec::new(),
            ceremonial_mode: false,
            atp_budget: None,
            confidence_threshold: FuzzyTruth::new_unchecked(0.7),
        }
    }
}

/// Buhera execution engine
#[derive(Debug)]
pub struct BuheraEngine {
    knowledge_base: Arc<RwLock<KnowledgeBase>>,
    interpreter: Arc<BuheraInterpreter>,
    atp_manager: Arc<AtpManager>,
    execution_context: Arc<RwLock<ExecutionContext>>,
    repositories: Arc<DashMap<String, Box<dyn RepositoryInterface>>>,
}

impl BuheraEngine {
    /// Create a new Buhera engine
    pub fn new(atp_manager: Arc<AtpManager>) -> Self {
        let knowledge_base = Arc::new(RwLock::new(KnowledgeBase::new()));
        let interpreter = Arc::new(BuheraInterpreter::new(atp_manager.clone()));
        let execution_context = Arc::new(RwLock::new(ExecutionContext::new()));
        let repositories = Arc::new(DashMap::new());

        Self {
            knowledge_base,
            interpreter,
            atp_manager,
            execution_context,
            repositories,
        }
    }

    /// Load a Buhera program into the knowledge base
    pub async fn load_program(&self, program: BuheraProgram) -> Result<()> {
        let atp_cost = self.calculate_load_cost(&program);
        let reservation = self.atp_manager.reserve_atp("load_program", atp_cost).await?;

        let mut kb = self.knowledge_base.write();
        
        // Process each statement
        for statement in &program.statements {
            match statement {
                Statement::Fact(fact) => {
                    kb.add_fact(fact.clone())?;
                }
                Statement::Rule(rule) => {
                    kb.add_rule(rule.clone())?;
                }
                Statement::Directive(directive) => {
                    self.process_directive(directive).await?;
                }
                _ => {} // Queries are processed separately
            }
        }

        kb.add_program(program);
        self.atp_manager.consume_atp(reservation, "load_program").await?;
        
        Ok(())
    }

    /// Execute a query against the knowledge base
    pub async fn query(&self, query: &Query) -> Result<FuzzyResult<Vec<Solution>>> {
        let atp_cost = self.calculate_query_cost(query);
        let reservation = self.atp_manager.reserve_atp("query_execution", atp_cost).await?;

        let result = self.interpreter.execute_query(
            query,
            &self.knowledge_base,
            &self.execution_context,
        ).await?;

        self.atp_manager.consume_atp(reservation, "query_execution").await?;
        
        Ok(result)
    }

    /// Execute a Buhera program
    pub async fn execute_program(&self, program: &BuheraProgram) -> Result<Vec<FuzzyResult<Vec<Solution>>>> {
        let mut results = Vec::new();
        
        // Load the program first
        self.load_program(program.clone()).await?;
        
        // Execute all queries in the program
        for query in program.queries() {
            let result = self.query(query).await?;
            results.push(result);
        }
        
        Ok(results)
    }

    /// Register a repository interface
    pub fn register_repository(&self, name: String, repository: Box<dyn RepositoryInterface>) {
        self.repositories.insert(name, repository);
    }

    /// Get knowledge base statistics
    pub fn knowledge_base_stats(&self) -> KnowledgeBaseStats {
        self.knowledge_base.read().stats()
    }

    /// Process a directive
    async fn process_directive(&self, directive: &Directive) -> Result<()> {
        match &directive.body {
            DirectiveBody::UseRepository(repo_name) => {
                // Repository registration would be handled here
                tracing::info!("Using repository: {}", repo_name);
            }
            DirectiveBody::SetThreshold(threshold_type, value) => {
                let mut context = self.execution_context.write();
                context.set_threshold(threshold_type.clone(), *value);
            }
            DirectiveBody::EnableDreaming(enabled) => {
                let mut context = self.execution_context.write();
                context.dreaming_enabled = enabled.value() > 0.5;
            }
            DirectiveBody::AtpCost(operation, cost) => {
                let mut context = self.execution_context.write();
                context.custom_atp_costs.insert(operation.clone(), *cost as u64);
            }
            DirectiveBody::GrayAreaDetection(lower, upper) => {
                let mut context = self.execution_context.write();
                context.gray_area_range = [lower.value(), upper.value()];
            }
        }
        Ok(())
    }

    /// Calculate ATP cost for loading a program
    fn calculate_load_cost(&self, program: &BuheraProgram) -> u64 {
        let base_cost = 100;
        let statement_cost = program.statements.len() as u64 * 10;
        let complexity_cost = self.calculate_complexity_cost(program);
        
        base_cost + statement_cost + complexity_cost
    }

    /// Calculate ATP cost for executing a query
    fn calculate_query_cost(&self, query: &Query) -> u64 {
        let base_cost = 200;
        let predicate_cost = query.body.predicates.len() as u64 * 50;
        let fuzzy_cost = if query.body.has_fuzzy_operations() { 100 } else { 0 };
        
        base_cost + predicate_cost + fuzzy_cost
    }

    /// Calculate complexity cost for a program
    fn calculate_complexity_cost(&self, program: &BuheraProgram) -> u64 {
        let mut cost = 0;
        
        for statement in &program.statements {
            cost += match statement {
                Statement::Fact(_) => 5,
                Statement::Rule(rule) => 20 + rule.body.predicates.len() as u64 * 10,
                Statement::Query(query) => 30 + query.body.predicates.len() as u64 * 15,
                Statement::Directive(_) => 10,
                Statement::FuzzyExpression(_) => 25,
                Statement::Uncertainty(_) => 40,
                Statement::TemporalExpression(_) => 35,
                Statement::ModalExpression(_) => 30,
                Statement::Aggregation(_) => 50,
                Statement::MetaPredicate(_) => 60,
            };
        }
        
        cost
    }
}

/// Execution context for Buhera programs
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub thresholds: HashMap<String, FuzzyTruth>,
    pub dreaming_enabled: bool,
    pub ceremonial_mode: bool,
    pub custom_atp_costs: HashMap<String, u64>,
    pub gray_area_range: [f64; 2],
    pub max_inference_depth: usize,
    pub timeout_seconds: u64,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("certain".to_string(), FuzzyTruth::new_unchecked(0.95));
        thresholds.insert("probable".to_string(), FuzzyTruth::new_unchecked(0.75));
        thresholds.insert("possible".to_string(), FuzzyTruth::new_unchecked(0.5));
        
        Self {
            thresholds,
            dreaming_enabled: false,
            ceremonial_mode: false,
            custom_atp_costs: HashMap::new(),
            gray_area_range: [0.4, 0.7],
            max_inference_depth: 100,
            timeout_seconds: 300,
        }
    }

    /// Set a threshold value
    pub fn set_threshold(&mut self, threshold_type: String, value: FuzzyTruth) {
        self.thresholds.insert(threshold_type, value);
    }

    /// Get a threshold value
    pub fn get_threshold(&self, threshold_type: &str) -> Option<FuzzyTruth> {
        self.thresholds.get(threshold_type).copied()
    }

    /// Check if in ceremonial mode
    pub fn is_ceremonial(&self) -> bool {
        self.ceremonial_mode
    }
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Solution to a query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    pub id: Uuid,
    pub bindings: HashMap<String, Term>,
    pub confidence: FuzzyTruth,
    pub derivation_path: Vec<String>,
    pub atp_cost: u64,
    pub timestamp: DateTime<Utc>,
}

impl Solution {
    /// Create a new solution
    pub fn new(
        bindings: HashMap<String, Term>,
        confidence: FuzzyTruth,
        derivation_path: Vec<String>,
        atp_cost: u64,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            bindings,
            confidence,
            derivation_path,
            atp_cost,
            timestamp: Utc::now(),
        }
    }

    /// Check if this solution meets the confidence threshold
    pub fn meets_threshold(&self, threshold: FuzzyTruth) -> bool {
        self.confidence.value() >= threshold.value()
    }

    /// Get the truth membership for this solution
    pub fn membership(&self, thresholds: &HashMap<String, FuzzyTruth>) -> TruthMembership {
        if let Some(certain) = thresholds.get("certain") {
            if self.confidence.value() >= certain.value() {
                return TruthMembership::Certain;
            }
        }
        
        if let Some(probable) = thresholds.get("probable") {
            if self.confidence.value() >= probable.value() {
                return TruthMembership::Probable;
            }
        }
        
        if let Some(possible) = thresholds.get("possible") {
            if self.confidence.value() >= possible.value() {
                return TruthMembership::Possible;
            }
        }
        
        TruthMembership::Unlikely
    }
}

/// Repository interface for external data sources
#[async_trait::async_trait]
pub trait RepositoryInterface: Send + Sync {
    /// Query the repository
    async fn query(&self, query: &str, parameters: &HashMap<String, String>) -> Result<Vec<RepositoryResult>>;
    
    /// Get repository capabilities
    fn capabilities(&self) -> Vec<String>;
    
    /// Get repository confidence model
    fn confidence_model(&self) -> String;
}

/// Result from a repository query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryResult {
    pub data: serde_json::Value,
    pub confidence: FuzzyTruth,
    pub source: String,
    pub timestamp: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use atp_manager::AtpCosts;

    #[test]
    fn test_program_creation() {
        let program = BuheraProgram::new("test_program".to_string());
        assert_eq!(program.name, "test_program");
        assert!(program.statements.is_empty());
    }

    #[tokio::test]
    async fn test_engine_creation() {
        let atp_manager = Arc::new(AtpManager::new(1000, 10000, 100, 50, AtpCosts::default()));
        let engine = BuheraEngine::new(atp_manager);
        
        let stats = engine.knowledge_base_stats();
        assert_eq!(stats.fact_count, 0);
        assert_eq!(stats.rule_count, 0);
    }

    #[test]
    fn test_execution_context() {
        let context = ExecutionContext::new();
        assert!(!context.ceremonial_mode);
        assert!(!context.dreaming_enabled);
        assert_eq!(context.max_inference_depth, 100);
    }

    #[test]
    fn test_solution_creation() {
        let mut bindings = HashMap::new();
        bindings.insert("X".to_string(), Term::Atom("test".to_string()));
        
        let solution = Solution::new(
            bindings,
            FuzzyTruth::new_unchecked(0.8),
            vec!["rule1".to_string()],
            100,
        );
        
        assert_eq!(solution.confidence.value(), 0.8);
        assert_eq!(solution.atp_cost, 100);
    }
} 