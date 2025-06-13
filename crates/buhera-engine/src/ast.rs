//! Abstract Syntax Tree for Buhera Language

use anyhow::Result;
use fuzzy_logic_core::FuzzyTruth;
use pest::iterators::Pair;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    Fact(Fact),
    Rule(Rule),
    Query(Query),
    Directive(Directive),
    FuzzyExpression(FuzzyExpression),
    Uncertainty(Uncertainty),
    TemporalExpression(TemporalExpression),
    ModalExpression(ModalExpression),
    Aggregation(Aggregation),
    MetaPredicate(MetaPredicate),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fact {
    pub predicate: Predicate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    pub head: Predicate,
    pub body: ClauseBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Query {
    pub body: ClauseBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseBody {
    pub predicates: Vec<Predicate>,
}

impl ClauseBody {
    pub fn has_fuzzy_operations(&self) -> bool {
        // Simplified check
        false
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predicate {
    pub name: String,
    pub args: Vec<Term>,
    pub confidence: Option<FuzzyTruth>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Term {
    Atom(String),
    Variable(String),
    Number(f64),
    String(String),
    List(Vec<Term>),
    Compound(String, Vec<Term>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Directive {
    pub body: DirectiveBody,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectiveBody {
    UseRepository(String),
    SetThreshold(String, FuzzyTruth),
    EnableDreaming(FuzzyTruth),
    AtpCost(String, f64),
    GrayAreaDetection(FuzzyTruth, FuzzyTruth),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyExpression {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Uncertainty {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalExpression {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalExpression {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Aggregation {
    pub expr: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaPredicate {
    pub expr: String,
}

// Simplified implementation for now
impl Statement {
    pub fn from_pest(_pair: Pair<crate::Rule>) -> Result<Self> {
        // This would be a complex parser implementation
        // For now, return a simple fact
        Ok(Statement::Fact(Fact {
            predicate: Predicate {
                name: "test".to_string(),
                args: vec![],
                confidence: Some(FuzzyTruth::new_unchecked(0.8)),
            },
        }))
    }
} 