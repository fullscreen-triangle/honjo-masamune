//! Repository Interfaces Module
//! 
//! Provides interfaces and implementations for connecting to external
//! repositories and data sources used by Honjo Masamune.

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use fuzzy_logic_core::FuzzyTruth;

/// Repository result from external queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryResult {
    pub data: serde_json::Value,
    pub confidence: FuzzyTruth,
    pub source: String,
    pub timestamp: DateTime<Utc>,
}

/// Repository interface trait
#[async_trait]
pub trait RepositoryInterface: Send + Sync {
    /// Query the repository
    async fn query(&self, query: &str, parameters: &HashMap<String, String>) -> Result<Vec<RepositoryResult>>;
    
    /// Get repository capabilities
    fn capabilities(&self) -> Vec<String>;
    
    /// Get repository confidence model
    fn confidence_model(&self) -> String;
    
    /// Get repository name
    fn name(&self) -> String;
}

/// Mock repository for testing
pub struct MockRepository {
    name: String,
}

impl MockRepository {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait]
impl RepositoryInterface for MockRepository {
    async fn query(&self, _query: &str, _parameters: &HashMap<String, String>) -> Result<Vec<RepositoryResult>> {
        Ok(vec![RepositoryResult {
            data: serde_json::json!({"mock": "data"}),
            confidence: FuzzyTruth::new_unchecked(0.8),
            source: self.name.clone(),
            timestamp: Utc::now(),
        }])
    }
    
    fn capabilities(&self) -> Vec<String> {
        vec!["mock_query".to_string()]
    }
    
    fn confidence_model(&self) -> String {
        "mock_model".to_string()
    }
    
    fn name(&self) -> String {
        self.name.clone()
    }
} 