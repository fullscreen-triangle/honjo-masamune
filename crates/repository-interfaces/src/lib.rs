//! Repository Interfaces
//! 
//! Provides interfaces for connecting to external repositories and data sources.

use async_trait::async_trait;
use anyhow::Result;
use std::collections::HashMap;

#[async_trait]
pub trait RepositoryInterface: Send + Sync {
    async fn query(&self, query: &str, params: &HashMap<String, String>) -> Result<Vec<String>>;
    fn capabilities(&self) -> Vec<String>;
} 