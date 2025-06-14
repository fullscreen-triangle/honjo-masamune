//! Local-First Data Processor
//! 
//! Implements Bloodhound's revolutionary approach where data never leaves its source.
//! All processing happens locally, with only patterns and insights shared.

use crate::{DataSource, ValidationPattern, LocalFirstProcessor};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::path::Path;
use tracing::{info, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfig {
    pub memory_limit_gb: u64,
    pub cpu_cores: u32,
    pub processing_threads: u32,
    pub batch_size: u32,
}

#[derive(Debug)]
pub struct LocalProcessor {
    config: OptimalConfig,
}

impl LocalProcessor {
    pub async fn new(config: OptimalConfig) -> Result<Self> {
        info!("Initializing local processor with optimal configuration");
        debug!("Config: {:?}", config);
        
        Ok(Self { config })
    }
    
    /// Auto-detect system capabilities and optimize processing
    pub async fn auto_optimize(&mut self) -> Result<()> {
        // Detect available system resources
        let available_memory = self.detect_available_memory().await?;
        let cpu_cores = self.detect_cpu_cores().await?;
        
        // Optimize configuration based on system capabilities
        self.config.memory_limit_gb = (available_memory * 0.8) as u64; // Use 80% of available memory
        self.config.cpu_cores = cpu_cores;
        self.config.processing_threads = cpu_cores * 2; // Hyperthreading consideration
        self.config.batch_size = std::cmp::max(1000, available_memory as u32 / 100);
        
        info!("Auto-optimized local processor configuration: {:?}", self.config);
        Ok(())
    }
    
    async fn detect_available_memory(&self) -> Result<f64> {
        // Simplified memory detection - in real implementation would use system APIs
        Ok(8.0) // 8GB default
    }
    
    async fn detect_cpu_cores(&self) -> Result<u32> {
        Ok(num_cpus::get() as u32)
    }
}

#[async_trait]
impl LocalFirstProcessor for LocalProcessor {
    async fn process_locally<P: AsRef<Path> + Send>(&self, data_path: P) -> Result<Vec<ValidationPattern>> {
        let path = data_path.as_ref();
        info!("Processing data locally: {:?}", path);
        
        // Verify file exists and is accessible
        if !path.exists() {
            return Err(anyhow::anyhow!("Data file does not exist: {:?}", path));
        }
        
        // Extract patterns without moving data
        let patterns = self.extract_patterns_from_file(path).await?;
        
        debug!("Extracted {} patterns from local file", patterns.len());
        Ok(patterns)
    }
    
    async fn extract_patterns(&self, data: &[DataSource], query: &str) -> Result<Vec<ValidationPattern>> {
        info!("Extracting patterns from {} data sources for query: {}", data.len(), query);
        
        let mut all_patterns = Vec::new();
        
        for source in data {
            match self.process_data_source(source, query).await {
                Ok(mut patterns) => all_patterns.append(&mut patterns),
                Err(e) => {
                    tracing::warn!("Failed to process data source {}: {}", source.path, e);
                    continue;
                }
            }
        }
        
        // Deduplicate and rank patterns
        let ranked_patterns = self.rank_patterns(all_patterns, query).await?;
        
        info!("Extracted {} unique patterns", ranked_patterns.len());
        Ok(ranked_patterns)
    }
    
    async fn validate_assumptions(&self, patterns: &[ValidationPattern]) -> Result<bool> {
        info!("Validating statistical assumptions for {} patterns", patterns.len());
        
        // Check pattern distribution
        let confidence_scores: Vec<f64> = patterns.iter().map(|p| p.confidence).collect();
        
        // Basic validation checks
        let mean_confidence = confidence_scores.iter().sum::<f64>() / confidence_scores.len() as f64;
        let has_sufficient_confidence = mean_confidence > 0.5;
        let has_sufficient_patterns = patterns.len() >= 3;
        
        let valid = has_sufficient_confidence && has_sufficient_patterns;
        
        info!("Assumption validation result: {} (mean confidence: {:.3})", valid, mean_confidence);
        Ok(valid)
    }
}

impl LocalProcessor {
    async fn extract_patterns_from_file(&self, path: &Path) -> Result<Vec<ValidationPattern>> {
        // Simplified pattern extraction - in real implementation would analyze file content
        let file_size = std::fs::metadata(path)?.len();
        let pattern_count = std::cmp::min(10, file_size / 1000); // Rough heuristic
        
        let mut patterns = Vec::new();
        for i in 0..pattern_count {
            patterns.push(ValidationPattern {
                pattern_hash: format!("pattern_{}_{}", path.file_name().unwrap().to_string_lossy(), i),
                confidence: 0.7 + (i as f64 * 0.05), // Increasing confidence
                source_count: 1,
            });
        }
        
        Ok(patterns)
    }
    
    async fn process_data_source(&self, source: &DataSource, query: &str) -> Result<Vec<ValidationPattern>> {
        debug!("Processing data source: {} for query: {}", source.path, query);
        
        // Health check
        if !matches!(source.health_status, crate::DataHealthStatus::Healthy) {
            return Err(anyhow::anyhow!("Data source is not healthy: {:?}", source.health_status));
        }
        
        // Process based on source type
        match source.source_type {
            crate::DataSourceType::LocalFile => {
                self.process_locally(&source.path).await
            },
            crate::DataSourceType::LocalDatabase => {
                self.process_local_database(&source.path, query).await
            },
            _ => {
                Err(anyhow::anyhow!("Remote processing not supported in local-first mode"))
            }
        }
    }
    
    async fn process_local_database(&self, connection_string: &str, query: &str) -> Result<Vec<ValidationPattern>> {
        debug!("Processing local database: {} with query: {}", connection_string, query);
        
        // Simplified database processing
        let hash = sha2::Sha256::digest(query.as_bytes());
        Ok(vec![
            ValidationPattern {
                pattern_hash: format!("db_pattern_{:x}", hash),
                confidence: 0.8,
                source_count: 1,
            }
        ])
    }
    
    async fn rank_patterns(&self, mut patterns: Vec<ValidationPattern>, query: &str) -> Result<Vec<ValidationPattern>> {
        // Remove duplicates based on pattern hash
        patterns.sort_by(|a, b| a.pattern_hash.cmp(&b.pattern_hash));
        patterns.dedup_by(|a, b| a.pattern_hash == b.pattern_hash);
        
        // Rank by confidence and relevance to query
        patterns.sort_by(|a, b| {
            let relevance_a = self.calculate_relevance(&a.pattern_hash, query);
            let relevance_b = self.calculate_relevance(&b.pattern_hash, query);
            let score_a = a.confidence * relevance_a;
            let score_b = b.confidence * relevance_b;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        Ok(patterns)
    }
    
    fn calculate_relevance(&self, pattern_hash: &str, query: &str) -> f64 {
        // Simplified relevance calculation
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let pattern_words: Vec<&str> = pattern_hash.split('_').collect();
        
        let matches = query_words.iter()
            .filter(|&word| pattern_words.iter().any(|&p| p.contains(word)))
            .count();
        
        if query_words.is_empty() {
            0.5
        } else {
            matches as f64 / query_words.len() as f64
        }
    }
} 