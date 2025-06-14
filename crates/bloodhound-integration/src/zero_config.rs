//! Zero Configuration Manager
//! 
//! Automatically detects system resources, optimizes configurations,
//! and manages all technical aspects without user intervention.

use crate::local_processor::OptimalConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, debug};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub max_peers: u32,
    pub connection_timeout_ms: u64,
    pub retry_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    pub enforce_local_processing: bool,
    pub allow_pattern_sharing: bool,
    pub require_encryption: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiConfig {
    pub natural_language_enabled: bool,
    pub explanation_detail_level: String,
    pub auto_assumption_validation: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretedRequest {
    pub structured_query: String,
    pub requires_validation: bool,
    pub confidence_threshold: f64,
}

#[derive(Debug)]
pub struct ZeroConfigManager {
    optimal_config: OptimalConfig,
    network_config: NetworkConfig,
    privacy_config: PrivacyConfig,
    ai_config: AiConfig,
}

impl ZeroConfigManager {
    /// Auto-initialize with zero configuration required
    pub async fn auto_initialize() -> Result<Self> {
        info!("Auto-initializing system with zero configuration");
        
        // Detect system capabilities
        let memory_gb = Self::detect_memory().await?;
        let cpu_cores = Self::detect_cpu_cores().await?;
        let network_capability = Self::detect_network_capability().await?;
        
        // Generate optimal configurations
        let optimal_config = OptimalConfig {
            memory_limit_gb: (memory_gb * 0.8) as u64,
            cpu_cores,
            processing_threads: cpu_cores * 2,
            batch_size: std::cmp::max(1000, (memory_gb * 1000.0) as u32 / 100),
        };
        
        let network_config = NetworkConfig {
            max_peers: if network_capability > 100.0 { 10 } else { 5 },
            connection_timeout_ms: 5000,
            retry_attempts: 3,
        };
        
        let privacy_config = PrivacyConfig {
            enforce_local_processing: true,
            allow_pattern_sharing: true,
            require_encryption: true,
        };
        
        let ai_config = AiConfig {
            natural_language_enabled: true,
            explanation_detail_level: "detailed".to_string(),
            auto_assumption_validation: true,
        };
        
        info!("Zero configuration complete - system optimized for local resources");
        debug!("Optimal config: {:?}", optimal_config);
        
        Ok(Self {
            optimal_config,
            network_config,
            privacy_config,
            ai_config,
        })
    }
    
    pub fn get_optimal_config(&self) -> OptimalConfig {
        self.optimal_config.clone()
    }
    
    pub fn get_network_config(&self) -> NetworkConfig {
        self.network_config.clone()
    }
    
    pub fn get_privacy_config(&self) -> PrivacyConfig {
        self.privacy_config.clone()
    }
    
    pub fn get_ai_config(&self) -> AiConfig {
        self.ai_config.clone()
    }
    
    /// Auto-detect relevant data sources for a query
    pub async fn auto_detect_relevant_sources(
        &self,
        request: &InterpretedRequest,
    ) -> Result<Vec<crate::DataSource>> {
        info!("Auto-detecting relevant data sources for query: {}", request.structured_query);
        
        let mut sources = Vec::new();
        
        // Scan common data directories
        let common_paths = vec![
            "./data",
            "./datasets",
            "./input",
            "~/Documents/data",
            "~/Desktop/data",
        ];
        
        for path in common_paths {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    if let Some(source) = self.evaluate_data_source(&entry.path(), &request.structured_query).await? {
                        sources.push(source);
                    }
                }
            }
        }
        
        info!("Auto-detected {} relevant data sources", sources.len());
        Ok(sources)
    }
    
    // Private helper methods
    
    async fn detect_memory() -> Result<f64> {
        // Simplified memory detection
        // In real implementation, would use system APIs
        Ok(8.0) // 8GB default
    }
    
    async fn detect_cpu_cores() -> Result<u32> {
        Ok(num_cpus::get() as u32)
    }
    
    async fn detect_network_capability() -> Result<f64> {
        // Simplified network capability detection
        // In real implementation, would test network speed
        Ok(100.0) // 100 Mbps default
    }
    
    async fn evaluate_data_source(
        &self,
        path: &std::path::Path,
        query: &str,
    ) -> Result<Option<crate::DataSource>> {
        if !path.is_file() {
            return Ok(None);
        }
        
        // Check file extension and relevance
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");
        
        let is_data_file = matches!(extension, "csv" | "json" | "xml" | "txt" | "tsv" | "parquet");
        
        if !is_data_file {
            return Ok(None);
        }
        
        // Simple relevance check based on filename
        let filename = path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");
        
        let query_words: Vec<&str> = query.split_whitespace().collect();
        let is_relevant = query_words.iter()
            .any(|&word| filename.to_lowercase().contains(&word.to_lowercase()));
        
        if !is_relevant {
            return Ok(None);
        }
        
        // Perform basic health check
        let health_status = if path.exists() && std::fs::metadata(path).is_ok() {
            crate::DataHealthStatus::Healthy
        } else {
            crate::DataHealthStatus::Inaccessible
        };
        
        Ok(Some(crate::DataSource {
            path: path.to_string_lossy().to_string(),
            source_type: crate::DataSourceType::LocalFile,
            health_status,
        }))
    }
} 