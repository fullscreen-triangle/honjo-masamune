//! Monitoring and Health Check Module
//! 
//! Provides health monitoring, metrics collection, and system status
//! reporting for the Honjo Masamune truth engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub overall: HealthLevel,
    pub components: HashMap<String, ComponentHealth>,
    pub timestamp: DateTime<Utc>,
}

/// Health levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthLevel {
    Healthy,
    Degraded,
    Critical,
    Down,
}

/// Component health information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthLevel,
    pub message: String,
    pub last_check: DateTime<Utc>,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self {
            overall: HealthLevel::Healthy,
            components: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
} 