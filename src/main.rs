use anyhow::Result;
use clap::{Arg, Command};
use std::sync::Arc;
use tokio::signal;
use tracing::{info, warn, error};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod engine;
mod api;
mod health;

use crate::config::HonjoMasamuneConfig;
use crate::engine::HonjoMasamuneEngine;
use crate::api::ApiServer;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "honjo_masamune=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let matches = Command::new("honjo-masamune")
        .version("0.1.0")
        .author("fullscreen-triangle")
        .about("The Ultimate Truth Engine - Biomimetic metacognitive truth synthesis system")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config/honjo-masamune.yml"),
        )
        .arg(
            Arg::new("ceremonial")
                .long("ceremonial")
                .help("Enable ceremonial mode (production only)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("port")
                .short('p')
                .long("port")
                .value_name("PORT")
                .help("API server port")
                .default_value("8080"),
        )
        .get_matches();

    let config_path = matches.get_one::<String>("config").unwrap();
    let ceremonial_mode = matches.get_flag("ceremonial");
    let port = matches.get_one::<String>("port").unwrap().parse::<u16>()?;

    info!("🗾 Honjo Masamune Truth Engine starting...");
    
    if ceremonial_mode {
        warn!("⚔️  CEREMONIAL MODE ACTIVATED - Each use permanently closes discussion on topics");
        warn!("⚔️  This is the legendary sword - there is no going back");
    }

    // Load configuration
    info!("📋 Loading configuration from: {}", config_path);
    let mut config = HonjoMasamuneConfig::load(config_path).await?;
    
    if ceremonial_mode {
        config.system.ceremonial_mode = true;
        info!("⚔️  Configuration updated for ceremonial mode");
    }

    // Validate elite organization access
    if config.security.authorization.elite_organizations_only {
        info!("🏛️  Elite organization verification required");
        // TODO: Implement certificate-based authentication
    }

    // Initialize the core engine
    info!("🧠 Initializing Honjo Masamune Engine...");
    let engine = Arc::new(HonjoMasamuneEngine::new(config.clone()).await?);

    // Check system readiness
    info!("🔍 Checking system readiness...");
    let readiness = engine.check_readiness().await?;
    
    match readiness.level {
        crate::engine::ReadinessLevel::CeremonialReady => {
            info!("⚔️  System is CEREMONIALLY READY - Can answer ultimate questions");
        }
        crate::engine::ReadinessLevel::HighConfidence => {
            info!("✅ System has HIGH CONFIDENCE - Can answer complex questions");
        }
        crate::engine::ReadinessLevel::Moderate => {
            info!("⚠️  System has MODERATE readiness - Can answer standard questions");
        }
        crate::engine::ReadinessLevel::Insufficient => {
            error!("❌ System readiness INSUFFICIENT - Complete preparation first");
            if ceremonial_mode {
                return Err(anyhow::anyhow!("Cannot activate ceremonial mode with insufficient readiness"));
            }
        }
    }

    // Start the API server
    info!("🌐 Starting API server on port {}", port);
    let api_server = ApiServer::new(engine.clone(), config.clone());
    let server_handle = tokio::spawn(async move {
        if let Err(e) = api_server.start(port).await {
            error!("API server error: {}", e);
        }
    });

    // Start background services
    info!("🔄 Starting background services...");
    
    // ATP regeneration service
    let atp_engine = engine.clone();
    let atp_handle = tokio::spawn(async move {
        atp_engine.start_atp_regeneration().await;
    });

    // Dreaming module (if enabled)
    if config.metabolism.dreaming.enabled {
        info!("💭 Starting dreaming module...");
        let dream_engine = engine.clone();
        let dream_handle = tokio::spawn(async move {
            dream_engine.start_dreaming_cycles().await;
        });
        
        // Don't await dream_handle as it runs indefinitely
        tokio::spawn(dream_handle);
    }

    // Health check service
    let health_engine = engine.clone();
    let health_handle = tokio::spawn(async move {
        health_engine.start_health_monitoring().await;
    });

    info!("🚀 Honjo Masamune Truth Engine is now running");
    info!("📊 Monitoring available at http://localhost:3000 (Grafana)");
    info!("🔍 Tracing available at http://localhost:16686 (Jaeger)");
    
    if ceremonial_mode {
        info!("⚔️  CEREMONIAL MODE ACTIVE");
        info!("⚔️  Maximum {} queries per year", config.ceremonial.restrictions.max_queries_per_year);
        info!("⚔️  Each query permanently closes discussion on the topic");
    }

    // Wait for shutdown signal
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("🛑 Received shutdown signal");
        }
        _ = server_handle => {
            error!("API server terminated unexpectedly");
        }
        _ = atp_handle => {
            error!("ATP regeneration service terminated unexpectedly");
        }
        _ = health_handle => {
            error!("Health monitoring service terminated unexpectedly");
        }
    }

    info!("🗾 Honjo Masamune Truth Engine shutting down...");
    
    // Graceful shutdown
    engine.shutdown().await?;
    
    if ceremonial_mode {
        info!("⚔️  Ceremonial session ended - The sword returns to its sheath");
    }
    
    info!("👋 Honjo Masamune Truth Engine stopped");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_initialization() {
        // Test basic engine initialization
        let config = HonjoMasamuneConfig::default();
        let engine = HonjoMasamuneEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_ceremonial_mode_restrictions() {
        let mut config = HonjoMasamuneConfig::default();
        config.system.ceremonial_mode = true;
        
        let engine = HonjoMasamuneEngine::new(config).await.unwrap();
        let readiness = engine.check_readiness().await.unwrap();
        
        // In ceremonial mode, readiness must be at least high confidence
        match readiness.level {
            crate::engine::ReadinessLevel::CeremonialReady | 
            crate::engine::ReadinessLevel::HighConfidence => {
                // OK
            }
            _ => {
                panic!("Ceremonial mode requires high readiness");
            }
        }
    }
} 