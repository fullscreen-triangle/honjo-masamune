use anyhow::Result;
use atp_manager::{AtpManager, AtpCosts};
use buhera_engine::BuheraEngine;
use clap::{Arg, Command};
use fuzzy_logic_core::FuzzyLogicEngine;
use std::sync::Arc;
use tokio::signal;
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod engine;
mod ceremonial;
mod monitoring;
mod repositories;
mod spectacular;
mod nicotine;
mod mzekezeke;
mod diggiden;
mod hatata;
mod zengeza;
mod diadochi;

use config::HonjoMasamuneConfig;
use engine::HonjoMasamuneEngine;
use ceremonial::CeremonialInterface;

/// Honjo Masamune Truth Engine
/// 
/// Named after the legendary Japanese sword that was so sharp it became blunt
/// after first use due to accumulation of human fat. This system permanently
/// closes discussion on topics it investigates.
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
        .about("Honjo Masamune Truth Engine - The sword that ends wonder")
        .arg(
            Arg::new("config")
                .short('c')
                .long("config")
                .value_name("FILE")
                .help("Configuration file path")
                .default_value("config/honjo-masamune.yaml"),
        )
        .arg(
            Arg::new("ceremonial")
                .long("ceremonial")
                .help("Enable ceremonial mode (requires elite authorization)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("prepare")
                .long("prepare")
                .help("Run preparation phase")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("query")
                .short('q')
                .long("query")
                .value_name("QUERY")
                .help("Execute a single query"),
        )
        .arg(
            Arg::new("program")
                .short('p')
                .long("program")
                .value_name("FILE")
                .help("Execute a Buhera program file"),
        )
        .get_matches();

    // Load configuration
    let config_path = matches.get_one::<String>("config").unwrap();
    let mut config = HonjoMasamuneConfig::load(config_path).await?;

    // Override ceremonial mode if specified
    if matches.get_flag("ceremonial") {
        config.system.ceremonial_mode = true;
    }

    // Validate configuration for ceremonial mode
    if config.system.ceremonial_mode {
        config.validate_ceremonial()?;
        display_ceremonial_warning().await;
    }

    info!("🗡️  Initializing Honjo Masamune Truth Engine");
    info!("⚔️  Version: {}", config.system.version);
    info!("🏛️  Ceremonial Mode: {}", config.system.ceremonial_mode);

    // Initialize core systems
    let atp_manager = Arc::new(AtpManager::new(
        config.system.atp.initial_pool,
        config.system.atp.max_pool,
        config.system.atp.emergency_reserve,
        config.system.atp.regeneration_rate,
        AtpCosts {
            basic_query: config.system.atp.costs.basic_query,
            fuzzy_operation: config.system.atp.costs.fuzzy_operation,
            uncertainty_processing: config.system.atp.costs.uncertainty_processing,
            repository_call: config.system.atp.costs.repository_call,
            synthesis_operation: config.system.atp.costs.synthesis_operation,
            verification_step: config.system.atp.costs.verification_step,
            dreaming_cycle: config.system.atp.costs.dreaming_cycle,
            gray_area_processing: 75,
            truth_spectrum_analysis: 300,
            lactic_fermentation: 10,
        },
    ));

    let fuzzy_engine = FuzzyLogicEngine::new(
        config.fuzzy_logic.truth_thresholds,
        config.fuzzy_logic.operators,
        config.fuzzy_logic.gray_areas,
    );

    let buhera_engine = Arc::new(BuheraEngine::new(atp_manager.clone()));

    // Initialize main engine
    let engine = Arc::new(HonjoMasamuneEngine::new(
        config.clone(),
        atp_manager,
        Arc::new(fuzzy_engine),
        buhera_engine,
    ).await?);

    info!("⚡ ATP Pool initialized: {} units", engine.atp_status().current);
    info!("🧠 Fuzzy logic engine ready");
    info!("📜 Buhera logical programming engine ready");

    // Handle different execution modes
    if matches.get_flag("prepare") {
        info!("🔄 Starting preparation phase...");
        run_preparation_phase(&engine).await?;
    } else if let Some(query) = matches.get_one::<String>("query") {
        info!("❓ Executing single query...");
        execute_single_query(&engine, query).await?;
    } else if let Some(program_path) = matches.get_one::<String>("program") {
        info!("📋 Executing Buhera program...");
        execute_program(&engine, program_path).await?;
    } else {
        info!("🎯 Starting interactive mode...");
        run_interactive_mode(&engine).await?;
    }

    Ok(())
}

/// Display ceremonial mode warning
async fn display_ceremonial_warning() {
    warn!("⚠️  CEREMONIAL MODE ACTIVATED ⚠️");
    warn!("🗡️  You are about to draw the legendary Honjo Masamune");
    warn!("💀 This action will permanently close discussion on investigated topics");
    warn!("🌟 Wonder will be eliminated for the subjects you query");
    warn!("🏛️  Ensure you have the moral authority to end human discourse");
    warn!("⏳ Cooling period will be enforced after each use");
    
    // In a real implementation, this would require multiple authorizations
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    
    warn!("🔥 Ceremonial mode confirmed. The sword is drawn.");
}

/// Run the preparation phase
async fn run_preparation_phase(engine: &Arc<HonjoMasamuneEngine>) -> Result<()> {
    info!("📚 Beginning corpus ingestion...");
    
    // This would integrate with the preparation engine
    // For now, we'll simulate the process
    let preparation_steps = [
        "Validating corpus integrity",
        "Extracting knowledge patterns", 
        "Building truth foundations",
        "Synthesizing fuzzy models",
        "Establishing confidence baselines",
        "Preparing repository interfaces",
        "Calibrating ATP metabolism",
        "Initializing dreaming cycles",
    ];

    for (i, step) in preparation_steps.iter().enumerate() {
        info!("📋 Step {}/{}: {}", i + 1, preparation_steps.len(), step);
        
        // Simulate processing time
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
        
        // Check ATP levels
        let atp_status = engine.atp_status();
        if atp_status.is_critical {
            warn!("⚡ ATP levels critical, initiating emergency production");
            // In real implementation, this would trigger ATP production
        }
    }

    info!("✅ Preparation phase complete. System ready for queries.");
    Ok(())
}

/// Execute a single query
async fn execute_single_query(engine: &Arc<HonjoMasamuneEngine>, query: &str) -> Result<()> {
    info!("🔍 Processing query: {}", query);
    
    // Parse and execute the query
    let result = engine.process_natural_language_query(query).await?;
    
    info!("📊 Query Results:");
    info!("   Confidence: {:.3}", result.confidence.value());
    info!("   Truth Membership: {:?}", result.membership(&engine.config.fuzzy_logic.truth_thresholds));
    info!("   Gray Areas: {}", result.gray_areas.len());
    info!("   ATP Cost: {} units", result.atp_cost);
    
    if result.confidence.value() >= 0.95 {
        warn!("🎯 CEREMONIAL CERTAINTY ACHIEVED");
        warn!("💀 This topic is now permanently closed to further discussion");
    } else if result.is_gray_area([0.4, 0.7]) {
        warn!("🌫️  Gray area detected - human judgment may be required");
    }
    
    Ok(())
}

/// Execute a Buhera program file
async fn execute_program(engine: &Arc<HonjoMasamuneEngine>, program_path: &str) -> Result<()> {
    info!("📜 Loading Buhera program: {}", program_path);
    
    let program_source = tokio::fs::read_to_string(program_path).await?;
    let results = engine.execute_buhera_program(&program_source).await?;
    
    info!("📊 Program execution complete:");
    info!("   Queries executed: {}", results.len());
    
    for (i, result) in results.iter().enumerate() {
        info!("   Query {}: confidence {:.3}, {} solutions", 
              i + 1, result.confidence.value(), result.value.len());
    }
    
    Ok(())
}

/// Run interactive mode
async fn run_interactive_mode(engine: &Arc<HonjoMasamuneEngine>) -> Result<()> {
    info!("🎮 Interactive mode started. Type 'help' for commands.");
    
    let ceremonial = CeremonialInterface::new(engine.clone());
    
    // Set up graceful shutdown
    let shutdown_signal = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
    };
    
    tokio::select! {
        result = ceremonial.run() => {
            if let Err(e) = result {
                error!("Interactive mode error: {}", e);
            }
        }
        _ = shutdown_signal => {
            info!("🛑 Shutdown signal received");
        }
    }
    
    info!("🗡️  Sheathing the Honjo Masamune. Wonder is preserved for another day.");
    Ok(())
}

/// Custom query result for the main interface
#[derive(Debug)]
pub struct QueryResult {
    pub confidence: fuzzy_logic_core::FuzzyTruth,
    pub gray_areas: Vec<String>,
    pub atp_cost: u64,
}

impl QueryResult {
    pub fn membership(&self, thresholds: &fuzzy_logic_core::TruthThresholds) -> fuzzy_logic_core::TruthMembership {
        self.confidence.to_membership(thresholds)
    }
    
    pub fn is_gray_area(&self, range: [f64; 2]) -> bool {
        self.confidence.is_gray_area(range)
    }
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