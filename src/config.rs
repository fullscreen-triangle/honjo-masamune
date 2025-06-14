use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonjoMasamuneConfig {
    pub system: SystemConfig,
    pub fuzzy_logic: FuzzyLogicConfig,
    pub metabolism: MetabolismConfig,
    pub repositories: RepositoriesConfig,
    pub databases: DatabasesConfig,
    pub preparation: PreparationConfig,
    pub dreaming: DreamingConfig,
    pub monitoring: MonitoringConfig,
    pub security: SecurityConfig,
    pub ceremonial: CeremonialConfig,
    pub development: DevelopmentConfig,
    pub bloodhound: BloodhoundConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub name: String,
    pub version: String,
    pub ceremonial_mode: bool,
    pub engine: EngineConfig,
    pub atp: AtpConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub max_concurrent_queries: u32,
    pub query_timeout_hours: u64,
    pub preparation_timeout_days: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpConfig {
    pub initial_pool: u64,
    pub max_pool: u64,
    pub regeneration_rate: u64,
    pub emergency_reserve: u64,
    pub low_threshold: u64,
    pub costs: AtpCosts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpCosts {
    pub basic_query: u64,
    pub fuzzy_operation: u64,
    pub uncertainty_processing: u64,
    pub repository_call: u64,
    pub synthesis_operation: u64,
    pub verification_step: u64,
    pub dreaming_cycle: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyLogicConfig {
    pub truth_thresholds: TruthThresholds,
    pub operators: FuzzyOperators,
    pub hedges: LinguisticHedges,
    pub gray_areas: GrayAreaConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthThresholds {
    pub certain: f64,
    pub probable: f64,
    pub possible: f64,
    pub unlikely: f64,
    pub improbable: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyOperators {
    pub t_norm: String,
    pub t_conorm: String,
    pub implication: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticHedges {
    pub very_concentration: f64,
    pub somewhat_dilation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrayAreaConfig {
    pub detection_range: [f64; 2],
    pub processing_overhead: f64,
    pub human_judgment_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolismConfig {
    pub respiration: RespirationConfig,
    pub lactic_fermentation: LacticFermentationConfig,
    pub dreaming: MetabolismDreamingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespirationConfig {
    pub glycolysis: GlycolysisConfig,
    pub krebs_cycle: KrebsCycleConfig,
    pub electron_transport: ElectronTransportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlycolysisConfig {
    pub atp_investment: u32,
    pub atp_yield: u32,
    pub net_gain: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrebsCycleConfig {
    pub atp_per_cycle: u32,
    pub nadh_per_cycle: u32,
    pub fadh2_per_cycle: u32,
    pub cycles_per_query: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectronTransportConfig {
    pub atp_from_nadh: u32,
    pub atp_from_fadh2: u32,
    pub total_atp_yield: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LacticFermentationConfig {
    pub atp_yield: u32,
    pub lactate_accumulation_rate: f64,
    pub recovery_time_hours: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetabolismDreamingConfig {
    pub enabled: bool,
    pub cycle_interval_hours: u64,
    pub lactate_processing_rate: f64,
    pub script_generation_rate: f64,
    pub max_dream_cycles: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoriesConfig {
    pub interface: RepositoryInterfaceConfig,
    #[serde(flatten)]
    pub repositories: HashMap<String, RepositoryConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryInterfaceConfig {
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub circuit_breaker_threshold: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepositoryConfig {
    pub url: String,
    pub capabilities: Vec<String>,
    pub confidence_model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabasesConfig {
    pub postgresql: PostgresConfig,
    pub neo4j: Neo4jConfig,
    pub clickhouse: ClickhouseConfig,
    pub redis: RedisConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostgresConfig {
    pub host: String,
    pub port: u16,
    pub database: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClickhouseConfig {
    pub url: String,
    pub database: String,
    pub username: String,
    pub password: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub password: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparationConfig {
    pub corpus: CorpusConfig,
    pub stages: PreparationStagesConfig,
    pub readiness: ReadinessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusConfig {
    pub minimum_documents: u64,
    pub minimum_size_gb: u64,
    pub supported_formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparationStagesConfig {
    pub ingestion: IngestionConfig,
    pub model_synthesis: ModelSynthesisConfig,
    pub truth_foundation: TruthFoundationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestionConfig {
    pub workers: u32,
    pub batch_size: u32,
    pub validation_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSynthesisConfig {
    pub workers: u32,
    pub memory_limit_gb: u32,
    pub gpu_acceleration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruthFoundationConfig {
    pub verification_threshold: f64,
    pub cross_reference_minimum: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadinessConfig {
    pub assessment_interval_hours: u64,
    pub minimum_confidence: f64,
    pub ceremonial_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamingConfig {
    pub lactate_processing: LactateProcessingConfig,
    pub script_generation: ScriptGenerationConfig,
    pub cycles: DreamCyclesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LactateProcessingConfig {
    pub extraction_rate: f64,
    pub pattern_synthesis_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptGenerationConfig {
    pub creativity_factor: f64,
    pub novelty_threshold: f64,
    pub validation_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamCyclesConfig {
    pub rem_equivalent_duration: u64,
    pub deep_processing_duration: u64,
    pub light_processing_duration: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub metrics: MetricsConfig,
    pub logging: LoggingConfig,
    pub tracing: TracingConfig,
    pub health: HealthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub prometheus_endpoint: String,
    pub collection_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,
    pub file_rotation: String,
    pub max_file_size_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    pub enabled: bool,
    pub jaeger_endpoint: String,
    pub sample_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    pub endpoint: String,
    pub interval_seconds: u64,
    pub timeout_seconds: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub authentication: AuthenticationConfig,
    pub authorization: AuthorizationConfig,
    pub encryption: EncryptionConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    pub required: bool,
    pub method: String,
    pub certificate_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationConfig {
    pub elite_organizations_only: bool,
    pub minimum_clearance_level: String,
    pub financial_verification_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    pub at_rest: bool,
    pub in_transit: bool,
    pub algorithm: String,
    pub key_rotation_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonialConfig {
    pub activation: CeremonialActivationConfig,
    pub restrictions: CeremonialRestrictionsConfig,
    pub consequences: CeremonialConsequencesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonialActivationConfig {
    pub requires_multiple_authorization: bool,
    pub minimum_authorizers: u32,
    pub cooling_period_days: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonialRestrictionsConfig {
    pub max_queries_per_year: u32,
    pub mandatory_review_period_days: u32,
    pub public_disclosure_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CeremonialConsequencesConfig {
    pub topic_closure_permanent: bool,
    pub discourse_termination: bool,
    pub wonder_elimination_acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentConfig {
    pub debug_mode: bool,
    pub mock_repositories: bool,
    pub fast_preparation: bool,
    pub skip_verification: bool,
    pub allow_low_confidence: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloodhoundConfig {
    pub local_first: LocalFirstConfig,
    pub federated_learning: FederatedLearningConfig,
    pub zero_config: ZeroConfigConfig,
    pub data_sovereignty: DataSovereigntyConfig,
    pub conversational_ai: ConversationalAiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalFirstConfig {
    pub enabled: bool,
    pub auto_resource_detection: bool,
    pub local_processing_priority: bool,
    pub data_never_leaves_source: bool,
    pub pattern_sharing_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedLearningConfig {
    pub enabled: bool,
    pub privacy_preserving: bool,
    pub consensus_threshold: f64,
    pub minimum_participants: u32,
    pub cross_validation_required: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZeroConfigConfig {
    pub auto_optimization: bool,
    pub self_healing: bool,
    pub intelligent_resource_allocation: bool,
    pub minimal_user_intervention: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSovereigntyConfig {
    pub enforce_local_processing: bool,
    pub p2p_only_when_necessary: bool,
    pub automatic_data_health_checks: bool,
    pub direct_lab_to_lab_transfer: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationalAiConfig {
    pub natural_language_interface: bool,
    pub automatic_assumption_validation: bool,
    pub intelligent_test_selection: bool,
    pub plain_language_explanations: bool,
}

impl HonjoMasamuneConfig {
    /// Load configuration from a YAML file
    pub async fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = tokio::fs::read_to_string(path).await?;
        let config: HonjoMasamuneConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a YAML file
    pub async fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_yaml::to_string(self)?;
        tokio::fs::write(path, content).await?;
        Ok(())
    }

    /// Get database URL for PostgreSQL
    pub fn postgres_url(&self) -> String {
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            self.databases.postgresql.username,
            self.databases.postgresql.password,
            self.databases.postgresql.host,
            self.databases.postgresql.port,
            self.databases.postgresql.database
        )
    }

    /// Check if system is in ceremonial mode
    pub fn is_ceremonial(&self) -> bool {
        self.system.ceremonial_mode
    }

    /// Validate configuration for ceremonial mode
    pub fn validate_ceremonial(&self) -> Result<()> {
        if self.system.ceremonial_mode {
            if !self.security.authentication.required {
                return Err(anyhow::anyhow!("Authentication required for ceremonial mode"));
            }
            
            if !self.security.authorization.elite_organizations_only {
                return Err(anyhow::anyhow!("Elite organization verification required for ceremonial mode"));
            }
            
            if self.development.debug_mode {
                return Err(anyhow::anyhow!("Debug mode not allowed in ceremonial mode"));
            }
        }
        Ok(())
    }
}

impl Default for HonjoMasamuneConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig {
                name: "Honjo Masamune".to_string(),
                version: "0.1.0".to_string(),
                ceremonial_mode: false,
                engine: EngineConfig {
                    max_concurrent_queries: 1,
                    query_timeout_hours: 168,
                    preparation_timeout_days: 120,
                },
                atp: AtpConfig {
                    initial_pool: 1000000,
                    max_pool: 10000000,
                    regeneration_rate: 1000,
                    emergency_reserve: 100000,
                    low_threshold: 50000,
                    costs: AtpCosts {
                        basic_query: 100,
                        fuzzy_operation: 50,
                        uncertainty_processing: 25,
                        repository_call: 100,
                        synthesis_operation: 200,
                        verification_step: 150,
                        dreaming_cycle: 500,
                    },
                },
            },
            fuzzy_logic: FuzzyLogicConfig {
                truth_thresholds: TruthThresholds {
                    certain: 0.95,
                    probable: 0.75,
                    possible: 0.50,
                    unlikely: 0.25,
                    improbable: 0.05,
                },
                operators: FuzzyOperators {
                    t_norm: "minimum".to_string(),
                    t_conorm: "maximum".to_string(),
                    implication: "kleene_dienes".to_string(),
                },
                hedges: LinguisticHedges {
                    very_concentration: 2.0,
                    somewhat_dilation: 0.5,
                },
                gray_areas: GrayAreaConfig {
                    detection_range: [0.4, 0.7],
                    processing_overhead: 1.5,
                    human_judgment_threshold: 0.5,
                },
            },
            metabolism: MetabolismConfig {
                respiration: RespirationConfig {
                    glycolysis: GlycolysisConfig {
                        atp_investment: 2,
                        atp_yield: 4,
                        net_gain: 2,
                    },
                    krebs_cycle: KrebsCycleConfig {
                        atp_per_cycle: 2,
                        nadh_per_cycle: 3,
                        fadh2_per_cycle: 1,
                        cycles_per_query: 8,
                    },
                    electron_transport: ElectronTransportConfig {
                        atp_from_nadh: 3,
                        atp_from_fadh2: 2,
                        total_atp_yield: 38,
                    },
                },
                lactic_fermentation: LacticFermentationConfig {
                    atp_yield: 2,
                    lactate_accumulation_rate: 0.1,
                    recovery_time_hours: 4,
                },
                dreaming: MetabolismDreamingConfig {
                    enabled: true,
                    cycle_interval_hours: 1,
                    lactate_processing_rate: 0.2,
                    script_generation_rate: 0.05,
                    max_dream_cycles: 24,
                },
            },
            repositories: RepositoriesConfig {
                interface: RepositoryInterfaceConfig {
                    timeout_seconds: 300,
                    retry_attempts: 3,
                    circuit_breaker_threshold: 5,
                },
                repositories: HashMap::new(),
            },
            databases: DatabasesConfig {
                postgresql: PostgresConfig {
                    host: "localhost".to_string(),
                    port: 5432,
                    database: "honjo_masamune".to_string(),
                    username: "honjo".to_string(),
                    password: "ceremonial_sword".to_string(),
                    max_connections: 20,
                },
                neo4j: Neo4jConfig {
                    uri: "bolt://localhost:7687".to_string(),
                    username: "neo4j".to_string(),
                    password: "truth_synthesis".to_string(),
                    max_connections: 10,
                },
                clickhouse: ClickhouseConfig {
                    url: "http://localhost:8123".to_string(),
                    database: "honjo_analytics".to_string(),
                    username: "honjo".to_string(),
                    password: "truth_analytics".to_string(),
                },
                redis: RedisConfig {
                    url: "redis://localhost:6379".to_string(),
                    password: "truth_cache".to_string(),
                    max_connections: 50,
                },
            },
            preparation: PreparationConfig {
                corpus: CorpusConfig {
                    minimum_documents: 100000,
                    minimum_size_gb: 1000,
                    supported_formats: vec![
                        "pdf".to_string(),
                        "txt".to_string(),
                        "docx".to_string(),
                        "html".to_string(),
                        "json".to_string(),
                        "xml".to_string(),
                    ],
                },
                stages: PreparationStagesConfig {
                    ingestion: IngestionConfig {
                        workers: 8,
                        batch_size: 1000,
                        validation_level: "strict".to_string(),
                    },
                    model_synthesis: ModelSynthesisConfig {
                        workers: 4,
                        memory_limit_gb: 32,
                        gpu_acceleration: true,
                    },
                    truth_foundation: TruthFoundationConfig {
                        verification_threshold: 0.9,
                        cross_reference_minimum: 3,
                    },
                },
                readiness: ReadinessConfig {
                    assessment_interval_hours: 24,
                    minimum_confidence: 0.7,
                    ceremonial_threshold: 0.95,
                },
            },
            dreaming: DreamingConfig {
                lactate_processing: LactateProcessingConfig {
                    extraction_rate: 0.2,
                    pattern_synthesis_threshold: 0.3,
                },
                script_generation: ScriptGenerationConfig {
                    creativity_factor: 0.1,
                    novelty_threshold: 0.8,
                    validation_required: true,
                },
                cycles: DreamCyclesConfig {
                    rem_equivalent_duration: 90,
                    deep_processing_duration: 180,
                    light_processing_duration: 30,
                },
            },
            monitoring: MonitoringConfig {
                metrics: MetricsConfig {
                    enabled: true,
                    prometheus_endpoint: "http://localhost:9090".to_string(),
                    collection_interval: 30,
                },
                logging: LoggingConfig {
                    level: "info".to_string(),
                    format: "json".to_string(),
                    file_rotation: "daily".to_string(),
                    max_file_size_mb: 100,
                },
                tracing: TracingConfig {
                    enabled: true,
                    jaeger_endpoint: "http://localhost:14268".to_string(),
                    sample_rate: 0.1,
                },
                health: HealthConfig {
                    endpoint: "/health".to_string(),
                    interval_seconds: 30,
                    timeout_seconds: 10,
                },
            },
            security: SecurityConfig {
                authentication: AuthenticationConfig {
                    required: false,
                    method: "certificate".to_string(),
                    certificate_path: "/app/certs".to_string(),
                },
                authorization: AuthorizationConfig {
                    elite_organizations_only: false,
                    minimum_clearance_level: "cosmic".to_string(),
                    financial_verification_required: false,
                },
                encryption: EncryptionConfig {
                    at_rest: true,
                    in_transit: true,
                    algorithm: "AES-256-GCM".to_string(),
                    key_rotation_days: 30,
                },
            },
            ceremonial: CeremonialConfig {
                activation: CeremonialActivationConfig {
                    requires_multiple_authorization: true,
                    minimum_authorizers: 3,
                    cooling_period_days: 30,
                },
                restrictions: CeremonialRestrictionsConfig {
                    max_queries_per_year: 12,
                    mandatory_review_period_days: 90,
                    public_disclosure_required: false,
                },
                consequences: CeremonialConsequencesConfig {
                    topic_closure_permanent: true,
                    discourse_termination: true,
                    wonder_elimination_acknowledged: true,
                },
            },
            development: DevelopmentConfig {
                debug_mode: true,
                mock_repositories: false,
                fast_preparation: false,
                skip_verification: false,
                allow_low_confidence: true,
            },
            bloodhound: BloodhoundConfig {
                local_first: LocalFirstConfig {
                    enabled: true,
                    auto_resource_detection: true,
                    local_processing_priority: true,
                    data_never_leaves_source: true,
                    pattern_sharing_only: true,
                },
                federated_learning: FederatedLearningConfig {
                    enabled: true,
                    privacy_preserving: true,
                    consensus_threshold: 0.7,
                    minimum_participants: 5,
                    cross_validation_required: true,
                },
                zero_config: ZeroConfigConfig {
                    auto_optimization: true,
                    self_healing: true,
                    intelligent_resource_allocation: true,
                    minimal_user_intervention: true,
                },
                data_sovereignty: DataSovereigntyConfig {
                    enforce_local_processing: true,
                    p2p_only_when_necessary: true,
                    automatic_data_health_checks: true,
                    direct_lab_to_lab_transfer: true,
                },
                conversational_ai: ConversationalAiConfig {
                    natural_language_interface: true,
                    automatic_assumption_validation: true,
                    intelligent_test_selection: true,
                    plain_language_explanations: true,
                },
            },
        }
    }
} 