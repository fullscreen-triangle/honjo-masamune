//! ATP Manager for Honjo Masamune Truth Engine
//! 
//! This crate implements the biological metabolism system using ATP (Adenosine Triphosphate)
//! as the energy currency. It models cellular respiration processes including glycolysis,
//! Krebs cycle, electron transport chain, and lactic acid fermentation for incomplete
//! processing scenarios.

use anyhow::Result;
use chrono::{DateTime, Utc};
use fuzzy_logic_core::FuzzyTruth;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use std::sync::Arc;
use tokio::sync::broadcast;
use uuid::Uuid;

pub mod glycolysis;
pub mod krebs_cycle;
pub mod electron_transport;
pub mod lactic_fermentation;
pub mod metabolism;

pub use glycolysis::*;
pub use krebs_cycle::*;
pub use electron_transport::*;
pub use lactic_fermentation::*;
pub use metabolism::*;

/// ATP pool representing the energy available for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpPool {
    pub current: u64,
    pub maximum: u64,
    pub reserved: u64,
    pub emergency_reserve: u64,
    pub regeneration_rate: u64,
    pub last_regeneration: DateTime<Utc>,
}

impl AtpPool {
    /// Create a new ATP pool
    pub fn new(initial: u64, maximum: u64, emergency_reserve: u64, regeneration_rate: u64) -> Self {
        Self {
            current: initial,
            maximum,
            reserved: 0,
            emergency_reserve,
            regeneration_rate,
            last_regeneration: Utc::now(),
        }
    }

    /// Check if sufficient ATP is available
    pub fn has_sufficient(&self, required: u64) -> bool {
        self.available() >= required
    }

    /// Get available ATP (current - reserved - emergency_reserve)
    pub fn available(&self) -> u64 {
        self.current.saturating_sub(self.reserved + self.emergency_reserve)
    }

    /// Reserve ATP for an operation
    pub fn reserve(&mut self, amount: u64) -> Result<AtpReservation> {
        if !self.has_sufficient(amount) {
            return Err(anyhow::anyhow!(
                "Insufficient ATP: required {}, available {}",
                amount,
                self.available()
            ));
        }

        self.reserved += amount;
        Ok(AtpReservation {
            id: Uuid::new_v4(),
            amount,
            timestamp: Utc::now(),
        })
    }

    /// Consume reserved ATP
    pub fn consume(&mut self, reservation: AtpReservation) -> Result<()> {
        if self.reserved < reservation.amount {
            return Err(anyhow::anyhow!("Invalid reservation: insufficient reserved ATP"));
        }

        self.reserved -= reservation.amount;
        self.current -= reservation.amount;
        Ok(())
    }

    /// Release reserved ATP without consuming
    pub fn release(&mut self, reservation: AtpReservation) {
        self.reserved = self.reserved.saturating_sub(reservation.amount);
    }

    /// Add ATP to the pool (from regeneration or production)
    pub fn add(&mut self, amount: u64) {
        self.current = (self.current + amount).min(self.maximum);
    }

    /// Regenerate ATP based on time elapsed
    pub fn regenerate(&mut self) {
        let now = Utc::now();
        let elapsed = now.timestamp() - self.last_regeneration.timestamp();
        
        if elapsed > 0 {
            let regenerated = (elapsed as u64) * self.regeneration_rate / 60; // per minute
            self.add(regenerated);
            self.last_regeneration = now;
        }
    }

    /// Check if pool is in critical state
    pub fn is_critical(&self) -> bool {
        self.available() < self.emergency_reserve
    }

    /// Get pool utilization percentage
    pub fn utilization(&self) -> f64 {
        (self.current as f64 / self.maximum as f64) * 100.0
    }
}

/// ATP reservation for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpReservation {
    pub id: Uuid,
    pub amount: u64,
    pub timestamp: DateTime<Utc>,
}

/// ATP cost configuration for different operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpCosts {
    pub basic_query: u64,
    pub fuzzy_operation: u64,
    pub uncertainty_processing: u64,
    pub repository_call: u64,
    pub synthesis_operation: u64,
    pub verification_step: u64,
    pub dreaming_cycle: u64,
    pub gray_area_processing: u64,
    pub truth_spectrum_analysis: u64,
    pub lactic_fermentation: u64,
}

impl Default for AtpCosts {
    fn default() -> Self {
        Self {
            basic_query: 100,
            fuzzy_operation: 50,
            uncertainty_processing: 25,
            repository_call: 100,
            synthesis_operation: 200,
            verification_step: 150,
            dreaming_cycle: 500,
            gray_area_processing: 75,
            truth_spectrum_analysis: 300,
            lactic_fermentation: 10,
        }
    }
}

/// ATP transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpTransaction {
    pub id: Uuid,
    pub operation_type: String,
    pub amount: u64,
    pub transaction_type: TransactionType,
    pub confidence_impact: Option<FuzzyTruth>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransactionType {
    Consumption,
    Production,
    Reservation,
    Release,
}

/// ATP manager that orchestrates the biological metabolism system
#[derive(Debug)]
pub struct AtpManager {
    pool: Arc<RwLock<AtpPool>>,
    costs: AtpCosts,
    transactions: Arc<Mutex<Vec<AtpTransaction>>>,
    metabolism: Arc<Mutex<MetabolismEngine>>,
    event_sender: broadcast::Sender<AtpEvent>,
    _event_receiver: broadcast::Receiver<AtpEvent>,
}

impl AtpManager {
    /// Create a new ATP manager
    pub fn new(
        initial_pool: u64,
        max_pool: u64,
        emergency_reserve: u64,
        regeneration_rate: u64,
        costs: AtpCosts,
    ) -> Self {
        let pool = Arc::new(RwLock::new(AtpPool::new(
            initial_pool,
            max_pool,
            emergency_reserve,
            regeneration_rate,
        )));

        let metabolism = Arc::new(Mutex::new(MetabolismEngine::new()));
        let (event_sender, event_receiver) = broadcast::channel(1000);

        Self {
            pool,
            costs,
            transactions: Arc::new(Mutex::new(Vec::new())),
            metabolism,
            event_sender,
            _event_receiver: event_receiver,
        }
    }

    /// Get current ATP status
    pub fn status(&self) -> AtpStatus {
        let pool = self.pool.read();
        AtpStatus {
            current: pool.current,
            maximum: pool.maximum,
            available: pool.available(),
            reserved: pool.reserved,
            emergency_reserve: pool.emergency_reserve,
            utilization: pool.utilization(),
            is_critical: pool.is_critical(),
            regeneration_rate: pool.regeneration_rate,
        }
    }

    /// Reserve ATP for an operation
    pub async fn reserve_atp(&self, operation: &str, amount: u64) -> Result<AtpReservation> {
        let mut pool = self.pool.write();
        pool.regenerate();

        let reservation = pool.reserve(amount)?;

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: operation.to_string(),
            amount,
            transaction_type: TransactionType::Reservation,
            confidence_impact: None,
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::Reserved {
            operation: operation.to_string(),
            amount,
            reservation_id: reservation.id,
        });

        Ok(reservation)
    }

    /// Consume reserved ATP
    pub async fn consume_atp(&self, reservation: AtpReservation, operation: &str) -> Result<()> {
        let reservation_id = reservation.id;
        let amount = reservation.amount;
        
        let mut pool = self.pool.write();
        pool.consume(reservation)?;

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: operation.to_string(),
            amount,
            transaction_type: TransactionType::Consumption,
            confidence_impact: None,
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::Consumed {
            operation: operation.to_string(),
            amount,
            reservation_id,
        });

        Ok(())
    }

    /// Release reserved ATP without consuming
    pub async fn release_atp(&self, reservation: AtpReservation, operation: &str) {
        let reservation_id = reservation.id;
        let amount = reservation.amount;
        
        let mut pool = self.pool.write();
        pool.release(reservation);

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: operation.to_string(),
            amount,
            transaction_type: TransactionType::Release,
            confidence_impact: None,
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::Released {
            operation: operation.to_string(),
            amount,
            reservation_id,
        });
    }

    /// Produce ATP through cellular respiration
    pub async fn produce_atp(&self, glucose_units: u64) -> Result<u64> {
        let metabolism = self.metabolism.lock();
        let produced = metabolism.cellular_respiration(glucose_units).await?;

        let mut pool = self.pool.write();
        pool.add(produced);

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: "cellular_respiration".to_string(),
            amount: produced,
            transaction_type: TransactionType::Production,
            confidence_impact: None,
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::Produced {
            amount: produced,
            source: "cellular_respiration".to_string(),
        });

        Ok(produced)
    }

    /// Handle incomplete processing through lactic fermentation
    pub async fn lactic_fermentation(&self, incomplete_data: u64) -> Result<(u64, u64)> {
        let metabolism = self.metabolism.lock();
        let (atp_produced, lactate_produced) = metabolism.lactic_fermentation(incomplete_data).await?;

        let mut pool = self.pool.write();
        pool.add(atp_produced);

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: "lactic_fermentation".to_string(),
            amount: atp_produced,
            transaction_type: TransactionType::Production,
            confidence_impact: Some(FuzzyTruth::new_unchecked(0.6)), // Lower confidence
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::LacticFermentation {
            atp_produced,
            lactate_produced,
        });

        Ok((atp_produced, lactate_produced))
    }

    /// Get ATP cost for an operation
    pub fn get_cost(&self, operation: &str) -> u64 {
        match operation {
            "basic_query" => self.costs.basic_query,
            "fuzzy_operation" => self.costs.fuzzy_operation,
            "uncertainty_processing" => self.costs.uncertainty_processing,
            "repository_call" => self.costs.repository_call,
            "synthesis_operation" => self.costs.synthesis_operation,
            "verification_step" => self.costs.verification_step,
            "dreaming_cycle" => self.costs.dreaming_cycle,
            "gray_area_processing" => self.costs.gray_area_processing,
            "truth_spectrum_analysis" => self.costs.truth_spectrum_analysis,
            "lactic_fermentation" => self.costs.lactic_fermentation,
            _ => self.costs.basic_query, // Default cost
        }
    }

    /// Get transaction history
    pub fn get_transactions(&self) -> Vec<AtpTransaction> {
        self.transactions.lock().clone()
    }

    /// Get metabolism status
    pub fn metabolism_status(&self) -> MetabolismStatus {
        self.metabolism.lock().status()
    }

    /// Subscribe to ATP events
    pub fn subscribe(&self) -> broadcast::Receiver<AtpEvent> {
        self.event_sender.subscribe()
    }

    /// Emergency ATP production (should be used sparingly)
    pub async fn emergency_production(&self, amount: u64) -> Result<()> {
        let mut pool = self.pool.write();
        
        if !pool.is_critical() {
            return Err(anyhow::anyhow!("Emergency production only allowed in critical state"));
        }

        pool.add(amount);

        // Record transaction
        let transaction = AtpTransaction {
            id: Uuid::new_v4(),
            operation_type: "emergency_production".to_string(),
            amount,
            transaction_type: TransactionType::Production,
            confidence_impact: Some(FuzzyTruth::new_unchecked(0.3)), // Very low confidence
            timestamp: Utc::now(),
        };

        self.transactions.lock().push(transaction);

        // Send event
        let _ = self.event_sender.send(AtpEvent::EmergencyProduction { amount });

        Ok(())
    }
}

/// ATP status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtpStatus {
    pub current: u64,
    pub maximum: u64,
    pub available: u64,
    pub reserved: u64,
    pub emergency_reserve: u64,
    pub utilization: f64,
    pub is_critical: bool,
    pub regeneration_rate: u64,
}

/// ATP events for monitoring and logging
#[derive(Debug, Clone)]
pub enum AtpEvent {
    Reserved {
        operation: String,
        amount: u64,
        reservation_id: Uuid,
    },
    Consumed {
        operation: String,
        amount: u64,
        reservation_id: Uuid,
    },
    Released {
        operation: String,
        amount: u64,
        reservation_id: Uuid,
    },
    Produced {
        amount: u64,
        source: String,
    },
    LacticFermentation {
        atp_produced: u64,
        lactate_produced: u64,
    },
    EmergencyProduction {
        amount: u64,
    },
    CriticalState,
    RecoveredFromCritical,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_atp_pool_creation() {
        let pool = AtpPool::new(1000, 10000, 500, 100);
        assert_eq!(pool.current, 1000);
        assert_eq!(pool.maximum, 10000);
        assert_eq!(pool.emergency_reserve, 500);
        assert_eq!(pool.available(), 500); // 1000 - 0 - 500
    }

    #[test]
    fn test_atp_reservation() {
        let mut pool = AtpPool::new(1000, 10000, 100, 50);
        let reservation = pool.reserve(200).unwrap();
        
        assert_eq!(reservation.amount, 200);
        assert_eq!(pool.reserved, 200);
        assert_eq!(pool.available(), 700); // 1000 - 200 - 100
    }

    #[test]
    fn test_insufficient_atp() {
        let mut pool = AtpPool::new(100, 1000, 50, 10);
        let result = pool.reserve(100); // Only 50 available (100 - 50 emergency)
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_atp_manager_operations() {
        let manager = AtpManager::new(1000, 10000, 100, 50, AtpCosts::default());
        
        let reservation = manager.reserve_atp("test_operation", 200).await.unwrap();
        assert_eq!(reservation.amount, 200);
        
        let status = manager.status();
        assert_eq!(status.reserved, 200);
        
        manager.consume_atp(reservation, "test_operation").await.unwrap();
        
        let status = manager.status();
        assert_eq!(status.current, 800);
        assert_eq!(status.reserved, 0);
    }

    #[test]
    fn test_atp_costs() {
        let costs = AtpCosts::default();
        assert_eq!(costs.basic_query, 100);
        assert_eq!(costs.dreaming_cycle, 500);
    }
} 