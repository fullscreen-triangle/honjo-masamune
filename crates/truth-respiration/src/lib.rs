//! Truth Respiration System
//! 
//! Implements biological metabolism for truth processing using cellular
//! respiration metaphors including glycolysis, Krebs cycle, and electron transport.

pub mod glycolysis;
pub mod krebs_cycle;
pub mod electron_transport;

pub use glycolysis::*;
pub use krebs_cycle::*;
pub use electron_transport::*; 