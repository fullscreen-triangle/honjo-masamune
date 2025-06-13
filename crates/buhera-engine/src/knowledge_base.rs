//! Knowledge Base

use crate::{Fact, Rule, BuheraProgram};
use anyhow::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeBaseStats {
    pub fact_count: usize,
    pub rule_count: usize,
    pub program_count: usize,
}

pub struct KnowledgeBase {
    facts: Vec<Fact>,
    rules: Vec<Rule>,
    programs: Vec<BuheraProgram>,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            facts: Vec::new(),
            rules: Vec::new(),
            programs: Vec::new(),
        }
    }
    
    pub fn add_fact(&mut self, fact: Fact) -> Result<()> {
        self.facts.push(fact);
        Ok(())
    }
    
    pub fn add_rule(&mut self, rule: Rule) -> Result<()> {
        self.rules.push(rule);
        Ok(())
    }
    
    pub fn add_program(&mut self, program: BuheraProgram) {
        self.programs.push(program);
    }
    
    pub fn stats(&self) -> KnowledgeBaseStats {
        KnowledgeBaseStats {
            fact_count: self.facts.len(),
            rule_count: self.rules.len(),
            program_count: self.programs.len(),
        }
    }
} 