use burn::{optim::AdamConfig, prelude::*};

use crate::model::config::ImageClassificationConfig;

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ImageClassificationConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}