#![recursion_limit = "512"]
use std::fs::create_dir;

use burn::{
    backend::{Autodiff, Wgpu}, data::dataset::{vision::MnistDataset, Dataset}, optim::AdamConfig
};

use crate::{
    inference::infer, model::config::ImageClassificationConfig, training::{config::TrainingConfig, train}
};

mod data;
mod inference;
mod model;
mod training;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = std::path::PathBuf::from("./artifacts");
    create_dir(&artifact_dir).ok();

    if !artifact_dir.exists() {
        let model = ImageClassificationConfig::new(10, 512);
        train::<MyAutodiffBackend>(
            artifact_dir.to_str().unwrap_or_default(),
            TrainingConfig::new(model, AdamConfig::new()),
            device,
        );
    } else {
        let item = MnistDataset::test()
            .get(44)
            .unwrap();
        infer::<MyAutodiffBackend>(artifact_dir.to_str().unwrap_or_default(), device, item);
    }
}
