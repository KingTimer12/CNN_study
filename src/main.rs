use std::path::Path;

use burn::{backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu}, data::dataset::{vision::MnistDataset, Dataset}, optim::AdamConfig};
use inference::infer;
use model::config::ImageClassificationConfig;
use training::{config::TrainingConfig, train};

mod model;
mod data;
mod training;
mod inference;

fn main() {
    type ImageBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type ImageAutodiffBackend = Autodiff<ImageBackend>;

    let artifact = "./artefact";
    let device = burn::backend::wgpu::WgpuDevice::default();
    if !Path::new(artifact).exists() {
        train::<ImageAutodiffBackend>(
            &artifact, 
            TrainingConfig::new(
                ImageClassificationConfig::new(10, 512), 
                AdamConfig::new()), 
            device.clone());
    }
    
    let item = MnistDataset::test().get(340).unwrap();
    infer::<ImageBackend>(&artifact, device, item)
}
