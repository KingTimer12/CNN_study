use burn::{config::Config, data::{dataloader::batcher::Batcher, dataset::vision::MnistItem}, module::Module, record::{CompactRecorder, Recorder}, tensor::backend::Backend};

use crate::{data::batcher::MnistBatcher, training::config::TrainingConfig};

pub fn infer<B: Backend>(dir: &str, device: B::Device, item: MnistItem) {
    let config = TrainingConfig::load(format!("{dir}/config.json"))
        .expect("Config should exist for the model");
    let record = CompactRecorder::new().load(format!("{dir}/model").into(), &device)
        .expect("Trained model should exist");

    let model = config.model.init::<B>(&device).load_record(record);

    let label = item.label;
    let batcher = MnistBatcher::new(device);
    let batch = batcher.batch(vec![item]);
    let output = model.forward(batch.images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    println!("Predicted {} Expected {}", predicted, label);
}