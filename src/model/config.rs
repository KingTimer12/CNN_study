use burn::{
    nn::{
        conv::Conv2dConfig,
        pool::AdaptiveAvgPool2dConfig,
        DropoutConfig,
        LinearConfig,
        Relu
    },
    prelude::*
};

use super::cnn::ImageClassificationModel;

#[derive(Config, Debug)]
pub struct ImageClassificationConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64
}

impl ImageClassificationConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ImageClassificationModel<B> {
        ImageClassificationModel {
            convulational1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            convulational2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            activation: Relu::new()
        }
    }
}