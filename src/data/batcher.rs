use burn::
    prelude::*
;

#[derive(Clone)]
pub struct MnistBatcher<B: Backend> {
    pub device: B::Device
}

impl<B: Backend> MnistBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}