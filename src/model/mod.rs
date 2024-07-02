use burn::prelude::*;
use cnn::ImageClassificationModel;

pub mod cnn;
pub mod config;

impl<B: Backend> ImageClassificationModel<B> {
    /// # Formas
    ///  - Images [batch_size, height, width]
    ///  - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        
        // Cria uma ligação para segunda dimensão.
        let x = images.reshape([batch_size, 1, height, width]);

        // Agora vamos criar o caminho lógico que deve ser seguido
        let x = self.convulational1.forward(x); // Output: [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.convulational2.forward(x); // Output: [batch_size, 16, _, _]
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x); // Output: [batch_size, 16, 8, 8]

        let x = self.pool.forward(x); // Output: [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16 * 8 ^ 2]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // Output: [batch_size, num_classes]
    }
}