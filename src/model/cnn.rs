use burn::{
    nn::{
        conv::Conv2d,
        pool::AdaptiveAvgPool2d,
        Dropout, Linear, Relu
    },
    prelude::*
};

// Linear define um Output usando a fórmula: O = I*W + b 
//                              | Output | Input | Weights | Bias |

#[derive(Module, Debug)]
pub struct ImageClassificationModel<B: Backend> {
    pub convulational1: Conv2d<B>,
    pub convulational2: Conv2d<B>,
    pub pool: AdaptiveAvgPool2d,
    pub dropout: Dropout,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: Relu
}