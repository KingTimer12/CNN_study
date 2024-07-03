use burn::{config::Config, data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset}, module::Module, record::CompactRecorder, tensor::backend::AutodiffBackend, train::{metric::{AccuracyMetric, LossMetric}, LearnerBuilder}};
use config::TrainingConfig;

use crate::data::batcher::MnistBatcher;

mod step;
pub mod config;

fn create_dir(artifact_dir: &str) {
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(dir: &str, config: TrainingConfig, device: B::Device) {
    create_dir(dir);
    config
        .save(format!("{dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let batcher_train = MnistBatcher::<B>::new(device.clone());
    let batcher_valid = MnistBatcher::<B::InnerBackend>::new(device.clone());

    /*
    O treino apresenta os dados que realmente serão usados para produção,
    mas para isso é necessário ajustar esse dados para melhorar precisão.
    O teste serve como cartão resposta, o treino vai tentar prever e depois
    olhará o cartão resposta, vendo onde errou e acertou, melhorando seus dados
    para aumentar sua precisão.
    */

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    /*
    O learner é nosso "indivíduo". 
    
    Explicando de maneira bem metafórica: imagina um estudante
    indo para a escola, antes da prova, ele precisa estudar o conteúdo.
    Ele começa a praticar(treinar) e fica vendo a quantidade de erros e
    acertos no cartão resposta(teste), para então melhorar seu conhecimento
    e ir bem na prova(produção).
    */

    let learner = LearnerBuilder::new(dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    // Aqui começa o treinamento do nosso indivíduo
    let model_trained = learner.fit(dataloader_train, dataloader_test);
    model_trained
        .save_file(format!("{dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}