use super::Module;

pub trait Trainer<T: Module> {
    fn train(&mut self, epochs: usize) -> f32;
    fn eval(&mut self, batch: &Batch) -> f32;
    fn validate(&mut self, batch: &Batch) -> f32;
    fn train_epoch(&mut self, dataset: &Dataset) -> f32;
    fn eval_epoch(&mut self, dataset: &Dataset) -> f32;
    fn validate_epoch(&mut self, dataset: &Dataset) -> f32;
    fn train_epochs(&mut self, dataset: &Dataset, epochs: usize) -> Vec<f32>;
    fn eval_epochs(&mut self, dataset: &Dataset, epochs: usize) -> Vec<f32>;
    fn train_batches(&mut self, dataset: &Dataset, batches: usize) -> Vec<f32>;
    fn eval_batches(&mut self, dataset: &Dataset, batches: usize) -> Vec<f32>;
    fn validate_batches(&mut self, dataset: &Dataset, batches: usize) -> Vec<f32>;
    fn model(&self) -> T;
}
