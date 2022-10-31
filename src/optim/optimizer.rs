use crate::{core::TensorCell, nn::module::Module};
pub struct Optimizer<T, U>
where
    T: OptimizerAlgorithm,
    U: SchedulerAlgorithm,
{
    opt: T,
    trainable_parameters: Vec<TensorCell>,
    scheduler: Option<U>,
    step: i64,
}

pub trait OptimizerAlgorithm {
    fn step(&mut self, training_parameters: &Vec<TensorCell>);
    fn init(&mut self, training_parameters: &Vec<TensorCell>);
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
}
pub trait SchedulerAlgorithm {
    fn init(&mut self, init_lr: f64);
    fn update(&mut self, step: i64, lr: f64) -> f64;
}

impl<T, U> Optimizer<T, U>
where
    T: OptimizerAlgorithm,
    U: SchedulerAlgorithm,
{
    pub fn step(&mut self) {
        self.step += 1;
        if let Some(scheduler) = &mut self.scheduler {
            self.opt
                .set_learning_rate(scheduler.update(self.step, self.opt.learning_rate()));
        }
        self.opt.step(&self.trainable_parameters);
    }
    pub fn new(model: &dyn Module, mut opt: T, mut scheduler: Option<U>) -> Optimizer<T, U> {
        opt.init(&model.training_parameters());
        let init_lr = opt.learning_rate();
        if let Some(sched) = &mut scheduler {
            sched.init(init_lr);
        }
        Optimizer {
            opt,
            trainable_parameters: model.training_parameters(),
            scheduler,
            step: 0,
        }
    }
}
