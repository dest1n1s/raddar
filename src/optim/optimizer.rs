use crate::core::{TensorCell, TensorNN};

pub struct Optimizer<T, U, Ts>
where
    T: OptimizerAlgorithm<Ts>,
    U: SchedulerAlgorithm,
    Ts: TensorNN
{
    pub opt: T,
    pub parameters: Vec<TensorCell<Ts>>,
    pub scheduler: Option<U>,
    pub step: i64,
}

pub trait OptimizerAlgorithm<Ts: TensorNN> {
    fn step(&mut self, training_parameters: &Vec<TensorCell<Ts>>);
    fn init(&mut self, training_parameters: &Vec<TensorCell<Ts>>);
    fn learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
}

pub trait SchedulerAlgorithm {
    fn init(&mut self, init_lr: f64);
    fn update(&mut self, step: i64, lr: f64) -> f64;
}
impl<T, U, Ts> Optimizer<T, U, Ts>
where
    T: OptimizerAlgorithm<Ts>,
    U: SchedulerAlgorithm,
    Ts: TensorNN
{
    pub fn step(&mut self) {
        self.step += 1;
        if let Some(scheduler) = &mut self.scheduler {
            self.opt
                .set_learning_rate(scheduler.update(self.step, self.opt.learning_rate()));
        }
        self.opt.step(&self.parameters);
    }
    pub fn new(
        parameters: Vec<TensorCell<Ts>>,
        mut opt: T,
        mut scheduler: Option<U>,
    ) -> Optimizer<T, U, Ts> {
        opt.init(&parameters);
        let init_lr = opt.learning_rate();
        if let Some(sched) = &mut scheduler {
            sched.init(init_lr);
        }
        Optimizer {
            opt,
            parameters,
            scheduler,
            step: 0,
        }
    }
}

pub struct ConstantScheduler {
    lr: f64,
}

impl SchedulerAlgorithm for ConstantScheduler {
    fn init(&mut self, init_lr: f64) {
        self.lr = init_lr;
    }

    fn update(&mut self, _step: i64, lr: f64) -> f64 {
        lr
    }
}

impl ConstantScheduler {
    pub fn new() -> ConstantScheduler {
        ConstantScheduler { lr: 0. }
    }
}
pub fn opt_with_sched<T, U, Ts>(parameters: Vec<TensorCell<Ts>>, opt: T, sched: U) -> Optimizer<T, U, Ts>
where
    T: OptimizerAlgorithm<Ts>,
    U: SchedulerAlgorithm,
    Ts: TensorNN
{
    Optimizer::new(parameters, opt, Some(sched))
}

pub fn opt<T, Ts>(parameters: Vec<TensorCell<Ts>>, opt: T) -> Optimizer<T, ConstantScheduler, Ts>
where
    T: OptimizerAlgorithm<Ts>,
    Ts: TensorNN
{
    Optimizer::new(parameters, opt, Some(ConstantScheduler::new()))
}
