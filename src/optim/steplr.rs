use raddar::optim::optimizer::SchedulerAlgorithm;
use raddar_derive::PartialBuilder;
#[derive(PartialBuilder)]
pub struct StepLR {
    #[builder(default = "500")]
    step_size: i64,
    #[builder(default = "0.99")]
    gamma: f64,
    last_step: i64,
    init_lr: f64,
}
impl SchedulerAlgorithm for StepLR {
    fn update(&mut self, step: i64, lr: f64) -> f64 {
        if step - self.last_step >= self.step_size {
            self.last_step = step;
            lr * self.gamma
        } else {
            lr
        }
    }
    fn init(&mut self, init_lr: f64) {
        self.init_lr = init_lr;
    }
}
impl StepLR {
    pub fn new(config: StepLRConfig) -> StepLR {
        StepLR {
            step_size: config.step_size,
            gamma: config.gamma,
            last_step: 0,
            init_lr: 0.,
        }
    }
}
pub fn step_lr(step_size: i64, gamma: f64) -> StepLR {
    StepLRBuilder::default()
        .step_size(step_size)
        .gamma(gamma)
        .build()
}
