use derive_builder::Builder;
use raddar::optim::optimizer::SchedulerAlgorithm;
#[derive(Builder)]
#[builder(pattern = "owned")]
pub struct StepLR {
    #[builder(default = "500")]
    step_size: i32,
    #[builder(default = "0.99")]
    gamma: f64,
    #[builder(default = "0")]
    last_step: i32,
    #[builder(default = "0.")]
    init_lr: f64,
}
impl SchedulerAlgorithm for StepLR {
    fn update(&mut self, step: i32, lr: f64) -> f64 {
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
    pub fn new(step_size: i32, gamma: f64) -> StepLR {
        StepLR {
            step_size,
            gamma,
            last_step: 0,
            init_lr: 0.,
        }
    }
}
