use derive_builder::Builder;
use raddar::optim::optimizer::SchedulerAlgorithm;
use std::f64::consts::PI;

#[derive(Builder)]
#[builder(pattern = "owned")]
struct CosineAnnealingLR {
    #[builder(default = "500")]
    step_size: i32,
    #[builder(default = "0.0001")]
    eta_min: f64,
    #[builder(default = "0")]
    last_step: i32,
    #[builder(default = "0.")]
    init_lr: f64,
}
impl SchedulerAlgorithm for CosineAnnealingLR {
    fn init(&mut self, init_lr: f64) {
        self.init_lr = init_lr;
    }
    fn update(&mut self, step: i32, _lr: f64) -> f64 {
        self.last_step = step;
        self.eta_min
            + (self.init_lr - self.eta_min)
                * (1. + (f64::from(step) / f64::from(self.step_size) * PI).cos())
    }
}
impl CosineAnnealingLR {
    fn new(step_size: i32, eta_min: f64) -> CosineAnnealingLR {
        CosineAnnealingLR {
            step_size,
            eta_min,
            last_step: 0,
            init_lr: 0.,
        }
    }
}
