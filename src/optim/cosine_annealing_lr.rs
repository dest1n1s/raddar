use raddar::optim::optimizer::SchedulerAlgorithm;
use raddar_derive::ArchitectureBuilder;
use std::f64::consts::PI;

#[derive(ArchitectureBuilder)]
pub struct CosineAnnealingLR {
    #[builder(default = "500")]
    step_size: i64,
    #[builder(default = "0.0001")]
    eta_min: f64,
    last_step: i64,
    init_lr: f64,
    #[builder(default = "100")]
    warmup_step: i64,
}
impl SchedulerAlgorithm for CosineAnnealingLR {
    fn init(&mut self, init_lr: f64) {
        self.init_lr = init_lr;
    }
    fn update(&mut self, step: i64, _lr: f64) -> f64 {
        self.last_step = step;
        if self.warmup_step > step {
            (step as f64) * (self.init_lr - self.eta_min) / (self.warmup_step as f64) + self.eta_min
        } else {
            self.eta_min
                + (self.init_lr - self.eta_min)
                    * (1. + ((step as f64) / (self.step_size as f64) * PI).cos())
        }
    }
}
impl CosineAnnealingLR {
    pub fn new(config: CosineAnnealingLRConfig) -> CosineAnnealingLR {
        CosineAnnealingLR {
            step_size: config.step_size,
            eta_min: config.eta_min,
            last_step: 0,
            init_lr: 0.,
            warmup_step: config.warmup_step,
        }
    }
}
pub fn cosine_annealing_lr(step_size: i64, warmup_step: i64, eta_min: f64) -> CosineAnnealingLR {
    CosineAnnealingLRBuilder::default()
        .step_size(step_size)
        .warmup_step(warmup_step)
        .eta_min(eta_min)
        .build()
}
