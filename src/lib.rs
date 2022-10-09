#![allow(incomplete_features)]
#![recursion_limit = "512"]
#![feature(specialization)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]

extern crate self as raddar;

pub mod nn;
pub mod optim;
pub mod dataset;
pub mod util;
pub mod core;