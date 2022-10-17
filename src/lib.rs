#![allow(incomplete_features)]
#![recursion_limit = "512"]
#![feature(specialization)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(generic_const_exprs)]
#![feature(slice_flatten)]

extern crate self as raddar;

pub mod core;
pub mod dataset;
pub mod models;
pub mod nn;
pub mod optim;
pub mod util;
