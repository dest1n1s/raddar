#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
#![recursion_limit = "512"]
#![feature(specialization)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(generic_const_exprs)]
#![feature(slice_flatten)]
#![feature(type_alias_impl_trait)]
#![feature(unsize)]
#![feature(coerce_unsized)]

extern crate self as raddar;

pub mod core;
pub mod dataset;
pub mod nn;
pub mod optim;
pub mod util;
