#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(unused_unsafe)]
#![allow(clippy::too_many_arguments)]

#[macro_use]
extern crate failure_derive;
#[macro_use]
extern crate derive_more;

#[macro_use]
mod util;

mod cuda;
pub use cuda::*;

#[cfg(feature = "nvrtc")]
pub mod nvrtc;
