[![MIT license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE-MIT)
[![Apache 2.0 license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE-APACHE)
[![Docs](https://img.shields.io/docsrs/ndim)](https://docs.rs/ndim)
[![Downloads](https://img.shields.io/crates/d/ndim)](https://crates.io/crates/ndim)
[![Latest crates.io](https://img.shields.io/crates/v/ndim)](https://crates.io/crates/ndim)


A pure Rust library for manipulating multi-dimensional coordinates.


_This crate is in a very preliminary stage of development; any version can introduce breaking changes._


## Crate Features

Ndim offers the following optional features (disabled by default):
* `approx` adds a dependency to the [`approx`](https://crates.io/crates/approx) crate and provides approximate comparison for coordinates of types that implement them.
* `experimental` relies on the experimental rustc feature `generic_const_exprs` and hence requires rust nightly to build.


# License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
