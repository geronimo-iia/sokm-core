# sokm-core

Core primitives for the SOKM (Self-Organizing Kernel Memory) system.

## Crates

- **[`sokm`](crates/sokm/)** — Hebbian link mechanics: decay, strengthen, prune, propagate
- **[`sokm-kernel`](crates/sokm-kernel/)** — Kernel unit layer: activation, one-pass growth, STM, KernelGraph

## Usage

```toml
[dependencies]
sokm = "0.1"
sokm-kernel = "0.1"
```

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
