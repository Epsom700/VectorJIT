# VectorJIT

VectorJIT is a small project that demonstrates a tiny tensor expression IR and a JIT
code-generator that emits optimized C++ (NEON/OpenMP) for elementwise and matmul
kernels. It's useful for experimentation with codegen, intrinsics, and simple
benchmarking.

## Contents

- `python/` - Python front-end (Tensor, Compiler, ops, small optimizer).
- `tests/` - Benchmarks and small tests (including `tests/benchmark_claude.py`).
- `examples/` - Example scripts.
- `training/` - tiny training scripts (XOR examples).

## Requirements

- Python 3.10+ (tested with 3.12 in the repo's venv)
- `numpy`
- A C++ toolchain (clang++/g++) that supports building shared libraries
- OpenMP (`libomp`) is required on macOS for parallel builds
Install Python deps:

```bash
python3 -m venv vectorJIT
source vectorJIT/bin/activate
pip install -r requirements.txt
```

If you prefer a pinned set of packages, install them from `vectorJIT` virtualenv
or run `pip freeze > requirements.txt` and commit the result.

## Quick start

1. Activate the virtualenv (see above).
2. Run the matmul benchmark (small sizes by default):

```bash
python3 tests/benchmark_claude.py
```

The benchmark compiles C++ code at runtime using `python/compiler.py` and then
loads the shared library via `ctypes`. On first run you will see C++ compilation
happen for each kernel.

## Notes about macOS / libomp

On macOS, Apple Clang requires a separate OpenMP runtime. If compilation fails
with missing OpenMP headers or linker errors, install libomp with Homebrew:

```bash
brew install libomp
```

`python/compiler.py` already includes the common Homebrew include and lib paths
(`/opt/homebrew/opt/libomp`) when it detects macOS. If your Homebrew is under
`/usr/local`, adjust the paths in `python/compiler.py` accordingly.

## Generated artifacts & cleanup

- The JIT writes a temporary `.cpp` file and compiles it to a `.so` in the
	system temp directory. The `.cpp` file is removed after loading; compiled
	`.so` files are left behind for inspection. You can remove them manually from
	`/tmp` or modify `Compiler.jit_compile` to delete them after `ctypes.CDLL`.

## Development notes

- `python/compiler.py` contains multiple code-generation backends (scalar and
	NEON/SIMD). The SIMD emitters use ARM NEON intrinsics (so they require an
	ARM/AArch64 toolchain). The code also generates scalar fallbacks for
	cross-platform execution in several places — review emitters before running
	on x86 if you see compilation errors.
- There are forward and backward kernel generators (`compile`,
	`compile_backward`, `compile_matmul_backward`). If you modify codegen, run
	the small tests in `tests/` to validate correctness.

## Useful files

- `requirements.txt` — Python dependencies.
- `.gitignore` — ignores the `vectorJIT/` venv, `__pycache__`, and compiled
	artifacts.
- `tests/benchmark_claude.py` — an example benchmark harness (matmul and
	elementwise variants).
- `python/compiler.py` — the JIT compiler & code generator.

## Contributing

Contributions welcome. Open an issue or a PR with small, focused changes. If
you add new generated kernels, include a short benchmark or test that
verifies correctness.

## License

TBD
# VectorJIT
