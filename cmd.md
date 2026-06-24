Commands, cuz i keep forgetting:

.\make or .\make check -> cargo check

.\make test -> cargo check + cargo test (all 33 tests)

.\make bench -> runs only the scan benchmark

.\make run -> ARGS="--no-ast-opt" cargo run -- <args>

.\make clean -> cargo clean + deletes stray .spv files

.\make help -> prints available targets

Make run examples:

.\make run ARGS="../kernels/benchmarks/scan_pass1.comp -o scan_pass1.spv --no-ast-opt"
