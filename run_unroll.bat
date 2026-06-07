@echo off
cd /d "%~dp0runtime"
echo === gcr unroll_test ===
cargo run -- ../kernels/unroll_test.comp -o unroll_test.spv %*
