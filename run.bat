@echo off
cd /d "%~dp0runtime"
echo === gcr %* ===
cargo run -- %*
