@echo off
cd /d "%~dp0runtime"
echo === cargo check ===
cargo check || exit /b %errorlevel%
echo === cargo test ===
cargo test || exit /b %errorlevel%
echo === done ===