@echo off
cd /d "%~dp0runtime"
del /f /q output.spv unroll_test.spv unroll_test_no_unroll.spv 2>nul
echo === cleaned output .spv files ===
