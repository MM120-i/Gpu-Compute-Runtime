RUNTIME = runtime
DASHBOARD = docs/WebPage/chart.js

.PHONY: all check test bench run unroll clean dashboard help mandelbrot

all: check

check:
	cd $(RUNTIME) && cargo check

test: check
	cd $(RUNTIME) && cargo test

bench:
	cd $(RUNTIME) && cargo test --test benchmark_test -- --nocapture

run:
	cd $(RUNTIME) && cargo run -- $(ARGS)

unroll:
	cd $(RUNTIME) && cargo run -- ../kernels/unroll_test.glsl -o unroll_test.spv $(ARGS)

dashboard:
	cd $(DASHBOARD) && npm run dev

clean:
	cd $(RUNTIME) && cargo clean
	-del /f /q $(RUNTIME)\*.spv $(RUNTIME)\*.json $(RUNTIME)\*.ppm>nul

mandelbrot:
	cd $(RUNTIME) && cargo test mandelbrot_render_full_1080p -- --nocapture

help:
	@echo Targets:
	@echo   make check         cargo check
	@echo   make test          cargo check + cargo test
	@echo   make bench         run scan benchmark
	@echo   make run ARGS=...  cargo run --
	@echo   make unroll        run unroll test
	@echo   make dashboard     npm run dev (Chart.js dashboard)
	@echo   make mandelbrot    render Mandelbrot 1080p
	@echo   make clean         cargo clean + delete .spv files
