RUNTIME = runtime
DASHBOARD = docs/WebPage/chart.js

.PHONY: all check test bench run unroll clean dashboard help

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
	cd $(RUNTIME) && cargo run -- ../kernels/unroll_test.comp -o unroll_test.spv $(ARGS)

dashboard:
	cd $(DASHBOARD) && npm run dev

clean:
	cd $(RUNTIME) && cargo clean
	-del /f /q $(RUNTIME)\*.spv $(RUNTIME)\*.json>nul

help:
	@echo Targets:
	@echo   make check         cargo check
	@echo   make test          cargo check + cargo test
	@echo   make bench         run scan benchmark
	@echo   make run ARGS=...  cargo run --
	@echo   make unroll        run unroll test
	@echo   make dashboard     npm run dev (Chart.js dashboard)
	@echo   make clean         cargo clean + delete .spv files
