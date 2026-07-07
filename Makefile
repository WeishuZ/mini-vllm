PYTHON ?= $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP ?= $(PYTHON) -m pip

.PHONY: setup test demo bench trace clean

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m mini_vllm.demo

bench:
	$(PYTHON) benchmarks/run_all.py

trace:
	$(PYTHON) -m mini_vllm.trace_viewer --output docs/trace.html

clean:
	rm -rf build dist *.egg-info .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
