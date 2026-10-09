.PHONY: install test lint demo benchmark fetch clean

install:
	pip install -e ".[dev]"

test:
	pytest --cov=wellforecast --cov-report=term-missing

lint:
	ruff check wellforecast tests

demo:
	python -m wellforecast.cli demo

benchmark:
	python -m wellforecast.real_benchmark

fetch:
	python -c "from wellforecast.real_data import fetch_volve; print(fetch_volve(), 'rows')"

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov datasets/_volve_raw.xlsx
