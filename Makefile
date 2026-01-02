.PHONY: help install test run clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

install-dev: ## Install dependencies including dev tools
	pip install -r requirements.txt
	pip install pytest black flake8 mypy

run: ## Run the EDA script
	python run_eda.py

test: ## Run tests
	pytest tests/ -v

clean: ## Clean generated files
	rm -rf reports/figures/*.png
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

format: ## Format code with black
	black src/ tests/

lint: ## Run linters
	flake8 src/ tests/
	mypy src/

setup: install ## Initial setup (install dependencies)
	@echo "Setup complete!"

