# Specify the shell to use
SHELL := /bin/bash

check:
	@if [ -z "$$(which rye)" ]; then echo "Rye is not installed. Please install Rye to proceed."; exit 1; fi
	@echo "Rye is installed."

install: check
	@rye sync
	@echo "Dependencies installed."


docs: check
	@cd docs; rm -rf source/python-api; python3 source/_scripts/make-python-api.py; make html; open build/html/index.html


open_docs: check
	@open docs/build/html/index.html


help:
	@echo "Available commands:"
	@echo "  make check           - Check if Rye is installed, throw an error if not."
	@echo "  make install         - Install dependencies using rye sync"
	@echo "  make help            - Show this help message."

# Set default target to help
.DEFAULT_GOAL := help
