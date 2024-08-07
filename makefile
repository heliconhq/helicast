# # Specify the shell to use
# SHELL := /bin/bash

# check:
# 	@if [ -z "$$(which rye)" ]; then echo "Rye is not installed. Please install Rye to proceed."; exit 1; fi
# 	@echo "Rye is installed."

# install: check
# 	@rye sync
# 	@echo "Dependencies installed."


# docs: check
# 	@cd docs; rm -rf source/python-api; python3 source/_scripts/make-python-api.py; make html; open build/html/index.html


# open_docs: check
# 	@open docs/build/html/index.html


# help:
# 	@echo "Available commands:"
# 	@echo "  make check           - Check if Rye is installed, throw an error if not."
# 	@echo "  make install         - Install dependencies using rye sync"
# 	@echo "  make help            - Show this help message."

# # Set default target to help
# .DEFAULT_GOAL := help


# Specify the shell to use
SHELL := /bin/bash

check:
	@if [ -z "$$(which rye)" ]; then echo "Rye is not installed. Please install Rye to proceed."; exit 1; fi
	@echo "Rye is installed."

install: check
	@rye sync
	@echo "Dependencies installed."

docs: check
	@set -e; cd docs; \
	rm -rf source/python-api; \
	python3 source/_scripts/make-python-api.py; \
	rye run sphinx-build -b html source build/html; \
	if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then \
	    explorer.exe build/html/index.html; \
	elif [ "$(shell uname)" = "Darwin" ]; then \
	    open build/html/index.html; \
	elif [ "$(shell uname)" = "Linux" ]; then \
	    xdg-open build/html/index.html; \
	fi

open_docs: check
	@if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then \
	    explorer.exe docs/build/html/index.html; \
	elif [ "$(shell uname)" = "Darwin" ]; then \
	    open docs/build/html/index.html; \
	elif [ "$(shell uname)" = "Linux" ]; then \
	    xdg-open docs/build/html/index.html; \
	fi

help:
	@echo "Available commands:"
	@echo "  make check           - Check if Rye is installed, throw an error if not."
	@echo "  make install         - Install dependencies using rye sync"
	@echo "  make docs            - Build the documentation and open it"
	@echo "  make open_docs       - Open the existing documentation"
	@echo "  make help            - Show this help message."

# Set default target to help
.DEFAULT_GOAL := help

