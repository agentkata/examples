SHELL := /bin/sh

.DEFAULT_GOAL := help

.PHONY: help sync run clean

help:
	@printf '%s\n' \
		'make sync   Install example dependencies with uv.' \
		'make run    Run the secret-echo example.' \
		'make clean  Remove the local virtual environment.'

sync:
	uv sync

run:
	uv run python main.py

clean:
	rm -rf .venv
