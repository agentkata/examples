SHELL := /bin/sh

DEFAULT_GO ?= go

.DEFAULT_GOAL := help

.PHONY: help sync run clean

help:
	@printf '%s\n' \
		'make sync   Resolve Go dependencies.' \
		'make run    Run the conditional-path example.' \
		'make clean  Remove local Go build artifacts.'

sync:
	$(DEFAULT_GO) mod tidy

run:
	$(DEFAULT_GO) run .

clean:
	rm -rf .cache
