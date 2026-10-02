# SELENE — one-command developer workflow. Everything is project-local (no global installs).
SHELL := /bin/bash
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PY    := $(ROOT)/.venv/bin/python
PIP   := $(ROOT)/.venv/bin/pip
NODE_BIN := $(ROOT)/.tools/node/bin
export PATH := $(NODE_BIN):$(ROOT)/.venv/bin:$(PATH)
export NUMBA_CACHE_DIR := $(ROOT)/.numba_cache

.PHONY: setup setup-py setup-js data test test-fast dev api web build precompute demo clean

setup: setup-py setup-js data

# Fetch the JPL kernels (≈33 MB) if they are not cached yet; everything else is committed.
NAIF := https://naif.jpl.nasa.gov/pub/naif/generic_kernels
data:
	@mkdir -p $(ROOT)/data/cache
	@test -s $(ROOT)/data/cache/de440s.bsp || curl -fL --retry 3 -o $(ROOT)/data/cache/de440s.bsp $(NAIF)/spk/planets/de440s.bsp
	@test -s $(ROOT)/data/cache/gm_de440.tpc || curl -fL --retry 3 -o $(ROOT)/data/cache/gm_de440.tpc $(NAIF)/pck/gm_de440.tpc

setup-py:
	@test -x $(PY) || python3 -m venv $(ROOT)/.venv
	$(PIP) install -q -r $(ROOT)/backend/requirements.lock
	$(PIP) install -q -e "$(ROOT)/backend[dev]"

setup-js:
	@cd $(ROOT)/frontend && npm install --no-audit --no-fund

test:
	cd $(ROOT)/backend && $(PY) -m pytest -q -n auto

test-fast:
	cd $(ROOT)/backend && $(PY) -m pytest -q -x -m "not slow"

api:
	cd $(ROOT)/backend && $(PY) -m uvicorn selene.api.app:app --host 127.0.0.1 --port 8000 --reload

web:
	cd $(ROOT)/frontend && npm run dev -- --host 127.0.0.1 --port 5173

# Run API + Vite together; Ctrl-C stops both.
dev:
	@trap 'kill 0' INT TERM EXIT; \
	( cd $(ROOT)/backend && $(PY) -m uvicorn selene.api.app:app --host 127.0.0.1 --port 8000 ) & \
	( cd $(ROOT)/frontend && npm run dev -- --host 127.0.0.1 --port 5173 ) & \
	wait

build:
	cd $(ROOT)/frontend && npm run build
	rm -rf $(ROOT)/backend/selene/api/static && cp -r $(ROOT)/frontend/dist $(ROOT)/backend/selene/api/static

# Regenerate committed caches: orbit library, Horizons samples, demo bundle.
precompute:
	cd $(ROOT)/backend && $(PY) -m selene.orbits.library --rebuild
	cd $(ROOT)/backend && $(PY) -m selene.objects.horizons --refresh || true
	cd $(ROOT)/backend && $(PY) -m selene.scenario.demo --rebuild

# Serve the production build from uvicorn only (single process, offline).
demo: build
	cd $(ROOT)/backend && $(PY) -m uvicorn selene.api.app:app --host 127.0.0.1 --port 8000

clean:
	rm -rf $(ROOT)/backend/.pytest_cache $(ROOT)/.numba_cache $(ROOT)/frontend/dist
