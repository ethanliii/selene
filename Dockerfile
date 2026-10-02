FROM node:24-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir -r /app/requirements.lock
COPY backend/ /app/backend/
RUN pip install --no-cache-dir --no-deps -e /app/backend
COPY data/orbits /app/data/orbits
COPY data/horizons /app/data/horizons
COPY data/demo /app/data/demo
# JPL kernels (~33 MB) are fetched at build time so a plain `docker run` works offline afterwards.
RUN mkdir -p /app/data/cache && \
    curl -fL --retry 3 -o /app/data/cache/de440s.bsp https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp && \
    curl -fL --retry 3 -o /app/data/cache/gm_de440.tpc https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc
COPY --from=web /web/dist /app/backend/selene/api/static
ENV NUMBA_CACHE_DIR=/tmp/numba
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "selene.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
