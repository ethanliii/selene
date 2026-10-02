FROM node:24-slim AS web
WORKDIR /web
COPY frontend/package*.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
COPY backend/ /app/backend/
COPY data/orbits /app/data/orbits
COPY data/horizons /app/data/horizons
COPY data/demo /app/data/demo
RUN pip install --no-cache-dir -e /app/backend
COPY --from=web /web/dist /app/backend/selene/api/static
ENV NUMBA_CACHE_DIR=/tmp/numba
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "selene.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app/backend"]
