# Deployment Plan — ClarityAI

## 1. Deployment Goals

- The application must run **outside the original development environment** with minimal setup friction.
- **Docker Compose** is the primary deployment method (local or on a single cloud VM).
- Cloud deployment is optional but recommended for demo purposes (portfolio value).
- Configuration must be environment-variable driven — no hard-coded secrets, paths, or hostnames.

## 2. Containerization Strategy

### 2.1 Backend Dockerfile (multi-stage build)

```dockerfile
# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Runtime image ---
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY ./app ./app
COPY ./models ./models
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage:** keeps the final image lean by not shipping build tools; separates dependency installation from application code for better layer caching.

### 2.2 Frontend Dockerfile

```dockerfile
# --- Stage 1: Build React app ---
FROM node:20-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# --- Stage 2: Serve via nginx ---
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### 2.3 Docker Compose

```yaml
version: "3.9"

services:
  db:
    image: postgres:16-alpine
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: ${DB_NAME}
    volumes:
      - db_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build: ./backend
    restart: unless-stopped
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${DB_USER}:${DB_PASSWORD}@db:5432/${DB_NAME}
      MODEL_PATH: /app/models/model_v1.pt
      CORS_ORIGINS: ${CORS_ORIGINS}
    ports:
      - "8000:8000"

  frontend:
    build: ./frontend
    restart: unless-stopped
    depends_on:
      - backend
    ports:
      - "80:80"

volumes:
  db_data:
```

### 2.4 Environment Configuration (`.env`)

```
DB_USER=clarityai
DB_PASSWORD=change_me
DB_NAME=clarityai_db
CORS_ORIGINS=http://localhost
MODEL_PATH=/app/models/model_v1.pt
LOG_LEVEL=info
```

`.env.example` should be committed; the real `.env` must be gitignored.

## 3. Local Deployment Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd ClarityAI

# 2. Copy and configure environment variables
cp .env.example .env
# edit .env as needed

# 3. Build and start all services
docker compose up --build

# 4. Verify services
curl http://localhost:8000/health
# Frontend available at http://localhost
```

To tear down (including volumes, for a clean reset):
```bash
docker compose down -v
```

## 4. Database Migrations

Migrations run automatically on backend startup, or manually via:
```bash
docker compose exec backend alembic upgrade head
```

## 5. Model Loading & Inference Strategy

- Model weights (`model_v1.pt` and any classical-model artifacts like `.joblib` files) are **baked into the backend image** at build time (copied from `./models` into the container) for reproducibility, OR mounted as a volume for easier iteration during development.
- The model is loaded **once at application startup** (in a FastAPI `lifespan`/startup event) and kept in memory — not reloaded per request — to keep inference latency low.
- `MODEL_PATH` environment variable allows swapping model versions without rebuilding the image.
- The `/health` endpoint checks that the model has successfully loaded (`model_loaded: true/false`) in addition to DB connectivity, so orchestration tools can detect a broken deployment.

## 6. Health & Monitoring

- `GET /health` returns:
```json
{
  "status": "ok",
  "database": "connected",
  "model_loaded": true,
  "version": "1.0.0"
}
```
- Docker Compose `healthcheck` blocks use this endpoint to determine container readiness.
- (Optional/bonus) Structured JSON logging + a lightweight log aggregation setup (e.g., shipping logs to a file volume) for production observability.

## 7. Cloud Deployment (Optional)

If a live demo URL is desired, low-friction options include:

- **Render** or **Railway** — support direct Docker Compose–style multi-service deployment with minimal configuration, good for demo-scale projects.
- **Fly.io** — good Docker-native support, generous free tier, supports Postgres add-ons.

General steps (platform-agnostic):
1. Push built images to a container registry (Docker Hub / GitHub Container Registry).
2. Provision a managed Postgres instance (or use the platform's own Postgres offering).
2. Set environment variables in the platform's dashboard (matching `.env`).
4. Point the platform at the `Dockerfile`s (or Compose file, if supported).
5. Confirm `/health` responds correctly post-deploy.
6. Record the deployed URL in the README.

## 8. Rollback Strategy

- Tag Docker images with version numbers (`clarityai-backend:v1.0.0`) rather than relying solely on `latest`.
- Keep the previous known-good image tag available so a bad deploy can be reverted by redeploying the prior tag.

## 9. Security Considerations

- No secrets committed to source control — all via `.env` / platform secret managers.
- CORS explicitly restricted to known frontend origin(s) via `CORS_ORIGINS`.
- File upload validation (type, size limits) enforced server-side to prevent abuse.
- Database credentials scoped to a dedicated application user, not a superuser.

## 10. Post-Deployment Checklist

- [ ] `/health` returns `200 OK` with `model_loaded: true`
- [ ] Image upload → analysis → history flow works end-to-end on the deployed URL
- [ ] Frontend correctly communicates with backend (no CORS errors)
- [ ] Database persists data across container restarts (volume correctly mounted)
- [ ] README updated with the live deployment URL (if applicable)