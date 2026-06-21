# Snowball Notes — runtime image.
# Builds the agent runtime with no API keys required (the default heuristic
# adapter and local embedding provider run fully offline). Mount a config and
# set provider keys via env to switch to a hosted model.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first for layer caching, then the package.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bundle the offline config and eval fixtures so the image is runnable as-is.
COPY ci/offline.config.yaml ./config.yaml
COPY eval ./eval

# Run as a non-root user.
RUN useradd --create-home --uid 1000 snowball \
    && mkdir -p /app/data /app/logs /app/vault \
    && chown -R snowball:snowball /app
USER snowball

# Default: print health. Override the command to run the worker, review, etc.
ENTRYPOINT ["python", "-m", "snowball_notes.cli", "--config", "config.yaml"]
CMD ["status"]
