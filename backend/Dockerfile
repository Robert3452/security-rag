# Slim y no alpine por compatibilidad de wheels para binarios como (pytorch, numpy)
# Se compilarían desde cero es lento y frágil
FROM python:3.12-slim 

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv
WORKDIR /app

# Compile los .pyc durante la instalación con UV_COMPILE_BYTECODE=1
# UV_LINK_MODE=COPY uv crea hardlinks desde su caché, copiarlos evita problemas o advertencias
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
# --frozen instala todas las dependencias que estan en uv.lock no resuelve versiones
# --no-dev sin pytest ni mypy, la imagen queda más liviana
# --no-install-project solo instala dependencias, no el proyecto, ya que el codigo aun no se copió
# RUN uv sync --frozen --no-dev --no-install-project
RUN uv sync --locked --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["fastapi", "run","app/main.py","--port","8000"]