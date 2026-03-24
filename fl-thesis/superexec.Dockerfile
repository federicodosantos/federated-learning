FROM flwr/superexec:1.22.0

USER root
RUN apt-get update \
    && apt-get -y --no-install-recommends install build-essential \
    && rm -rf /var/lib/apt/lists/*

USER app
WORKDIR /app

# Copy project metadata
COPY --chown=app:app pyproject.toml .

# 1. Remove any leftover torch lines (safety)
RUN sed -i '/"torch==/d' pyproject.toml && \
    sed -i '/"torchvision==/d' pyproject.toml

# 2. Install PyTorch CPU manually
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchao>=0.15.0

# 3. Install all Flower app dependencies
RUN pip install --no-cache-dir .

# 4. Copy the actual source code
COPY --chown=app:app . .

ENTRYPOINT ["flower-superexec"]
