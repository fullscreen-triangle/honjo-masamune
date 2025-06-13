# Multi-stage build for Honjo Masamune Truth Engine
FROM rust:1.75-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

# Create dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release --bin honjo-masamune
RUN rm src/main.rs

# Copy source code
COPY src/ ./src/
COPY config/ ./config/

# Build the actual application
RUN cargo build --release --bin honjo-masamune
RUN cargo build --release --bin buhera-cli
RUN cargo build --release --bin preparation-manager
RUN cargo build --release --bin truth-synthesizer

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r honjo && useradd -r -g honjo honjo

# Create app directory
WORKDIR /app

# Copy binaries from builder stage
COPY --from=builder /app/target/release/honjo-masamune /usr/local/bin/
COPY --from=builder /app/target/release/buhera-cli /usr/local/bin/
COPY --from=builder /app/target/release/preparation-manager /usr/local/bin/
COPY --from=builder /app/target/release/truth-synthesizer /usr/local/bin/

# Copy configuration
COPY --from=builder /app/config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs /app/data/corpus /app/atp_state /app/data/models
RUN chown -R honjo:honjo /app

# Switch to app user
USER honjo

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8081 8082

# Default command
CMD ["honjo-masamune"] 