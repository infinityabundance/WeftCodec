FROM rust:1-slim-bookworm AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential pkg-config ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin weftcodec \
 && strip /src/target/release/weftcodec

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/weftcodec /usr/local/bin/weftcodec
LABEL org.opencontainers.image.title="WeftCodec" \
      org.opencontainers.image.description="WeftCodec v1 lossless context-conditioned codec (Rust)" \
      org.opencontainers.image.source="https://github.com/infinityabundance/WeftCodec" \
      org.opencontainers.image.licenses="Apache-2.0"
ENTRYPOINT ["weftcodec"]
