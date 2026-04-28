# SSIS Sovereign Production Image
# <plan_complexity>Medium</plan_complexity><total_lines_estimate>120</total_lines_estimate>

# Stage 1: Build
FROM golang:1.26-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    wget

# Download and install ONNX Runtime C/C++ libraries
# Target: Version 1.17.1 (stable for CGO/oneDNN)
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.17.1.tgz \
    && cp onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-1.17.1/include/* /usr/local/include/ \
    && ldconfig

WORKDIR /app
COPY . .

# Build the sovereign engine
# CGO_ENABLED=1 is required for ONNX Runtime bindings
# -tags onnx compiles real ONNX inference (not MockInferrer stubs)
RUN CGO_ENABLED=1 GOOS=linux go build -tags onnx -o ssis_engine ./engine/main.go

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies (OpenVINO / oneDNN support)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy ONNX Runtime shared libraries from builder
COPY --from=builder /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

WORKDIR /app
COPY --from=builder /app/ssis_engine .
COPY --from=builder /app/pkg/ml/models ./pkg/ml/models
COPY --from=builder /app/configs ./configs

# SSIS Operational Environment Variables
ENV SSIS_MODEL_DIR=/app/pkg/ml/models
ENV SSIS_CONFIG_PATH=/app/configs
ENV SSIS_GRPC_PORT=50051
ENV SSIS_REST_PORT=8080
ENV LD_LIBRARY_PATH=/usr/local/lib

EXPOSE 50051 8080

ENTRYPOINT ["./ssis_engine", "--provider=CPU"]
