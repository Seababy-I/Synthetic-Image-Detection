# Build Stage: Frontend
FROM node:18 AS build-stage
WORKDIR /frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Run Stage: Backend
FROM python:3.9-slim AS run-stage

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY Backend/Requirements.txt ./
# For Hugging Face, use CPU-only to avoid massive installs
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r Requirements.txt

# Copy backend code and model
COPY Backend/ ./ 

# Copy built frontend from build stage
COPY --from=build-stage /frontend/dist ./static

# Expose the default Hugging Face Space port
EXPOSE 7860

# Command to run the application (directing FastAPI to serve static files)
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:7860"]
