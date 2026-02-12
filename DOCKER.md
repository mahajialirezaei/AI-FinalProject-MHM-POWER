# Docker Deployment Guide

This guide explains how to run the Bank Marketing Campaign Predictor UI using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose (optional, for easier deployment)

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and run the container
docker-compose up --build

# Run in detached mode
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

The UI will be available at `http://localhost:8501`

### Option 2: Using Docker directly

```bash
# Build the image
docker build -t bank-marketing-ui .

# Run the container
docker run -d \
  --name bank-marketing-predictor \
  -p 8501:8501 \
  -v $(pwd)/src/models:/app/src/models:ro \
  -v $(pwd)/config:/app/config:ro \
  bank-marketing-ui

# View logs
docker logs -f bank-marketing-predictor

# Stop the container
docker stop bank-marketing-predictor
docker rm bank-marketing-predictor
```

## Important Notes

### Models and Preprocessor

The Docker container expects:
- **Models**: Located in `src/models/*.pkl` (mounted as read-only volume)
- **Preprocessor**: Located in `src/models/preprocessor.pkl` (mounted as read-only volume)
- **Config**: Located in `config/config.yaml` (mounted as read-only volume)

If you haven't trained models yet, you need to:

1. **Train models first** (outside Docker):
   ```bash
   # Preprocess data
   python -m src.preprocessing.main
   
   # Train models (at least one)
   python -m src.training.train_baseline
   # ... or other training scripts
   ```

2. **Then run Docker** - the models will be mounted into the container

### Volume Mounts

The docker-compose.yml mounts these directories as volumes:
- `./src/models` → Contains all model files (.pkl) including preprocessor.pkl
- `./config` → Contains config.yaml

This allows you to:
- Update models without rebuilding the image
- Keep models outside the container (they're gitignored)
- Easily update configuration

### Building Without Models

If you want to build a Docker image that includes models (not recommended for large files):

1. Copy your trained models to `src/models/`
2. Copy your preprocessor to `src/preprocessing/`
3. Build the image - models will be baked into the image

**Note**: This increases image size significantly. Using volume mounts is preferred.

## Customization

### Change Port

Edit `docker-compose.yml`:
```yaml
ports:
  - "8080:8501"  # Change 8080 to your desired port
```

Or with Docker:
```bash
docker run -p 8080:8501 ...
```

### Environment Variables

You can set Streamlit environment variables:
```yaml
environment:
  - STREAMLIT_SERVER_PORT=8501
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0
  - STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

## Troubleshooting

### Container won't start

1. Check if port 8501 is already in use:
   ```bash
   # Linux/Mac
   lsof -i :8501
   
   # Windows
   netstat -ano | findstr :8501
   ```

2. Check container logs:
   ```bash
   docker-compose logs
   # or
   docker logs bank-marketing-predictor
   ```

### Models not loading

1. Verify models exist:
   ```bash
   ls -la src/models/*.pkl
   ```

2. Check file permissions (should be readable)

3. Verify volume mounts in docker-compose.yml

### Preprocessor not found

1. Run preprocessing:
   ```bash
   python -m src.preprocessing.main
   ```

2. Verify `src/models/preprocessor.pkl` exists

## Production Deployment

For production, consider:

1. **Use a reverse proxy** (nginx, Traefik) in front of Streamlit
2. **Set up SSL/TLS** certificates
3. **Use environment-specific configs**
4. **Monitor container health** (healthcheck is already configured)
5. **Set resource limits** in docker-compose.yml:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 4G
   ```

## Clean Up

```bash
# Remove container and volumes
docker-compose down -v

# Remove image
docker rmi bank-marketing-ui

# Remove all unused Docker resources
docker system prune -a
```
