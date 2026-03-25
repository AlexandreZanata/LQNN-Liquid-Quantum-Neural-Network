# LQNN v2 - Setup Guide

## Prerequisites

### Hardware
- **GPU**: NVIDIA with 8GB+ VRAM (tested on RTX 4060)
- **RAM**: 32GB recommended
- **CPU**: Intel i7 13th gen or equivalent
- **Storage**: 10GB+ free (for model downloads)

### Software
- **OS**: Ubuntu 22.04+ (or any Linux with NVIDIA drivers)
- **Docker**: 24.0+ with Docker Compose v2
- **NVIDIA Container Toolkit**: For GPU passthrough to Docker

## Docker Setup (Recommended)

### 1. Install NVIDIA Container Toolkit

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Start the System

```bash
git clone git@github.com:AlexandreZanata/LQNN-Liquid-Quantum-Neural-Network.git
cd LQNN-Liquid-Quantum-Neural-Network
docker compose up -d --build
```

### 3. First Run

On first startup, the system will:
1. Download OpenCLIP ViT-B/32 (~400MB)
2. Download Phi-3.5-mini-instruct 4-bit (~2GB)
3. Initialize ChromaDB
4. Connect to MongoDB
5. Start the continuous training loop
6. Serve the web UI on port 8000

Models are cached in a Docker volume (`model-cache`) so they only download once.

### 4. Access the UI

- **Chat**: http://localhost:8000/chat
- **Training Dashboard**: http://localhost:8000/training
- **Health Check**: http://localhost:8000/health

## Local Development (Without Docker)

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start MongoDB (optional, for logging)

```bash
docker run -d --name lqnn-mongo -p 27017:27017 mongo:7
```

### 3. Run the System

```bash
python main_loop.py
```

### 4. Run Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

### 5. Run Linter

```bash
pip install ruff
ruff check lqnn/ tests/ ui/ --select=E,F,W --ignore=E501
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `MONGO_URI` | `mongodb://localhost:27017/lqnn` | MongoDB connection string |
| `MONGO_DB` | `lqnn` | MongoDB database name |
| `HF_HOME` | `models/cache` | Hugging Face model cache directory |
| `TORCH_HOME` | `models/cache` | PyTorch model cache directory |
| `CHROMA_DIR` | `data/chroma` | ChromaDB persistent storage |
| `HOST` | `0.0.0.0` | Web server bind address |
| `PORT` | `8000` | Web server port |

## Docker Volumes

| Volume | Mount | Purpose |
|--------|-------|---------|
| `model-cache` | `/app/models/cache` | AI model files (persists across rebuilds) |
| `mongo-data` | `/data/db` | MongoDB data |
| `./data` | `/app/data` | ChromaDB vectors and state |
| `./logs` | `/app/logs` | Application logs |

## Troubleshooting

### GPU Not Detected in Docker
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

### Models Download Slow
Models are downloaded from Hugging Face. If behind a firewall, set:
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### MongoDB Connection Failed
The system works without MongoDB -- logging is simply disabled. Check:
```bash
docker compose logs mongo
```

### ChromaDB Permission Error
Ensure `data/chroma` directory is writable:
```bash
mkdir -p data/chroma && chmod 777 data/chroma
```
