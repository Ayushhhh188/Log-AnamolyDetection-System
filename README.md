# Log Anomaly Detection System

A real-time log anomaly detection system built around a deep learning pipeline. Ingests raw system logs, runs them through an autoencoder-based anomaly detector, classifies severity, persists results to MongoDB, and streams everything live to a terminal monitor.

---

## Demo
### Click Below to See Project Demo ▶️
[![Watch the demo](https://github.com/Ayushhhh188/Log-AnamolyDetection-System/blob/0b571c837af516c4ce02a78922f9e3a8cc1dfbac/simulation/Screenshot%202026-05-20%20011304.png)](https://youtu.be/m31Yegi6grk)
> Clone the repo, add your MongoDB URI, run one command. Everything works.

```bash
git clone https://github.com/Ayushhhh188/Log-AnamolyDetection-System.git
cd Log-AnamolyDetection-System
cp .env.example .env        # fill in your MONGO_URI
docker-compose up backend   # terminal 1
docker-compose run --rm monitor  # terminal 2
```

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [ML Pipeline](#ml-pipeline)
- [Setup — With Docker](#setup--with-docker)
- [Setup — Without Docker](#setup--without-docker)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Terminal Monitor](#terminal-monitor)
- [Configuration](#configuration)
- [Design Decisions](#design-decisions)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

Most anomaly detection demos either fake the detection (pre-labelling logs before "predicting" them) or wrap everything in a frontend that obscures what the system actually does. This project does neither.

Raw, unlabelled Hadoop system logs are generated and passed through a real DL inference pipeline. The model assigns an anomaly score based on reconstruction error, labels each log as normal, suspicious, or critical, and the result is stored in MongoDB and streamed live to the terminal.

**Stack:** Python · FastAPI · WebSocket · TensorFlow/Keras · Scikit-learn · MongoDB · NumPy · Pandas · Docker

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Log Sources                              │
│              (simulated Hadoop-style raw logs)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  unlabelled log entries
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                             │
│                                                                 │
│   POST /logs/predict          — single log inference            │
│   POST /logs/batch-predict    — batch inference                 │
│   POST /simulation/start      — start live stream               │
│   POST /simulation/stop       — stop stream                     │
│   WS   /ws/logs               — WebSocket live feed             │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
              ▼                            ▼
┌─────────────────────────┐   ┌────────────────────────────────┐
│     DL Pipeline         │   │         MongoDB Atlas          │
│                         │   │                                │
│  1. Feature extraction  │   │  db: logs                      │
│  2. Autoencoder         │   │  collection: predicted_logs    │
│     reconstruction      │   │                                │
│  3. Isolation Forest    │   │  Stores every prediction with  │
│     scoring             │   │  score, label, severity,       │
│  4. Severity threshold  │   │  timestamp, component          │
│     classification      │   └────────────────────────────────┘
└─────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Terminal Monitor (monitor.py)                 │
│                                                                 │
│   TIME          LVL    SEV       SCORE    COMPONENT   CONTENT  │
│   09:14:32.441  INFO     OK      0.031    DataNode    Block...  │
│   09:14:33.012  WARN    WARN     0.463    FSNames..   Slow...   │
│   09:14:33.445  ERROR   CRIT     0.821    NameNode    Lost...   │
│                                                                 │
│   TOTAL:152 | WARN:10 | CRIT:8 | RATE:11.8% | [P] [C] [Q]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
Logs-Anamoly-Detection-System-Term/
│
├── backend/                        # FastAPI application
│   ├── models/
│   │   └── schema.py               # Pydantic request/response models
│   ├── routes/
│   │   └── log.py                  # /logs/* endpoints
│   ├── services/
│   │   └── predictor.py            # Bridge between API and DL pipeline
│   ├── db.py                       # MongoDB connection
│   └── app.py                      # FastAPI app, WebSocket, simulation control
│
├── DL/                             # Deep learning pipeline
│   ├── autoencoder/                # Autoencoder model definition & weights
│   ├── isolation_forest/           # Isolation Forest model & weights
│   └── pipeline/
│       └── batch_inference.py      # predict_batch() — core inference entry point
│
├── simulation/
│   └── log_generator.py            # LogSimulator — generates realistic log entries
│
├── data/
│   ├── raw/                        # Raw HDFS log dataset (training source)
│   └── scripts/                    # Data preprocessing scripts
│
├── outputs/
│   ├── models/                     # Saved model checkpoints
│   └── logs/                       # Training logs
│
├── monitor.py                      # Terminal monitor — runs in PowerShell/bash/Docker
├── Dockerfile                      # Backend container image
├── Dockerfile.monitor              # Monitor container image
├── docker-compose.yml              # Orchestrates backend + monitor together
├── .dockerignore                   # Excludes venv, cache, secrets from image
├── debug_import.py                 # Import diagnostic tool
├── requirements.txt
├── .env.example
└── README.md
```

---

## How It Works

### 1. Log Generation

`monitor.py` generates raw, unlabelled log entries structured exactly like real Hadoop system logs:

```python
{
    "timestamp":  "2026-04-09 09:14:32.441",
    "process_id": "DataNode",
    "log_level":  "WARN",
    "component":  "org.apache.hadoop.hdfs.server.datanode.DataNode",
    "content":    "Lost contact with NameNode /192.168.1.4 after 9 missed heartbeats"
}
```

No type label. No pre-assigned severity. The model decides.

### 2. Inference Pipeline

Each batch is passed to `predict_batch()` in `DL/pipeline/batch_inference.py`:

- Logs are converted to a DataFrame
- Features are extracted and encoded
- The autoencoder attempts to reconstruct each log
- High reconstruction error → the log is anomalous
- Isolation Forest provides a secondary anomaly score
- Scores are combined and thresholded into `normal / suspicious / critical`

### 3. Result

```python
{
    "anomaly_score":        0.821,   # 0.0 = normal, 1.0 = anomalous
    "reconstruction_error": 0.634,   # raw autoencoder output
    "label":                -1,      # 1 = normal, -1 = anomaly
    "severity":             "critical"
}
```

### 4. Storage & Streaming

Results are stored in MongoDB and broadcast over WebSocket simultaneously.

---

## ML Pipeline

### Model: Autoencoder

Trained exclusively on normal log patterns from the HDFS dataset. It learns to reconstruct normal logs with low error. When it encounters an anomalous log, reconstruction error spikes — that spike is the anomaly signal.

This is an unsupervised approach. No labelled anomaly data is required during training, which is realistic for real security environments where anomalies are rare and often unknown in advance.

### Model: Isolation Forest

Provides a complementary score by isolating anomalies through random feature partitioning. Anomalous points are isolated faster (shorter path length) than normal ones.

### Sensitivity Levels

| Sensitivity | Threshold  | Use case |
|-------------|------------|----------|
| `low`       | Permissive | Reduce false positives |
| `normal`    | Balanced   | General monitoring |
| `high`      | Strict     | Flag anything unusual |

### Training Data

The HDFS (Hadoop Distributed File System) log dataset — a standard benchmark in log-based anomaly detection research. Contains real system logs from a 203-node Hadoop cluster on Amazon EC2.

---

## Setup — With Docker

Docker is the recommended way. No Python installation, no venv, no path issues — one command runs everything.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- A MongoDB Atlas account (free tier works) — [create one here](https://www.mongodb.com/cloud/atlas)

### Steps

```bash
# 1. Clone
git clone https://github.com/Ayushhhh188/Log-AnamolyDetection-System.git
cd Log-AnamolyDetection-System

# 2. Set up environment
cp .env.example .env
# Open .env and fill in your MONGO_URI

# 3. Build images (first time only — takes 5-10 min)
docker-compose build

# 4. Start backend (Terminal 1)
docker-compose up backend

# 5. Run monitor (Terminal 2)
docker-compose run --rm monitor
```

### What Docker does here

- `Dockerfile` — builds the backend image: installs Python 3.11, all pip dependencies, copies your code, starts uvicorn
- `Dockerfile.monitor` — builds the monitor image: same deps, runs `monitor.py` instead
- `docker-compose.yml` — orchestrates both: backend starts first, monitor waits for it, volumes mount your model weights from your machine into the containers
- Model weights stay on your machine and are mounted as volumes — they don't get baked into the image

### Different modes

```bash
# DDoS simulation
docker-compose run --rm monitor python monitor.py --mode ddos

# High sensitivity
docker-compose run --rm monitor python monitor.py --sensitivity high

# No database
docker-compose run --rm monitor python monitor.py --no-db
```

---

## Setup — Without Docker

```bash
# 1. Clone
git clone https://github.com/Ayushhhh188/Log-AnamolyDetection-System.git
cd Log-AnamolyDetection-System

# 2. Virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Environment variables
cp .env.example .env
# Edit .env with your MONGO_URI

# 5. Create __init__.py files if missing
New-Item -ItemType File -Force "DL\__init__.py"
New-Item -ItemType File -Force "DL\pipeline\__init__.py"
New-Item -ItemType File -Force "simulation\__init__.py"
New-Item -ItemType File -Force "backend\__init__.py"
New-Item -ItemType File -Force "backend\routes\__init__.py"
New-Item -ItemType File -Force "backend\services\__init__.py"
New-Item -ItemType File -Force "backend\models\__init__.py"
```

---

## Running the Project

Always run from the project root.

### Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

API available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

### Terminal Monitor

```bash
python monitor.py                          # random mode
python monitor.py --mode ddos              # DDoS simulation
python monitor.py --sensitivity high       # stricter detection
python monitor.py --batch 10              # larger batches
python monitor.py --no-db                 # skip MongoDB
```

### Monitor Controls

| Key | Action |
|-----|--------|
| `P` | Pause / resume stream |
| `C` | Clear log window |
| `Q` | Quit |

---

## API Reference

### Simulation Control

```
POST /simulation/start?mode=random    Start streaming (random mode)
POST /simulation/start?mode=ddos      Start streaming (DDoS mode)
POST /simulation/stop                 Stop streaming
```

### Log Prediction

```
POST /logs/predict?sensitivity=low         Predict single log
POST /logs/batch-predict?sensitivity=low   Predict batch
POST /logs/simulate                        Generate + predict logs
```

**Request body:**

```json
{
  "timestamp":  "2026-04-09 09:14:32.441",
  "process_id": "DataNode",
  "log_level":  "WARN",
  "component":  "org.apache.hadoop.hdfs.server.datanode.DataNode",
  "content":    "Lost contact with NameNode after 9 missed heartbeats"
}
```

**Response:**

```json
{
  "anomaly_score":        0.821,
  "reconstruction_error": 0.634,
  "label":                -1,
  "severity":             "critical"
}
```

### WebSocket

```
WS /ws/logs    Live log stream
```

### Utility

```
GET /health    System status, active connections, simulation state
GET /docs      Swagger UI
GET /          API info
```

---

## Terminal Monitor

The monitor is a standalone Python script — no browser, no frontend. Runs in PowerShell, CMD, bash, or inside Docker.

This matches how real security tooling works. Suricata, Wazuh agents, and Splunk forwarders are all terminal processes. They don't have UIs — they write to stdout and are monitored over SSH on remote servers.

```
log-anomaly-detection  mode:random  sens:low  total:152  warn:10  crit:8  rate:11.8%  db:on  [LIVE]
────────────────────────────────────────────────────────────────────────────────────────────────────
TIME            LVL    SEV       SCORE    COMPONENT              CONTENT
────────────────────────────────────────────────────────────────────────────────────────────────────
09:14:32.441    INFO     OK      0.031    DataNode               Checkpoint done
09:14:33.012    WARN    WARN     0.463    FSNamesystem           Slow BlockReceiver: 4821ms
09:14:33.445    ERROR   CRIT     0.821    NameNode               Lost contact: 9 missed heartbeats
────────────────────────────────────────────────────────────────────────────────────────────────────
[p] pause/resume  [c] clear  [q] quit
```

The `SCORE` column is the model's raw `anomaly_score`. The label and severity come entirely from `predict_batch()` — nothing is pre-determined.

---

## Configuration

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--mode` | `random` | `random`, `ddos` | Traffic pattern |
| `--sensitivity` | `low` | `low`, `normal`, `high` | Detection threshold |
| `--batch` | `5` | any int | Logs per inference call |
| `--no-db` | off | flag | Disable MongoDB persistence |

---

## Design Decisions

**Why Docker?**
Eliminates the "works on my machine" problem entirely. Model weights are mounted as volumes so they don't inflate the image size. Anyone with Docker Desktop can run the full project in under 10 minutes.

**Why no frontend?**
A browser UI is the wrong abstraction for a log monitoring system. Real SOC tools run as terminal processes. The monitor is SSH-able, scriptable, and has zero browser dependency.

**Why unsupervised learning?**
In real environments, anomalies are rare and often novel — you can't label what you don't know about. An autoencoder trained only on normal patterns detects deviations without requiring a labelled anomaly dataset.

**Why MongoDB over relational?**
Log schemas vary. MongoDB's flexible document model handles variable log structures without schema migrations. For time-series queries at scale, Elasticsearch or ClickHouse would be the next step.

**Why batch inference in the monitor?**
Calling the HTTP API per log adds network overhead. The monitor calls `predict_batch()` directly, keeping inference tight and latency low.

---

## Limitations & Future Work

**Current limitations:**
- Model trained on HDFS logs — performance degrades on logs from different systems
- No authentication on API endpoints
- Simulation generates synthetic logs; production would ingest from Filebeat or Fluentd
- Single-node — no horizontal scaling

**Logical next steps:**
- **Kafka** between ingestion and inference — decouples producers from consumers, handles traffic spikes
- **Horizontal scaling** — multiple inference workers pulling from Kafka partitions
- **Alerting** — POST to Slack/PagerDuty when `severity == critical`
- **Model retraining pipeline** — adapt to new log patterns over time

---

## Requirements

See `requirements.txt` for full pinned list. Key dependencies:

```
fastapi
uvicorn
websockets
pymongo
python-dotenv
tensorflow
scikit-learn
numpy
pandas
```

---

