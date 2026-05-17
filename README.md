# Log Anomaly Detection System

A real-time log anomaly detection system built around a deep learning pipeline. Ingests raw system logs, runs them through an autoencoder-based anomaly detector, classifies severity, persists results to MongoDB, and streams everything live to a terminal monitor — no frontend, no browser.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [ML Pipeline](#ml-pipeline)
- [Setup](#setup)
- [Running the Project](#running-the-project)
- [API Reference](#api-reference)
- [Terminal Monitor](#terminal-monitor)
- [Configuration](#configuration)
- [Design Decisions](#design-decisions)
- [Limitations & Future Work](#limitations--future-work)

---

## Overview

Most anomaly detection demos either fake the detection (pre-labelling logs before "predicting" them) or wrap everything in a flashy frontend that obscures what the system actually does. This project does neither.

Raw, unlabelled Hadoop system logs are generated and passed through a real DL inference pipeline. The model assigns an anomaly score based on reconstruction error, labels each log as normal, suspicious, or critical, and the result is stored in MongoDB and streamed live to the terminal.

**Stack:** Python · FastAPI · WebSocket · TensorFlow/Keras · Scikit-learn · MongoDB · NumPy · Pandas

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
├── monitor.py                      # Terminal monitor — runs in PowerShell/bash
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

No type label. No pre-assigned severity. The model doesn't know what it's looking at — it has to figure it out.

### 2. Inference Pipeline

Each batch is passed to `predict_batch()` in `DL/pipeline/batch_inference.py`:

- Logs are converted to a DataFrame
- Features are extracted and encoded
- The autoencoder attempts to reconstruct each log
- High reconstruction error → the log is anomalous (it doesn't look like anything the model learned as "normal")
- Isolation Forest provides a secondary anomaly score
- Scores are combined and thresholded into `normal / suspicious / critical`

### 3. Result

Each log comes back enriched:

```python
{
    "anomaly_score":        0.821,   # 0.0 = definitely normal, 1.0 = definitely anomalous
    "reconstruction_error": 0.634,   # raw autoencoder output
    "label":                -1,      # 1 = normal, -1 = anomaly
    "severity":             "critical"
}
```

### 4. Storage & Streaming

Results are stored in MongoDB and broadcast over WebSocket simultaneously. The terminal monitor receives them and renders live.

---

## ML Pipeline

### Model: Autoencoder

The autoencoder is trained exclusively on normal log patterns from the HDFS dataset. It learns to reconstruct normal logs with low error. When it encounters an anomalous log, reconstruction error spikes — that spike is the anomaly signal.

This is an unsupervised approach. No labelled anomaly data is required during training, which is realistic for real security environments where anomalies are rare and often unknown in advance.

### Model: Isolation Forest

Isolation Forest provides a complementary score. It isolates anomalies by randomly partitioning the feature space — anomalous points are isolated faster (shorter path length in the tree) than normal points.

### Sensitivity Levels

Three sensitivity modes control the thresholds:

| Sensitivity | Anomaly threshold | Use case |
|-------------|-------------------|----------|
| `low`       | Permissive        | Reduce false positives, catch only clear anomalies |
| `normal`    | Balanced          | General monitoring |
| `high`      | Strict            | High-security environments, flag anything unusual |

### Training Data

The HDFS (Hadoop Distributed File System) log dataset — a standard benchmark dataset in log-based anomaly detection research. Contains real system logs from a 203-node Hadoop cluster running on Amazon EC2.

---

## Setup

### Prerequisites

- Python 3.10+
- MongoDB Atlas account (free tier works)
- Git

### Installation

```bash
# Clone the repo
git clone https://github.com/Ayushhhh188/Log-AnamolyDetection-System.git
cd Log-AnamolyDetection-System

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

```
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
```

### `__init__.py` Files

Python requires these to treat directories as importable modules. If any are missing, create them:

```powershell
# Windows PowerShell
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

Always run from the project root so path resolution works correctly.

### Start the Backend

```bash
uvicorn backend.app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### Run the Terminal Monitor

Open a second terminal in the same directory:

```bash
# Random simulation — mix of normal and anomalous logs
python monitor.py

# DDoS attack simulation — high anomaly rate
python monitor.py --mode ddos

# Stricter anomaly detection
python monitor.py --sensitivity high

# Larger batch size for higher throughput
python monitor.py --batch 10

# Run without MongoDB (no storage)
python monitor.py --no-db
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
POST /simulation/start?mode=random    Start log streaming (random mode)
POST /simulation/start?mode=ddos      Start log streaming (DDoS mode)
POST /simulation/stop                 Stop streaming
```

### Log Prediction

```
POST /logs/predict?sensitivity=low         Predict single log
POST /logs/batch-predict?sensitivity=low   Predict batch of logs
POST /logs/simulate                        Generate + predict logs
```

**Single log request body:**

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
WS /ws/logs    Live log stream — connects and receives JSON log entries
```

Each message:

```json
{
  "timestamp":  "09:14:32.441",
  "level":      "ERROR",
  "component":  "ddos.shield",
  "message":    "DDoS detected — 4821 req/sec from 10.0.0.0/24",
  "type":       "critical",
  "source":     "websocket_stream"
}
```

### Health Check

```
GET /health    Returns system status and active connection count
GET /          API info
```

---

## Terminal Monitor

The monitor is a standalone Python script — no browser, no frontend, no Node.js. It runs directly in PowerShell, CMD, or any Linux/macOS terminal over SSH.

This matches how real security tooling works. Tools like Suricata, Wazuh agents, and Splunk forwarders are all terminal processes. They don't have UIs — they write to stdout, sockets, or log files and are typically monitored over SSH on remote infrastructure.

**What the display shows:**

```
log-anomaly-detection  mode:random  sens:low  total:152  warn:10  crit:8  rate:11.8%  db:on  2026-05-16T09:14:32Z  [LIVE]
--------------------------------------------------------------------------------
TIME            LVL    SEV       SCORE    COMPONENT              CONTENT
--------------------------------------------------------------------------------
09:14:32.441    INFO     OK      0.031    DataNode               Checkpoint done
09:14:33.012    WARN    WARN     0.463    FSNamesystem           Slow BlockReceiver: 4821ms
09:14:33.445    ERROR   CRIT     0.821    NameNode               Lost contact: 9 missed heartbeats
--------------------------------------------------------------------------------
[p] pause/resume  [c] clear  [q] quit
```

The `SCORE` column is the model's raw `anomaly_score`. Everything else — `SEV`, coloured output, the label — comes from what `predict_batch()` returns. Nothing is pre-determined.

---

## Configuration

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--mode` | `random` | `random`, `ddos` | Simulation traffic pattern |
| `--sensitivity` | `low` | `low`, `normal`, `high` | Model detection threshold |
| `--batch` | `5` | any int | Logs per inference call |
| `--no-db` | off | flag | Disable MongoDB persistence |

---

## Design Decisions

**Why no frontend?**
A browser UI is the wrong abstraction for a log monitoring system. Real SOC tools (Suricata, Zeek, Wazuh, Splunk forwarders) run as terminal processes. The monitor is SSH-able, scriptable, and has no browser dependency — you can run it on a remote server with no display.

**Why unsupervised learning?**
In real environments, anomalies are rare and often novel. You can't label what you don't know about. An autoencoder trained only on normal patterns can detect deviations without requiring a labelled anomaly dataset.

**Why batch inference in the monitor instead of the `/logs/predict` API?**
Latency. Calling the HTTP API per log adds network overhead. The monitor calls `predict_batch()` directly from the same process, keeping inference tight.

**Why MongoDB over a relational database?**
Log schemas vary. MongoDB's flexible document model handles variable log structures without schema migrations. For a time-series query pattern at scale, a purpose-built store like Elasticsearch or ClickHouse would be next.

---

## Limitations & Future Work

**Current limitations:**

- Model weights are not in the repository — must be trained locally before running inference
- No authentication on API endpoints
- Simulation generates synthetic logs; a production version would ingest from real log forwarders (Filebeat, Fluentd)
- Single-node — no horizontal scaling of the inference worker

**Logical next steps for production:**

- **Kafka / Redis Streams** between log ingestion and inference — decouples producers from consumers, handles traffic spikes without dropping logs
- **Containerise with Docker** — `Dockerfile` + `docker-compose.yml` to package backend + model weights together, one-command setup anywhere
- **Horizontal scaling** — multiple inference workers pulling from Kafka partitions, same model weights loaded per instance
- **Alerting integration** — POST to Slack / PagerDuty when `severity == critical` and anomaly rate exceeds threshold
- **Model retraining pipeline** — periodically retrain on new "confirmed normal" logs to adapt to system drift

---

## Requirements

Key dependencies — see `requirements.txt` for full list with pinned versions.

```
fastapi
uvicorn
websockets
pymongo
python-dotenv
tensorflow / keras
scikit-learn
numpy
pandas
rich
```

---

## License

MIT
