from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import numpy as np
from typing import List, Optional

from backend.routes.log import router as log_router
from backend.db import get_collection
from log_generator import LogSimulator


# App Initialization
app = FastAPI(
    title="Log Anomaly Detection API",
    description="Real-time log streaming + anomaly detection",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing routes
app.include_router(log_router)

# WebSocket Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"Client connected ({len(self.active_connections)})")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"❌ Client disconnected ({len(self.active_connections)})")

    async def broadcast(self, message: dict):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_text(json.dumps(message))
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


# Simulator (fallback)
import random, datetime

simulator = LogSimulator(sensitivity="low")

LOG_TEMPLATES = {
    "normal": [
        ("INFO",  "auth.service",     "User '{user}' logged in successfully from {ip}"),
        ("INFO",  "db.pool",          "Connection pool: {num}/50 active, {ms}ms avg wait"),
        ("INFO",  "cache.redis",      "Cache hit ratio: {pct}% over last 60s"),
        ("INFO",  "api.gateway",      "GET /api/v1/{endpoint} 200 OK in {ms}ms"),
        ("INFO",  "scheduler",        "Cron job '{job}' completed in {ms}ms"),
        ("INFO",  "storage.s3",       "Uploaded {file} ({size}MB) to bucket {bucket}"),
        ("INFO",  "auth.service",     "JWT issued for user {user}, expires in 3600s"),
        ("INFO",  "metrics",          "CPU: {cpu}%  MEM: {mem}%  DISK: {disk}%"),
        ("INFO",  "backup.service",   "Incremental backup done — {num} files synced"),
        ("INFO",  "cdn.edge",         "Cache purged for zone {zone}, {num} assets invalidated"),
        ("INFO",  "health.check",     "All {num} services healthy — uptime {uptime}h"),
        ("INFO",  "search.index",     "Indexed {num} documents in {ms}ms"),
        ("INFO",  "tls.manager",      "Certificate renewed for {domain}, valid {days} days"),
        ("INFO",  "queue.worker",     "Processed {num} jobs from '{queue}' in {ms}ms"),
        ("INFO",  "email.worker",     "Dispatched {num} emails via {provider}"),
    ],
    "suspicious": [
        ("WARNING", "auth.service",   "Failed login for '{user}' — attempt {num}/5 from {ip}"),
        ("WARNING", "api.gateway",    "Rate limit at {pct}% — {num} req/min on /api/{endpoint}"),
        ("WARNING", "db.query",       "Slow query: {ms}ms on table '{table}'"),
        ("WARNING", "firewall",       "Port scan from {ip} on ports {port1}-{port2}"),
        ("WARNING", "memory.monitor", "Heap usage at {pct}% — GC pressure increasing"),
        ("WARNING", "network",        "Unusual outbound traffic to {ip}: {size}MB/60s"),
        ("WARNING", "auth.service",   "Password reset requested {num}x for {user} in 10min"),
        ("WARNING", "api.gateway",    "Malformed JWT from {ip} — request rejected"),
        ("WARNING", "disk.monitor",   "Disk I/O wait at {pct}% on {device}"),
        ("WARNING", "session",        "Concurrent session limit reached for '{user}'"),
    ],
    "critical": [
        ("ERROR", "ddos.shield",   "DDoS detected — {num} req/sec from {subnet}"),
        ("ERROR", "auth.service",  "Brute force lockout: {num} accounts from {ip}"),
        ("ERROR", "db.primary",    "Replication lag {ms}ms — failover threshold exceeded"),
        ("ERROR", "api.gateway",   "Circuit breaker OPEN for '{service}' — 0% success rate"),
        ("ERROR", "network.ids",   "Intrusion signature {sig} matched from {ip}"),
        ("ERROR", "disk.monitor",  "Disk {device} at {pct}% capacity — write failures imminent"),
        ("ERROR", "container",     "Pod '{pod}' crash-looped {num}x in last 5 minutes"),
        ("ERROR", "ssl.monitor",   "Certificate for {domain} EXPIRED — HTTPS failing"),
        ("ERROR", "queue",         "Dead letter queue overflow: {num} unprocessed messages"),
        ("ERROR", "memory",        "OOM killer on '{node}' — pid {pid} terminated"),
    ],
}

_FD = {
    "user":     ["alice","bob","admin","deploy_bot","svc_account","root","jenkins","carol"],
    "ip":       ["192.168.1.{n}","10.0.{n}.{m}","203.0.113.{n}","172.16.{n}.1"],
    "endpoint": ["users","orders","auth/token","products","analytics","payments","search"],
    "job":      ["db-cleanup","report-gen","cache-warm","audit-log","snapshot","reindex"],
    "file":     ["backup_{n}.tar.gz","export_{n}.csv","model_{n}.pkl","dump_{n}.sql"],
    "bucket":   ["prod-assets","ml-models","user-uploads","audit-logs","media-cdn"],
    "zone":     ["us-east-1","eu-west-2","ap-south-1","us-west-2"],
    "domain":   ["api.internal","app.prod","auth.service","cdn.edge","payments.svc"],
    "table":    ["users","orders","sessions","audit_logs","events","transactions"],
    "device":   ["/dev/sda","/dev/nvme0n1","/dev/sdb","/dev/xvda"],
    "service":  ["payment-svc","auth-svc","notification-svc","search-svc","billing-svc"],
    "sig":      ["ET.SCAN.NMAP","EXPLOIT.CVE-2024-1234","MALWARE.C2","SCAN.MASSCAN"],
    "subnet":   ["10.0.0.0/24","192.168.0.0/16","203.0.113.0/24","172.16.0.0/12"],
    "pod":      ["api-deployment-{n}","worker-{n}","cache-{n}","consumer-{n}"],
    "node":     ["node-01","node-02","worker-03","master-01"],
    "queue":    ["email-queue","webhook-queue","analytics-queue","payment-queue"],
    "provider": ["SendGrid","SES","Mailgun","Postmark"],
}

def _r(key: str) -> str:
    v = random.choice(_FD[key])
    return v.format(n=random.randint(1,254), m=random.randint(1,254))

def generate_varied_log(log_type: str) -> dict:
    level, component, msg_tpl = random.choice(LOG_TEMPLATES[log_type])
    msg = msg_tpl.format(
        user=_r("user"), ip=_r("ip"), endpoint=_r("endpoint"), job=_r("job"),
        file=_r("file"), bucket=_r("bucket"), zone=_r("zone"), domain=_r("domain"),
        table=_r("table"), device=_r("device"), service=_r("service"), sig=_r("sig"),
        subnet=_r("subnet"), pod=_r("pod"), node=_r("node"), queue=_r("queue"),
        provider=_r("provider"),
        num=random.randint(1,9999), pct=random.randint(50,99),
        ms=random.randint(10,8000), size=round(random.uniform(0.1,500),1),
        cpu=random.randint(10,95), mem=random.randint(20,98), disk=random.randint(5,90),
        uptime=random.randint(1,720), days=random.randint(1,90),
        port1=random.randint(1024,8000), port2=random.randint(8001,65535),
        pid=random.randint(1000,99999),
    )
    now = datetime.datetime.now()
    return {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond//1000:03d}",
        "level":     level,
        "component": component,
        "message":   msg,
    }


# Simulation State
class SimulationState:
    def __init__(self):
        self.mode: Optional[str] = None
        self.running: bool = False

state = SimulationState()

LEVEL_MAP = {
    "normal":     "INFO",
    "suspicious": "WARNING",
    "critical":   "ERROR",
}


# DB helper
def store_log(payload: dict):
    try:
        # get_collection() returns the collection object directly
        collection = get_collection()
        result = collection.insert_one(payload.copy())
        print(f"Stored [{payload['type']}] log → _id={result.inserted_id}")
    except Exception as e:
        print(f"DB insert error: {e}")


# Real-Time Log Streamer
async def stream_logs():
    while True:
        if not state.running or not state.mode:
            await asyncio.sleep(0.5)
            continue

        try:
            if state.mode == "random":
                log_type = np.random.choice(
                    ["normal", "suspicious", "critical"],
                    p=[0.85, 0.10, 0.05]
                )
            else:  # ddos
                log_type = np.random.choice(
                    ["critical", "suspicious"],
                    p=[0.7, 0.3]
                )

            log = generate_varied_log(log_type)

            payload = {
                "timestamp": log["timestamp"],
                "level":     log["level"],
                "component": log["component"],
                "message":   log["message"],
                "type":      log_type,
                "source":    "websocket_stream",
            }

            print(f"Broadcasting: type={log_type} level={payload['level']}")

            # 1. Send to frontend
            await manager.broadcast(payload)

            # 2. Save to MongoDB in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, store_log, payload)

            await asyncio.sleep(0.2 if state.mode == "ddos" else 0.5)

        except Exception as e:
            print(f"Streaming error: {e}")
            await asyncio.sleep(1)

# Startup
@app.on_event("startup")
async def startup():
    print("Starting log streaming service...")
    asyncio.create_task(stream_logs())


# WebSocket endpoint
@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Simulation Control
control_router = APIRouter(prefix="/simulation", tags=["Simulation"])

@control_router.post("/start")
async def start_simulation(mode: str = "random"):
    if mode not in ["random", "ddos"]:
        return {"error": "Invalid mode. Use 'random' or 'ddos'."}
    state.mode = mode
    state.running = True
    print(f"Simulation started: {mode}")
    return {"status": "started", "mode": mode}

@control_router.post("/stop")
async def stop_simulation():
    state.running = False
    state.mode = None
    print("Simulation stopped")
    return {"status": "stopped"}

app.include_router(control_router)

# Root
@app.get("/")
def root():
    return {
        "message": "Log Anomaly Detection API is running",
        "features": [
            "Real-time WebSocket log streaming",
            "Anomaly detection (single + batch)",
            "MongoDB persistence — all log types",
            "Simulation modes: random / ddos",
        ]
    }
# Health Check
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "log-anomaly-detection",
        "active_connections": len(manager.active_connections),
        "simulation_running": state.running,
        "mode": state.mode,
    }