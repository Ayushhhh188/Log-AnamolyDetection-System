"""
monitor.py — log anomaly detection terminal monitor

Usage:
    python monitor.py
    python monitor.py --mode ddos
    python monitor.py --sensitivity high
    python monitor.py --batch 10
    python monitor.py --no-db

Controls:
    P  pause/resume
    C  clear
    Q  quit
"""

import argparse
import datetime
import os
import random
import signal
import sys
import threading
import time
from collections import deque
from typing import Any, Dict, List

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR       = os.getcwd()
PIPELINE_DIR   = os.path.join(BASE_DIR, "DL", "pipeline")
SIMULATION_DIR = os.path.join(BASE_DIR, "simulation")

for p in (PIPELINE_DIR, SIMULATION_DIR, BASE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ── pipeline ───────────────────────────────────────────────────────────────
try:
    from batch_inference import predict_batch
except ImportError as e:
    print(f"ERROR: cannot import predict_batch: {e}")
    sys.exit(1)

import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# ── mongodb ────────────────────────────────────────────────────────────────
MONGO_OK    = False
_collection = None
try:
    from pymongo import MongoClient
    _uri = os.getenv("MONGO_URI")
    if _uri:
        _mc = MongoClient(_uri, serverSelectionTimeoutMS=3000)
        _mc.server_info()
        _collection = _mc["logs"]["predicted_logs"]
        MONGO_OK = True
except Exception:
    pass

# ── raw log pool (no labels) ───────────────────────────────────────────────
_COMPONENTS = [
    "org.apache.hadoop.ipc.Server",
    "org.apache.hadoop.hdfs.server.datanode.DataNode",
    "org.apache.hadoop.hdfs.server.namenode.NameNode",
    "org.apache.hadoop.hdfs.server.namenode.FSNamesystem",
    "org.apache.hadoop.metrics2.impl.MetricsSystemImpl",
    "org.apache.hadoop.http.HttpServer2",
    "org.apache.hadoop.util.JvmPauseMonitor",
    "org.apache.hadoop.security.UserGroupInformation",
    "org.apache.hadoop.net.NetworkTopology",
    "org.apache.hadoop.hdfs.server.blockmanagement.BlockManager",
]

_PROCESS_IDS = [
    "DataNode", "NameNode", "SecondaryNameNode",
    "ResourceManager", "NodeManager", "JobHistoryServer",
]

_LEVELS = ["INFO", "INFO", "INFO", "INFO", "WARN", "ERROR"]

_MESSAGES = [
    "Scheduled snapshot period at 10 second(s)",
    "Block received from /192.168.1.{n}",
    "Finished writing block blk_{n}",
    "Http request log for http.requests.datanode is initialized",
    "DataNode registered with NameNode",
    "IPC Server Responder: starting",
    "Rolling edit logs",
    "Starting checkpoint at txid {n}",
    "Checkpoint done. New image checkpoint",
    "JVM heap memory: {n}MB used out of {n2}MB",
    "DataXceiver: Bytes read={n} written={n2}",
    "Slow BlockReceiver write data to mirror took {ms}ms",
    "Retrying connect to server: /{ip}:{port}. Already tried {n} time(s)",
    "Exception in receiveBlock for block blk_{n}",
    "Got exception while serving blk_{n} to /{ip}: java.io.IOException",
    "PipelineAck RPC time out: {n}",
    "An exception was caught while connecting to {ip}:{port}",
    "Failed to place enough replicas: target of {n} storage type",
    "Block blk_{n} is corrupt on {ip}, replacing replica",
    "Blacklisting node /{ip} due to repeated failures",
    "java.lang.OutOfMemoryError: GC overhead limit exceeded",
    "Too many open connections from /{ip}, rejecting",
    "Detected slow disk: write latency {ms}ms exceeds threshold",
    "Heartbeat from DataNode /{ip} received late by {n}ms",
    "Lost contact with DataNode /{ip} after {n} missed heartbeats",
    "Authentication failed for user '{user}' from {ip}: invalid credentials",
    "Unauthorized access attempt on /hdfs/{path} from {ip}",
    "DDoS-like traffic spike: {n} connections/sec from {subnet}",
    "Port scan from {ip}: {n} ports probed in {ms}ms",
    "Brute force login: {n} failed attempts for user '{user}'",
    "Anomalous data exfiltration: {n}MB sent to {ip}",
    "kernel: possible SYN flood on port {port}",
    "Triggering emergency replication of {n} under-replicated blocks",
    "HDFS is {pct}% full, approaching capacity threshold",
]

_DDOS_MESSAGES = [
    "DDoS-like traffic spike: {n} connections/sec from {subnet}",
    "Port scan from {ip}: {n} ports probed in {ms}ms",
    "Brute force login: {n} failed attempts for user '{user}'",
    "Too many open connections from /{ip}, rejecting",
    "kernel: possible SYN flood on port {port}",
    "Blacklisting node /{ip} due to repeated failures",
    "Anomalous data exfiltration: {n}MB sent to {ip}",
    "Authentication failed for user '{user}' from {ip}: invalid credentials",
    "Lost contact with DataNode /{ip} after {n} missed heartbeats",
    "Unauthorized access attempt on /hdfs/{path} from {ip}",
]

_FD = {
    "ip":     ["192.168.1.{n}", "10.0.{n}.{m}", "172.16.{n}.1", "203.0.113.{n}"],
    "user":   ["hdfs", "yarn", "mapred", "admin", "root", "alice"],
    "subnet": ["10.0.0.0/24", "192.168.0.0/16", "203.0.113.0/24"],
    "path":   ["user/hdfs", "tmp/hadoop-yarn", "data/blocks", "logs/audit"],
}

def _r(key):
    v = random.choice(_FD[key])
    return str(v).format(n=random.randint(1, 254), m=random.randint(1, 254))

def make_log(ddos=False):
    pool = _DDOS_MESSAGES if (ddos and random.random() < 0.6) else _MESSAGES
    tpl  = random.choice(pool)
    msg  = tpl.format(
        n=random.randint(1, 999999), n2=random.randint(1, 999999),
        ms=random.randint(1, 9999),  pct=random.randint(50, 99),
        port=random.randint(1024, 65535),
        ip=_r("ip"), user=random.choice(_FD["user"]),
        path=random.choice(_FD["path"]), subnet=random.choice(_FD["subnet"]),
    )
    now = datetime.datetime.now()
    return {
        "timestamp":  now.strftime("%Y-%m-%d %H:%M:%S.") + f"{now.microsecond//1000:03d}",
        "process_id": random.choice(_PROCESS_IDS),
        "log_level":  random.choice(_LEVELS) if not ddos else random.choice(["WARN","ERROR","INFO"]),
        "component":  random.choice(_COMPONENTS),
        "content":    msg,
    }

# ── prediction ─────────────────────────────────────────────────────────────
def predict(raw: List[Dict], sensitivity: str) -> List[Dict]:
    df_in  = pd.DataFrame(raw)
    df_out = predict_batch(df_in, sensitivity=sensitivity)
    out = []
    for i in range(min(len(df_in), len(df_out))):
        r = df_in.iloc[i]
        p = df_out.iloc[i]
        out.append({
            "timestamp":  str(r.get("timestamp",  "")),
            "level":      str(r.get("log_level",  "")),
            "component":  str(r.get("component",  "")).split(".")[-1],
            "content":    str(r.get("content",    "")),
            "score":      float(p.get("anomaly_score",        0.0)),
            "label":      int(p.get("label",                  1)),
            "severity":   str(p.get("severity",               "normal")),
            "recon_err":  float(p.get("reconstruction_error", 0.0)),
        })
    return out

def store(results: List[Dict]):
    if _collection is None:
        return
    try:
        _collection.insert_many([{**r, "source": "cli_monitor"} for r in results])
    except Exception:
        pass

# ── state ──────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.logs       = deque(maxlen=500)
        self.total      = 0
        self.suspicious = 0
        self.critical   = 0
        self.running    = True
        self.paused     = False
        self.err        = ""
        self.lock       = threading.Lock()

state = State()

# ── worker ─────────────────────────────────────────────────────────────────
def worker(batch: int, sensitivity: str, mode: str):
    ddos  = (mode == "ddos")
    delay = 0.15 if ddos else 0.4
    while state.running:
        if state.paused:
            time.sleep(0.1)
            continue
        try:
            raw     = [make_log(ddos) for _ in range(batch)]
            results = predict(raw, sensitivity)
            threading.Thread(target=store, args=(results,), daemon=True).start()
            with state.lock:
                for r in results:
                    state.logs.append(r)
                    state.total += 1
                    if r["severity"] == "critical":   state.critical   += 1
                    elif r["severity"] == "suspicious": state.suspicious += 1
            state.err = ""
        except Exception as e:
            state.err = str(e)
        time.sleep(delay + random.uniform(0, 0.1))

# ── keyboard ───────────────────────────────────────────────────────────────
def keyboard():
    try:
        import msvcrt
        while state.running:
            if msvcrt.kbhit():
                ch = msvcrt.getwch().lower()
                if   ch == 'q': state.running = False
                elif ch == 'p': state.paused  = not state.paused
                elif ch == 'c':
                    with state.lock: state.logs.clear()
            time.sleep(0.05)
    except ImportError:
        import tty, termios
        fd  = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while state.running:
                ch = sys.stdin.read(1).lower()
                if   ch == 'q': state.running = False
                elif ch == 'p': state.paused  = not state.paused
                elif ch == 'c':
                    with state.lock: state.logs.clear()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ── display ────────────────────────────────────────────────────────────────
SEV_TAG = {"normal": "  OK  ", "suspicious": " WARN ", "critical": " CRIT "}

def render(mode: str, sensitivity: str, use_db: bool):
    W     = os.get_terminal_size().columns
    lines = os.get_terminal_size().lines

    out = []

    # header
    now     = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    anom    = state.suspicious + state.critical
    rate    = f"{anom/state.total*100:.1f}%" if state.total else "0.0%"
    db_str  = "db:on" if (use_db and MONGO_OK) else "db:off"
    status  = "PAUSED" if state.paused else "LIVE"
    hdr = (
        f"log-anomaly-detection  mode:{mode}  sens:{sensitivity}  "
        f"total:{state.total}  warn:{state.suspicious}  crit:{state.critical}  "
        f"rate:{rate}  {db_str}  {now}  [{status}]"
    )
    out.append(hdr[:W])
    out.append("-" * W)

    # column header
    out.append(f"{'TIME':<15} {'LVL':<5} {'SEV':<8} {'SCORE':<7} {'COMPONENT':<22} CONTENT")
    out.append("-" * W)

    # log rows
    rows_available = lines - 7  # header(2) + col header(2) + footer(3)
    with state.lock:
        visible = list(state.logs)[-rows_available:]

    for r in visible:
        ts    = r["timestamp"][11:23]
        lvl   = r["level"][:4]
        sev   = SEV_TAG.get(r["severity"], "      ")
        score = f"{r['score']:.3f}"
        comp  = r["component"][:22]
        # truncate content to fit terminal width
        used    = 15 + 1 + 5 + 1 + 8 + 1 + 7 + 1 + 22 + 1
        content = r["content"][:max(W - used - 1, 10)]
        out.append(f"{ts:<15} {lvl:<5} {sev:<8} {score:<7} {comp:<22} {content}")

    # pad blank lines so footer always stays at bottom
    while len(out) < lines - 3:
        out.append("")

    # footer
    out.append("-" * W)
    if state.err:
        out.append(f"error: {state.err[:W-7]}")
    else:
        out.append(f"[p] pause/resume  [c] clear  [q] quit")

    return "\n".join(out)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="log anomaly detection — terminal monitor")
    parser.add_argument("--mode",        default="random", choices=["random","ddos"])
    parser.add_argument("--sensitivity", default="low",    choices=["low","normal","high"])
    parser.add_argument("--batch",       default=5, type=int, metavar="N")
    parser.add_argument("--no-db",       action="store_true")
    args = parser.parse_args()

    use_db = not args.no_db
    if use_db and not MONGO_OK:
        print("warning: mongodb unavailable, running without storage")
        time.sleep(1)

    def shutdown(sig=None, frame=None):
        state.running = False
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("initialising pipeline...")
    try:
        predict([make_log()], args.sensitivity)
        print("pipeline ready")
    except Exception as e:
        print(f"pipeline error: {e}")
        sys.exit(1)

    time.sleep(0.3)

    threading.Thread(target=worker,   args=(args.batch, args.sensitivity, args.mode), daemon=True).start()
    threading.Thread(target=keyboard, daemon=True).start()

    # clear screen once
    os.system("cls" if os.name == "nt" else "clear")

    while state.running:
        frame = render(args.mode, args.sensitivity, use_db)
        # move cursor to top-left and overwrite
        sys.stdout.write("\033[H" + frame)
        sys.stdout.flush()
        time.sleep(0.15)

    # exit summary
    os.system("cls" if os.name == "nt" else "clear")
    print("session ended")
    print(f"  total      : {state.total}")
    print(f"  suspicious : {state.suspicious}")
    print(f"  critical   : {state.critical}")
    print(f"  anom rate  : {(state.suspicious+state.critical)/max(state.total,1)*100:.1f}%")
    print(f"  db stored  : {'yes' if (use_db and MONGO_OK) else 'no'}")

if __name__ == "__main__":
    main()