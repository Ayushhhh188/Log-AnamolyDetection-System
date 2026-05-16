import pandas as pd
import numpy as np
import sys
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

sys.path.append('DL/pipeline')
from batch_inference import predict_batch


class LogSimulator:
    """
    Simulates log streams for anomaly detection testing
    Can be used for demo, testing, or as a backend service
    """

    def __init__(self, sensitivity: str = 'low'):
        self.sensitivity = sensitivity

        self.normal_templates = [
            "registered UNIX signal handlers for [TERM, HUP]",
            "loaded properties from hadoop-metrics.properties",
            "Scheduled snapshot period at 10 second(s)",
            "Opened streaming server at /0.0.0.0:50010",
            "Http request log for http.requests.datanode is initialized",
            "Block received from /192.168.1.100",
            "Successfully deleted block blk_123456",
            "Heartbeat response received from NameNode",
            "Finished writing block blk_789012",
            "DataNode registered with NameNode"
        ]

        self.suspicious_templates = [
            "Slow block recovery detected for block blk_123",
            "Replication timeout for block blk_456",
            "High memory usage: 85% of limit",
            "Connection retry to NameNode attempt 3/10",
            "GC overhead limit exceeded in DataNode",
            "Response time exceeded threshold: 5000ms",
            "Replica removal delayed for block blk_789",
            "Pending replication count: 150"
        ]

        self.critical_templates = [
            "FATAL: Disk failure detected on volume /data",
            "CRITICAL: DataNode out of service",
            "ERROR: Corrupted block blk_999 detected",
            "FATAL: NameNode failed to start",
            "CRITICAL: HDFS cluster is in safe mode",
            "ERROR: Lost connection to NameNode",
            "FATAL: Metadata corruption detected",
            "CRITICAL: No space left on device"
        ]

        self.components = [
            "org.apache.hadoop.hdfs.server.datanode.DataNode",
            "org.apache.hadoop.hdfs.server.namenode.NameNode",
            "org.apache.hadoop.metrics2.impl.MetricsSystemImpl",
            "org.apache.hadoop.ipc.Server",
            "org.apache.hadoop.http.HttpServer2"
        ]

        self.process_ids = ["DataNode", "NameNode", "MetricsSystem", "IPCServer", "HttpServer"]

        print("Log Simulator initialized")
        print(f"Sensitivity: {self.sensitivity}")

    async def stream_logs(self, manager, mode: str = "random", delay: float = 0.5):
        """
        Stream logs in real-time to WebSocket

        Args:
            manager: WebSocket manager
            mode: "random" or "ddos"
            delay: time between logs
        """

        while True:
            if mode == "random":
                log_type = np.random.choice(
                    ["normal", "suspicious", "critical"],
                    p=[0.85, 0.10, 0.05]
                )
            elif mode == "ddos":
                log_type = np.random.choice(
                    ["critical", "suspicious"],
                    p=[0.7, 0.3]
                )
            else:
                await asyncio.sleep(1)
                continue

            log = self.generate_log(log_type)

            formatted_log = {
                "timestamp": log["timestamp"],
                "level": log["log_level"],
                "component": log["component"],
                "message": log["content"],
                "type": log_type
            }

            await manager.broadcast(formatted_log)
            await asyncio.sleep(delay)

    def generate_log(
        self,
        log_type: str = 'normal',
        timestamp: Optional[datetime] = None,
        custom_component: Optional[str] = None,
        custom_content: Optional[str] = None
    ) -> Dict:

        if timestamp is None:
            timestamp = datetime.now()

        if custom_content:
            content = custom_content
        elif log_type == 'normal':
            content = np.random.choice(self.normal_templates)
        elif log_type == 'suspicious':
            content = np.random.choice(self.suspicious_templates)
        elif log_type == 'critical':
            content = np.random.choice(self.critical_templates)
        else:
            content = np.random.choice(self.normal_templates)

        component = custom_component or np.random.choice(self.components)

        if log_type == 'normal':
            log_level = 'INFO'
        elif log_type == 'suspicious':
            log_level = np.random.choice(['WARN', 'INFO'])
        else:
            log_level = np.random.choice(['ERROR', 'FATAL'])

        process_id = np.random.choice(self.process_ids)

        return {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "process_id": process_id,
            "log_level": log_level,
            "component": component,
            "content": content,
            "metadata": {
                "log_type": log_type,
                "generated_by": "LogSimulator"
            }
        }

    def generate_batch(
        self,
        total_logs: int = 100,
        normal_ratio: float = 0.85,
        suspicious_ratio: float = 0.10,
        critical_ratio: float = 0.05,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:

        total_ratio = normal_ratio + suspicious_ratio + critical_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0 (got {total_ratio})")

        normal_count = int(total_logs * normal_ratio)
        suspicious_count = int(total_logs * suspicious_ratio)
        critical_count = total_logs - normal_count - suspicious_count

        logs = []
        current_time = start_time or datetime.now()

        for _ in range(normal_count):
            logs.append(self.generate_log('normal', current_time))
            current_time += timedelta(microseconds=np.random.randint(100000, 500000))

        for _ in range(suspicious_count):
            logs.append(self.generate_log('suspicious', current_time))
            current_time += timedelta(microseconds=np.random.randint(100000, 500000))

        for _ in range(critical_count):
            logs.append(self.generate_log('critical', current_time))
            current_time += timedelta(microseconds=np.random.randint(100000, 500000))

        np.random.shuffle(logs)

        df = pd.DataFrame(logs)
        return df.sort_values('timestamp').reset_index(drop=True)

    def run_detection(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        print(f"Running anomaly detection on {len(logs_df)} logs")
        return predict_batch(logs_df, sensitivity=self.sensitivity)