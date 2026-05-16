from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal


class LogEntry(BaseModel):
    timestamp: str = Field(..., example="2026-04-05 10:30:22.123")
    process_id: str = Field(..., example="DataNode")
    log_level: str = Field(..., example="INFO")
    component: str = Field(..., example="org.apache.hadoop.hdfs.server.datanode.DataNode")
    content: str = Field(..., example="Block received from /192.168.1.100")


class BatchLogRequest(BaseModel):
    logs: List[LogEntry]


class PredictionResult(BaseModel):
    timestamp: str
    process_id: str
    log_level: str
    component: str
    content: str
    anomaly_score: float
    label: int
    severity: Literal["normal", "suspicious", "critical"]
    reconstruction_error: Optional[float] = 0.0
    ground_truth: Optional[str] = None


class SummaryResponse(BaseModel):
    total_logs: int
    normal_count: int
    anomaly_count: int
    suspicious_count: int
    critical_count: int
    anomaly_percentage: float


class BatchPredictionResponse(BaseModel):
    summary: SummaryResponse
    results: List[PredictionResult]


class SimulationRequest(BaseModel):
    total_logs: int = Field(100, ge=1, le=10000, example=100)
    sensitivity: Literal["low", "normal", "high"] = "low"
    normal_ratio: float = Field(0.85, ge=0.0, le=1.0)
    suspicious_ratio: float = Field(0.10, ge=0.0, le=1.0)
    critical_ratio: float = Field(0.05, ge=0.0, le=1.0)
    export_json: bool = False

    @field_validator("critical_ratio")
    @classmethod
    def validate_ratios(cls, v, info):
        data = info.data
        normal = data.get("normal_ratio", 0.0)
        suspicious = data.get("suspicious_ratio", 0.0)
        total = normal + suspicious + v

        if abs(total - 1.0) > 0.01:
            raise ValueError("normal_ratio + suspicious_ratio + critical_ratio must sum to 1.0")
        return v


class SimulationResponse(BaseModel):
    summary: SummaryResponse
    results: List[PredictionResult]


class APIMessage(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str
