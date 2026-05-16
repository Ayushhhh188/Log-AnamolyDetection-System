import os
import sys
import logging
from typing import List, Dict, Any

import pandas as pd

# -----------------------------
# Path setup
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PIPELINE_DIR = os.path.join(BASE_DIR, "DL", "pipeline")
SIMULATION_DIR = os.path.join(BASE_DIR, "simulation")

if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

if SIMULATION_DIR not in sys.path:
    sys.path.append(SIMULATION_DIR)

# -----------------------------
# External imports (your working modules)
# -----------------------------
try:
    from batch_inference import predict_batch
except ImportError as e:
    raise ImportError(f"Failed to import predict_batch from DL/pipeline/batch_inference.py: {e}")

try:
    from log_generator import LogSimulator
except ImportError as e:
    raise ImportError(f"Failed to import LogSimulator from simulation/log_generator.py: {e}")

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# Utility: Convert logs to DataFrame
# =========================================================
def logs_to_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of log dictionaries into a pandas DataFrame.

    Args:
        logs: List of log dictionaries

    Returns:
        pd.DataFrame
    """
    if not logs:
        raise ValueError("No logs provided for prediction.")

    required_fields = {"timestamp", "process_id", "log_level", "component", "content"}

    for idx, log in enumerate(logs):
        missing = required_fields - set(log.keys())
        if missing:
            raise ValueError(f"Log at index {idx} is missing required fields: {missing}")

    df = pd.DataFrame(logs)
    return df


# =========================================================
# Utility: Build summary stats
# =========================================================
def build_summary(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build dashboard/API summary from prediction results.

    Args:
        results_df: DataFrame returned from prediction pipeline

    Returns:
        dict summary
    """
    total_logs = len(results_df)
    anomaly_count = int((results_df["label"] == -1).sum())
    normal_count = int((results_df["label"] == 1).sum())
    suspicious_count = int((results_df["severity"] == "suspicious").sum())
    critical_count = int((results_df["severity"] == "critical").sum())

    anomaly_percentage = round((anomaly_count / total_logs) * 100, 2) if total_logs > 0 else 0.0

    return {
        "total_logs": total_logs,
        "normal_count": normal_count,
        "anomaly_count": anomaly_count,
        "suspicious_count": suspicious_count,
        "critical_count": critical_count,
        "anomaly_percentage": anomaly_percentage,
    }


# =========================================================
# Utility: Format prediction output
# =========================================================
def format_prediction_results(
    logs_df: pd.DataFrame,
    results_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Merge original logs with prediction results into JSON-safe output.

    Args:
        logs_df: Original logs DataFrame
        results_df: Prediction output DataFrame

    Returns:
        List of formatted prediction dictionaries
    """
    formatted_results = []

    min_len = min(len(logs_df), len(results_df))

    for i in range(min_len):
        log_row = logs_df.iloc[i]
        result_row = results_df.iloc[i]

        entry = {
            "timestamp": str(log_row.get("timestamp", "")),
            "process_id": str(log_row.get("process_id", "")),
            "log_level": str(log_row.get("log_level", "")),
            "component": str(log_row.get("component", "")),
            "content": str(log_row.get("content", "")),
            "anomaly_score": float(result_row.get("anomaly_score", 0.0)),
            "label": int(result_row.get("label", 1)),
            "severity": str(result_row.get("severity", "normal")),
            "reconstruction_error": float(result_row.get("reconstruction_error", 0.0)),
        }

        # Optional ground truth from simulator metadata
        metadata = log_row.get("metadata", None)
        if isinstance(metadata, dict):
            entry["ground_truth"] = metadata.get("log_type")

        formatted_results.append(entry)

    return formatted_results


# =========================================================
# Core: Predict logs (batch)
# =========================================================
def predict_logs(logs: List[Dict[str, Any]], sensitivity: str = "low") -> Dict[str, Any]:
    """
    Run anomaly detection on a batch of logs.

    Args:
        logs: List of log dictionaries
        sensitivity: low / normal / high

    Returns:
        dict with summary and results
    """
    try:
        logger.info(f"Running batch prediction on {len(logs)} logs with sensitivity='{sensitivity}'")

        logs_df = logs_to_dataframe(logs)
        results_df = predict_batch(logs_df, sensitivity=sensitivity)

        formatted_results = format_prediction_results(logs_df, results_df)
        summary = build_summary(results_df)

        return {
            "summary": summary,
            "results": formatted_results
        }

    except Exception as e:
        logger.exception("Error during batch prediction")
        raise RuntimeError(f"Prediction failed: {str(e)}")


# =========================================================
# Core: Predict a single log
# =========================================================
def predict_single_log(log: Dict[str, Any], sensitivity: str = "low") -> Dict[str, Any]:
    """
    Predict anomaly status for a single log.

    Args:
        log: Single log dictionary
        sensitivity: low / normal / high

    Returns:
        dict for one prediction result
    """
    result = predict_logs([log], sensitivity=sensitivity)
    if not result["results"]:
        raise RuntimeError("Prediction returned no result for single log.")
    return result["results"][0]


# =========================================================
# Core: Simulate + Detect
# =========================================================
def simulate_logs(
    total_logs: int = 100,
    sensitivity: str = "low",
    normal_ratio: float = 0.85,
    suspicious_ratio: float = 0.10,
    critical_ratio: float = 0.05,
) -> Dict[str, Any]:
    """
    Generate simulated logs and run anomaly detection.

    Args:
        total_logs: Number of logs to generate
        sensitivity: low / normal / high
        normal_ratio: ratio of normal logs
        suspicious_ratio: ratio of suspicious logs
        critical_ratio: ratio of critical logs

    Returns:
        dict with summary and results
    """
    try:
        logger.info(
            f"Running simulation with total_logs={total_logs}, sensitivity={sensitivity}, "
            f"ratios=({normal_ratio}, {suspicious_ratio}, {critical_ratio})"
        )

        simulator = LogSimulator(sensitivity=sensitivity)

        logs_df = simulator.generate_batch(
            total_logs=total_logs,
            normal_ratio=normal_ratio,
            suspicious_ratio=suspicious_ratio,
            critical_ratio=critical_ratio
        )

        results_df = simulator.run_detection(logs_df)

        formatted_results = format_prediction_results(logs_df, results_df)
        summary = build_summary(results_df)

        return {
            "summary": summary,
            "results": formatted_results
        }

    except Exception as e:
        logger.exception("Error during simulation")
        raise RuntimeError(f"Simulation failed: {str(e)}")


# =========================================================
# Optional: Dashboard helper
# =========================================================
def get_summary_only(logs: List[Dict[str, Any]], sensitivity: str = "low") -> Dict[str, Any]:
    """
    Return only summary statistics for a batch of logs.

    Args:
        logs: List of log dictionaries
        sensitivity: low / normal / high

    Returns:
        dict summary
    """
    try:
        logs_df = logs_to_dataframe(logs)
        results_df = predict_batch(logs_df, sensitivity=sensitivity)
        return build_summary(results_df)

    except Exception as e:
        logger.exception("Error generating summary")
        raise RuntimeError(f"Summary generation failed: {str(e)}")