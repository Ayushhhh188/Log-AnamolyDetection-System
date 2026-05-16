from fastapi import APIRouter, HTTPException, Query

from backend.models.schema import (
    LogEntry,
    BatchLogRequest,
    PredictionResult,
    BatchPredictionResponse,
    SimulationRequest,
    SimulationResponse,
    APIMessage,
)

from backend.services.predictor import (
    predict_single_log,
    predict_logs,
    simulate_logs,
)

#NEW: MongoDB import
from backend.db import get_collection


router = APIRouter(
    prefix="/logs",
    tags=["Logs"]
)


# Test route
@router.get("/test", response_model=APIMessage)
def test_logs_route():
    return {"message": "Logs route is working."}



# Predict a single log + STORE IN DB
@router.post("/predict", response_model=PredictionResult)
def predict_log(
    log: LogEntry,
    sensitivity: str = Query("low", enum=["low", "normal", "high"])
):
    try:
        result = predict_single_log(log.model_dump(), sensitivity=sensitivity)

        # ✅ STORE IN MONGODB
        collection = get_collection()
        collection.insert_one({
            **log.model_dump(),
            **result
        })

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Single log prediction failed: {str(e)}")


# Predict batch logs + STORE IN DB
@router.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict_logs(
    request: BatchLogRequest,
    sensitivity: str = Query("low", enum=["low", "normal", "high"])
):
    try:
        result = predict_logs(
            logs=[log.model_dump() for log in request.logs],
            sensitivity=sensitivity
        )

        #STORE BATCH RESULTS IN MONGODB
        collection = get_collection()

        documents = []
        for log, pred in zip(request.logs, result["results"]):
            documents.append({
                **log.model_dump(),
                **pred
            })

        if documents:
            collection.insert_many(documents)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Simulate logs + STORE IN DB
@router.post("/simulate", response_model=SimulationResponse)
def simulate_logs_route(request: SimulationRequest):
    try:
        result = simulate_logs(
            total_logs=request.total_logs,
            sensitivity=request.sensitivity,
            normal_ratio=request.normal_ratio,
            suspicious_ratio=request.suspicious_ratio,
            critical_ratio=request.critical_ratio,
        )

        #STORE SIMULATION RESULTS IN MONGODB
        collection = get_collection()

        if result.get("results"):
            collection.insert_many(result["results"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")