import asyncio
import time
from typing import List, Optional, Dict
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import logging
from contextlib import asynccontextmanager

from ..models.hybrid_model import HybridAnomalyPipeline
from ..rules.engine import RuleEngine
from ..mitre.mapper import MitreMapper
from ..monitoring.drift import DriftMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("api")

# Global instances
pipeline: Optional[HybridAnomalyPipeline] = None
rule_engine = RuleEngine()
mitre_mapper = MitreMapper()
drift_monitor = DriftMonitor()

# Async Queue for streaming ingestion
ingestion_queue = asyncio.Queue()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global pipeline
    model_path = "models/siem_model"
    try:
        pipeline = HybridAnomalyPipeline.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.warning(f"Could not load model from {model_path}: {e}. Starting with empty pipeline.")
        pipeline = HybridAnomalyPipeline()

    # Start queue worker
    worker_task = asyncio.create_task(process_queue())
    
    yield
    
    # Clean up
    worker_task.cancel()

app = FastAPI(title="AI-Augmented SOC Detection Engine", lifespan=lifespan)

class LogRequest(BaseModel):
    log_id: str
    log_text: str
    metadata: Optional[Dict] = {}

class DetectionResponse(BaseModel):
    predicted_label: str  # "Anomaly" or "Benign"
    confidence_score: float
    rule_based_alert: bool
    rule_reason: Optional[str]
    mitre_technique: Optional[str]
    mitre_tactic: Optional[str]
    drift_alert: bool
    processing_time_ms: float

async def process_queue():
    """Background worker to process logs from queue (Simulated Streaming)."""
    while True:
        try:
            log_req = await ingestion_queue.get()
            # In a real streaming scenario, we might batch these or push to a DB
            logger.info(f"Background processing log: {log_req.log_id}")
            ingestion_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in queue worker: {e}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_log(request: LogRequest):
    """
    Real-time detection endpoint.
    Combines BERT model + Rule Engine + Drift Detection + MITRE Mapping.
    """
    start_time = time.time()
    
    # 1. BERT/Hybrid Model Detection
    model_result = {}
    if pipeline and pipeline.is_fitted:
        try:
            # Running synchronous model in threadpool to not block async loop
            model_result = await asyncio.to_thread(
                pipeline.detect, request.log_text, request.metadata
            )
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            model_result = {'score': 0.0, 'is_anomaly': False}
    else:
        model_result = {'score': 0.0, 'is_anomaly': False}

    # 2. Rule Engine
    rule_result = rule_engine.check_rules(request.log_text, request.metadata)
    
    import numpy as np
    
    # 3. Drift Detection (Update monitor)
    embedding_list = model_result.get('embedding', [])
    drift_alert = False
    
    if embedding_list:
        try:
            # Flatten if it's a list within a list (e.g. [[...]])
            embedding_arr = np.array(embedding_list)
            if embedding_arr.ndim > 1:
                embedding_arr = embedding_arr.flatten()
            
            # Check for drift
            drift_alert = drift_monitor.add_sample(embedding_arr)
        except Exception as e:
            logger.warning(f"Drift detection error: {e}")
    
    # 4. MITRE Mapping
    mitre_info = mitre_mapper.map_alert(
        rule_id=rule_result.get('rule_id'), 
        log_text=request.log_text
    )
    
    # Final Decision Logic
    is_anomaly = model_result.get('is_anomaly', False) or rule_result['rule_based_alert']
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "predicted_label": "Attack" if is_anomaly else "Benign",
        "confidence_score": float(model_result.get('score', 0.0)),
        "rule_based_alert": rule_result['rule_based_alert'],
        "rule_reason": rule_result['rule_reason'],
        "mitre_technique": mitre_info['mitre_technique_id'],
        "mitre_tactic": mitre_info['mitre_tactic'],
        "drift_alert": drift_alert,  # Placeholder until we integrate embedding extraction
        "processing_time_ms": processing_time
    }

@app.post("/stream/ingest")
async def ingest_log_stream(request: LogRequest, background_tasks: BackgroundTasks):
    """
    Async ingestion for high-throughput streaming.
    Puts log into queue for background processing.
    """
    await ingestion_queue.put(request)
    return {"status": "queued", "log_id": request.log_id}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipeline.is_fitted if pipeline else False}
