"""
main.py — FastAPI app for easy-wakeword-trainer.

Endpoints:
  GET  /                        — serve the single-page UI
  GET  /api/health              — data asset check
  POST /api/train               — start a training job
  GET  /api/train/{job_id}/events  — SSE log stream
  GET  /api/train/{job_id}/status  — JSON status
  GET  /api/train/{job_id}/download        — zip of .onnx + .tflite
  GET  /api/train/{job_id}/download/onnx   — .onnx only
  GET  /api/train/{job_id}/download/tflite — .tflite only
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import uuid

from app import pipeline as pl

app = FastAPI(title="easy-wakeword-trainer", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Root — serve UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")


# ---------------------------------------------------------------------------
# Health — data asset check
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health():
    present = pl.check_data()
    missing = [k for k, v in present.items() if not v]
    descriptions = pl.REQUIRED_DATA
    return {
        "ready": len(missing) == 0,
        "assets": {
            name: {
                "present": present[name],
                "description": descriptions[name],
            }
            for name in present
        },
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Prepare — first-run data download
# ---------------------------------------------------------------------------

@app.post("/api/prepare", status_code=202)
async def start_prepare(background_tasks: BackgroundTasks):
    """Start the data-download job. Idempotent if data is already present."""
    if pl.prepare_running():
        job = pl.get_prepare_job()
        return {"job_id": job.job_id, "status": "already_running"}

    if pl.all_data_present():
        return {"job_id": None, "status": "already_ready"}

    job = pl.PrepareJob(job_id=str(uuid.uuid4()))
    background_tasks.add_task(pl.run_prepare, job)
    return {"job_id": job.job_id, "status": "started"}


@app.get("/api/prepare/status")
async def prepare_status():
    job = pl.get_prepare_job()
    if not job:
        return {"stage": "not_started", "log_lines": []}
    return {
        "job_id": job.job_id,
        "stage": job.stage.value,
        "error": job.error,
        "log_lines": job.log_lines[-50:],  # last 50 lines for polling fallback
    }


@app.get("/api/prepare/events")
async def prepare_events():
    job = pl.get_prepare_job()
    if not job:
        raise HTTPException(status_code=404, detail="No prepare job started yet.")
    return StreamingResponse(
        pl.stream_prepare_events(job),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Start training job
# ---------------------------------------------------------------------------

class TrainRequest(BaseModel):
    phrase: str


@app.post("/api/train", status_code=202)
async def start_train(req: TrainRequest, background_tasks: BackgroundTasks):
    # Validate phrase
    try:
        phrase = pl.validate_phrase(req.phrase)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Only one job at a time
    if pl.active_job():
        raise HTTPException(
            status_code=409,
            detail="A training job is already running. Wait for it to finish.",
        )

    # Check data is present
    if not pl.all_data_present():
        missing = [k for k, v in pl.check_data().items() if not v]
        raise HTTPException(
            status_code=428,
            detail=f"Required data assets are missing: {missing}. "
                   "Run scripts/prepare_data.py first.",
        )

    model_name = pl.phrase_to_model_name(phrase)
    job_id = str(uuid.uuid4())
    job = pl.TrainJob(job_id=job_id, phrase=phrase, model_name=model_name)
    pl._jobs[job_id] = job

    background_tasks.add_task(pl.run_training, job)

    return {"job_id": job_id, "model_name": model_name, "phrase": phrase}


# ---------------------------------------------------------------------------
# SSE event stream
# ---------------------------------------------------------------------------

@app.get("/api/train/{job_id}/events")
async def events(job_id: str):
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    return StreamingResponse(
        pl.stream_events(job),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@app.get("/api/train/{job_id}/status")
async def status(job_id: str):
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job.job_id,
        "phrase": job.phrase,
        "model_name": job.model_name,
        "stage": job.stage.value,
        "error": job.error,
        "artifacts_ready": job.artifacts_ready(),
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

def _require_job(job_id: str) -> pl.TrainJob:
    job = pl.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.stage == pl.Stage.ERROR:
        raise HTTPException(status_code=500, detail=f"Job failed: {job.error}")
    if job.stage != pl.Stage.DONE:
        raise HTTPException(status_code=202, detail="Training not finished yet.")
    return job


@app.get("/api/train/{job_id}/download")
async def download_zip(job_id: str):
    job = _require_job(job_id)
    if not job.zip_path.exists():
        raise HTTPException(status_code=404, detail="Zip not found.")
    return FileResponse(
        str(job.zip_path),
        media_type="application/zip",
        filename=f"{job.model_name}.zip",
    )


@app.get("/api/train/{job_id}/download/onnx")
async def download_onnx(job_id: str):
    job = _require_job(job_id)
    if not job.onnx_path.exists():
        raise HTTPException(status_code=404, detail="ONNX not found.")
    return FileResponse(
        str(job.onnx_path),
        media_type="application/octet-stream",
        filename=f"{job.model_name}.onnx",
    )


@app.get("/api/train/{job_id}/download/tflite")
async def download_tflite(job_id: str):
    job = _require_job(job_id)
    if not job.tflite_path.exists():
        raise HTTPException(status_code=404, detail="TFLite not found.")
    return FileResponse(
        str(job.tflite_path),
        media_type="application/octet-stream",
        filename=f"{job.model_name}.tflite",
    )
