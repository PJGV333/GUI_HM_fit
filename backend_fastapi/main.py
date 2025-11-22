"""Minimal FastAPI backend for HM Fit prototype.

This module intentionally does not modify existing calculation logic.
Endpoints are placeholders where we can later call functions from:
- Simulation_controls.py (simulation routines)
- Methods.py, NR_conc_algoritm.py, LM_conc_algoritm.py (fitting routines)
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn


class DummyFitRequest(BaseModel):
    """Placeholder request model for testing the backend wiring."""

    x: list[float]
    y: list[float]


app = FastAPI(title="HM Fit FastAPI prototype")


@app.get("/health")
def health():
    """Health check endpoint used by the Tauri frontend."""

    return {"status": "ok"}


@app.post("/dummy_fit")
def dummy_fit(req: DummyFitRequest):
    """Dummy endpoint that will later delegate to HM Fit calculation modules."""

    return {"sum_y": sum(req.y), "n_points": len(req.y)}


if __name__ == "__main__":
    uvicorn.run("backend_fastapi.main:app", host="127.0.0.1", port=8000, reload=True)
