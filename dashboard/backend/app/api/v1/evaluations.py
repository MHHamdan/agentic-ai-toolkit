"""Evaluations API endpoints."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional

from app.schemas.evaluation import (
    EvaluationConfig,
    EvaluationStatus,
    EvaluationResult,
)
from app.services.evaluation_service import evaluation_service

router = APIRouter()


@router.get("/", response_model=List[EvaluationStatus])
@router.get("", response_model=List[EvaluationStatus], include_in_schema=False)
async def list_evaluations(
    status: Optional[str] = None,
    limit: int = 20,
):
    """List all evaluations."""
    return await evaluation_service.list_evaluations(status, limit)


@router.post("/", response_model=EvaluationStatus)
@router.post("", response_model=EvaluationStatus, include_in_schema=False)
async def start_evaluation(
    config: EvaluationConfig,
    background_tasks: BackgroundTasks,
):
    """Start a new evaluation."""
    evaluation = await evaluation_service.create_evaluation(config)
    background_tasks.add_task(evaluation_service.run_evaluation, evaluation.evaluation_id)
    return evaluation


@router.get("/{evaluation_id}", response_model=EvaluationStatus)
async def get_evaluation(evaluation_id: str):
    """Get evaluation status."""
    evaluation = await evaluation_service.get_evaluation(evaluation_id)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return evaluation


@router.get("/{evaluation_id}/results", response_model=EvaluationResult)
async def get_evaluation_results(evaluation_id: str):
    """Get full evaluation results."""
    results = await evaluation_service.get_results(evaluation_id)
    if not results:
        raise HTTPException(status_code=404, detail="Results not found")
    return results


@router.delete("/{evaluation_id}")
async def cancel_evaluation(evaluation_id: str):
    """Cancel a running evaluation."""
    success = await evaluation_service.cancel_evaluation(evaluation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    return {"status": "cancelled", "evaluation_id": evaluation_id}


@router.post("/demo")
async def run_demo_evaluation(model: str = "gemma2:2b"):
    """Run a quick demo evaluation with real Ollama inference."""
    result = await evaluation_service.run_quick_demo(model)
    if not result:
        raise HTTPException(status_code=500, detail="Demo evaluation failed")
    return result
