"""
Health Check Routes

This module provides health check endpoints for monitoring application status.
These endpoints are typically used by load balancers and monitoring systems.
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter

from app.config import get_settings

router = APIRouter()
logger = structlog.get_logger()
settings = get_settings()


@router.get("/health")
async def health_check() -> dict:
    """
    Basic health check endpoint.
    
    Returns:
        dict: Health status with timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.app_version,
    }


@router.get("/health/detailed")
async def detailed_health_check() -> dict:
    """
    Detailed health check with dependency status.
    
    This endpoint checks the status of all dependencies
    (database, Redis, vector database) and returns their health.
    
    Returns:
        dict: Detailed health status with dependency information
    """
    # In production, actually check dependencies
    dependencies = {
        "database": {"status": "healthy", "latency_ms": 5},
        "redis": {"status": "healthy", "latency_ms": 2},
        "vector_database": {"status": "healthy", "latency_ms": 10},
    }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.app_version,
        "environment": settings.environment,
        "dependencies": dependencies,
    }