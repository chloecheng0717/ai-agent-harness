"""
AI Agent Harness - FastAPI Application

This module creates and configures the FastAPI application instance.
It sets up middleware, CORS, logging, and includes all API routers.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.routes import health, tasks, agents

# Configure structured logging
settings = get_settings()

# Configure structlog based on settings
if settings.log_format == "json":
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
else:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the application.
    This is where you initialize database connections, caches,
    and other resources that need to be cleaned up on shutdown.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info(
        "Starting AI Agent Harness",
        version=settings.app_version,
        environment=settings.environment,
        debug=settings.debug,
    )
    
    # Initialize database connections, caches, etc.
    # await init_database()
    # await init_redis()
    # await init_vector_db()
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Agent Harness")
    
    # Cleanup resources
    # await close_database()
    # await close_redis()
    # await close_vector_db()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    This factory function creates the FastAPI application with all
    necessary configuration, middleware, and routes.
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="A reusable framework for building multi-agent AI systems",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, prefix="/api/v1", tags=["health"])
    app.include_router(tasks.router, prefix="/api/v1", tags=["tasks"])
    app.include_router(agents.router, prefix="/api/v1", tags=["agents"])
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> dict:
        """Root endpoint with basic application info."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
            "docs": "/docs" if settings.debug else None,
        }
    
    return app


# Create application instance
app = create_app()