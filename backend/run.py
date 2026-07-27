"""
Application Runner

This script runs the FastAPI application with uvicorn.
It's used for development and testing.
"""

import uvicorn

from app.config import get_settings


def main():
    """Run the application."""
    settings = get_settings()
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=settings.workers if not settings.debug else 1,
    )


if __name__ == "__main__":
    main()