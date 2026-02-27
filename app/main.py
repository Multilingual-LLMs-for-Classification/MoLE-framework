"""
FastAPI application entry point.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import admin, analytics, auth, classify, health
from app.services.routing_service import routing_service
from app.middleware.error_handler import ErrorHandlerMiddleware, add_request_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events:
    - Startup: Initialize routing system and load models
    - Shutdown: Cleanup resources
    """
    # Startup
    mode = settings.service_mode
    print("=" * 60)
    print(f"Starting MoLE Classification Service  [mode={mode}]")
    print("=" * 60)

    # Initialize routing system in background thread
    # (model loading is CPU/IO bound and can block the event loop)
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, routing_service.initialize)

    if success:
        print("=" * 60)
        if mode == "coordinator":
            print("Coordinator ready! (gating pipeline + gateway to expert workers)")
        else:
            print("Classification Service ready! (monolithic mode)")
        stats = routing_service.get_system_stats()
        print(f"  Domains: {stats.get('total_domains', 0)}")
        print(f"  Tasks: {stats.get('total_tasks', 0)}")
        print(f"  Languages: {stats.get('supported_languages', 0)}")
        print("=" * 60)
    else:
        print("WARNING: Failed to initialize routing system")
        print("Service will start but classification endpoints will be unavailable")

    yield  # Application runs here

    # Shutdown
    print(f"Shutting down MoLE Classification Service [mode={mode}] ...")


# Create FastAPI application
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handling middleware
app.add_middleware(ErrorHandlerMiddleware)

# Add request ID middleware
app.middleware("http")(add_request_id)

# Include routers
app.include_router(admin.router)
app.include_router(analytics.router)
app.include_router(auth.router)
app.include_router(classify.router)
app.include_router(health.router)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
