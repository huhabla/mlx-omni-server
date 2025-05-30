import argparse
import os

import uvicorn
from fastapi import FastAPI

from .config import settings  # Import zentrale Konfiguration
from .middleware.logging import RequestResponseLoggingMiddleware
from .routers import api_router
from .utils.logger import logger

app = FastAPI(title="MLX Omni Server")

# Add request/response logging middleware with custom levels
app.add_middleware(
    RequestResponseLoggingMiddleware,
    # exclude_paths=["/health"]
)

app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    """Log configuration on startup"""
    logger.info("MLX Omni Server starting...")
    logger.info(f"Cache directory: {settings.cache_directory}")
    logger.info(f"Log level: {settings.log_level}")
    if settings.default_model:
        logger.info(f"Default model: {settings.default_model}")


def build_parser():
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(description="MLX Omni Server")
    parser.add_argument(
        "--host",
        type=str,
        default=settings.host,
        help=f"Host to bind the server to, defaults to {settings.host}",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help=f"Port to bind the server to, defaults to {settings.port}",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.workers,
        help=f"Number of workers to use, defaults to {settings.workers}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=settings.log_level,
        choices=["debug", "info", "warning", "error", "critical"],
        help=f"Set the logging level, defaults to {settings.log_level}",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"Override cache directory (default from env: {settings.cache_directory})",
    )
    return parser


def start():
    """Start the MLX Omni Server."""
    parser = build_parser()
    args = parser.parse_args()

    # Override settings with command line arguments if provided
    if args.cache_dir:
        settings.cache_directory = args.cache_dir
        logger.info(f"Cache directory overridden to: {args.cache_dir}")

    # Set log level through environment variable
    os.environ["MLX_OMNI_LOG_LEVEL"] = args.log_level

    # Start server with uvicorn
    uvicorn.run(
        "mlx_omni_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )
