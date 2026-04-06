"""
FastAPI application for the Annotation QA Environment.

Wires up the environment to the OpenEnv HTTP/WebSocket server.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError:
    # Minimal fallback for standalone testing
    from openenv.core.env_server import create_fastapi_app as create_app

from .environment import AnnotationQAEnvironment

# Import models for type registration
import sys
import os

# Add parent to path for model imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import AnnotationQAAction, AnnotationQAObservation

# Create the app
app = create_app(
    AnnotationQAEnvironment,
    AnnotationQAAction,
    AnnotationQAObservation,
    env_name="annotation_qa_env",
)


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
