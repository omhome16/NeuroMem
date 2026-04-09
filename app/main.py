"""
NeuroMem — Human-Inspired Persistent Memory Engine for LLMs.

FastAPI application entry point with lifespan management.
Initializes all database connections (PostgreSQL, Redis, Qdrant, Neo4j)
and mounts API routes.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect databases. Shutdown: close connections."""
    logger.info("starting_neuromem")

    # Initialize database connections
    from app.db.postgres import create_pg_pool, close_pg_pool
    from app.db.redis_client import create_redis_pool, close_redis_pool
    from app.db.qdrant_client import init_qdrant
    from app.db.neo4j_client import init_neo4j, close_neo4j

    await create_pg_pool()
    await create_redis_pool()
    await init_qdrant()

    try:
        await init_neo4j()
    except Exception as e:
        logger.warning(f"neo4j_init_failed: {e} — graph features will be unavailable")

    logger.info("all_services_connected")

    # Preload ML models to eliminate cold-start latency on first request
    # Without this, first request pays ~3-5s for model loading from disk
    try:
        from app.memory.semantic.embedder import get_embedder
        _embedder = get_embedder()  # Loads BGE-small-en-v1.5 and caches it
        _embedder.embed("warmup")  # Force the model to run one forward pass
        logger.info("embedding_model_preloaded")

        from app.core.token_budget import get_reranker
        _reranker = get_reranker()  # Loads ms-marco-MiniLM and caches it
        _reranker.predict([("warmup", "warmup")]) # Warm up the cross-encoder
        logger.info("reranker_model_preloaded")
    except Exception as e:
        logger.warning(f"model_preload_failed: {e} — first request will be slower")

    yield

    # Cleanup
    await close_pg_pool()
    await close_redis_pool()
    try:
        await close_neo4j()
    except Exception:
        pass
    logger.info("neuromem_shutdown_complete")


app = FastAPI(
    title="NeuroMem",
    description=(
        "Human-inspired persistent memory engine for LLMs. "
        "Features a 5-tier cognitive architecture with surprise-gated storage, "
        "procedural memory, temporal knowledge graph, Ebbinghaus forgetting curve, "
        "and LangGraph-powered retrieval with cross-encoder re-ranking."
    ),
    version="3.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (dashboard) ────────────────────────────────────────
import os
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "traceback": traceback.format_exc()}
    )

dashboard_dir = os.path.join(os.path.dirname(__file__), "..", "dashboard")
if os.path.isdir(dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

# ── Routes ───────────────────────────────────────────────────────────
from app.api.routes import chat, memory, admin, health, graph

app.include_router(chat.router, tags=["Chat"])
app.include_router(memory.router, prefix="/memory", tags=["Memory"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])
app.include_router(graph.router, prefix="/graph", tags=["Knowledge Graph"])
app.include_router(health.router, tags=["Health"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "NeuroMem",
        "version": "3.0.0",
        "description": "Human-inspired persistent memory engine for LLMs",
        "docs": "/docs",
        "dashboard": "/dashboard",
        "features": [
            "5-tier cognitive memory (Working + Episodic + Semantic + Graph + Procedural)",
            "Surprise-gated memory storage (Titans-inspired)",
            "Procedural memory — learns user interaction patterns",
            "Temporal knowledge graph with fact invalidation",
            "Contradiction detection with automatic memory invalidation",
            "Ebbinghaus forgetting curve with adaptive decay",
            "Cross-encoder re-ranking with token budget management",
            "Heuristic routing — zero-latency tier selection (<1ms)",
            "LangGraph v2 orchestration (10-node pipeline)",
        ],
    }
