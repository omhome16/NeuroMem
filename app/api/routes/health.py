"""
Health check endpoint — GET /health

Reports status of all infrastructure dependencies.
"""
import logging

from fastapi import APIRouter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """Readiness + liveness probe for all services."""
    checks = {}

    # PostgreSQL
    try:
        from app.db.postgres import get_pg_pool
        pg = await get_pg_pool()
        await pg.fetchval("SELECT 1")
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = f"error: {e}"

    # Redis
    try:
        from app.db.redis_client import get_redis_client
        redis = await get_redis_client()
        await redis.ping()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {e}"

    # Qdrant
    try:
        from app.db.qdrant_client import get_qdrant_client
        qdrant = get_qdrant_client()
        await qdrant.get_collections()
        checks["qdrant"] = "ok"
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "checks": checks,
    }
