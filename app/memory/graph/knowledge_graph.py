"""
Temporal Knowledge Graph — Neo4j-backed (Tier 4).

Stores entity-relationship triples with temporal validity.
When a contradiction is detected, old facts get a valid_to timestamp
instead of being deleted — maintaining full provenance.

Graph Schema:
    (:Entity {name, entity_type, user_id})
        -[:RELATION {predicate, valid_from, valid_to, confidence, source_memory_id}]->
    (:Entity {name, entity_type, user_id})

Inspired by Zep's temporal context graph approach.
"""
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID

from neo4j import AsyncDriver

from app.config import get_settings
from app.memory.graph.entity_extractor import EntityTriple

logger = logging.getLogger(__name__)
settings = get_settings()


class KnowledgeGraph:
    """
    Neo4j-backed temporal knowledge graph.

    Stores entity-relationship triples with valid_from/valid_to.
    Supports contradiction-aware updates (old facts are invalidated,
    not deleted) and graph traversal for context assembly.
    """

    def __init__(self, driver: AsyncDriver):
        self.driver = driver

    async def add_triples(
        self,
        user_id: str,
        triples: List[EntityTriple],
        source_memory_id: Optional[str] = None,
        now_iso: Optional[str] = None,
    ) -> int:
        """
        Add entity-relationship triples to the knowledge graph.

        For each triple, checks for contradictions and invalidates
        old facts if necessary (temporal provenance).
        """
        added = 0
        async with self.driver.session() as session:
            for triple in triples:
                # Check for existing contradicting relations
                existing = await self._get_active_relation(
                    session, user_id, triple.subject, triple.predicate
                )

                if existing and existing["object"] != triple.object:
                    # Contradiction detected — invalidate old fact
                    await self._invalidate_relation(
                        session, existing["rel_id"], now_iso=now_iso
                    )
                    logger.info(
                        "graph_fact_invalidated",
                        extra={
                            "subject": triple.subject,
                            "predicate": triple.predicate,
                            "old": existing["object"],
                            "new": triple.object,
                        },
                    )

                # Add new triple
                await self._create_triple(
                    session, user_id, triple, source_memory_id, now_iso=now_iso
                )
                added += 1

        logger.info("graph_triples_added", extra={"count": added, "user_id": user_id})
        return added

    async def _get_active_relation(
        self, session, user_id: str, subject: str, predicate: str
    ) -> Optional[Dict[str, Any]]:
        """Find currently active relation (valid_to IS NULL)."""
        result = await session.run(
            """
            MATCH (s:Entity {name: $subject, user_id: $user_id})
                  -[r:RELATION {predicate: $predicate}]->
                  (o:Entity)
            WHERE r.valid_to IS NULL
            RETURN elementId(r) AS rel_id, o.name AS object, r.valid_from AS valid_from
            LIMIT 1
            """,
            subject=subject, user_id=user_id, predicate=predicate,
        )
        record = await result.single()
        return dict(record) if record else None

    async def _invalidate_relation(self, session, rel_id: str, now_iso: Optional[str] = None) -> None:
        """Set valid_to on an existing relation (temporal invalidation)."""
        now_expr = "datetime($now_iso)" if now_iso else "datetime()"
        await session.run(
            f"""
            MATCH ()-[r:RELATION]->()
            WHERE elementId(r) = $rel_id
            SET r.valid_to = {now_expr}
            """,
            rel_id=rel_id,
            now_iso=now_iso,
        )

    async def _create_triple(
        self,
        session,
        user_id: str,
        triple: EntityTriple,
        source_memory_id: Optional[str],
        now_iso: Optional[str] = None,
    ) -> None:
        """Create a new entity-relationship triple."""
        now_expr = "datetime($now_iso)" if now_iso else "datetime()"
        await session.run(
            f"""
            MERGE (s:Entity {{name: $subject, user_id: $user_id}})
            ON CREATE SET s.entity_type = 'unknown', s.created_at = {now_expr}
            MERGE (o:Entity {{name: $object, user_id: $user_id}})
            ON CREATE SET o.entity_type = 'unknown', o.created_at = {now_expr}
            CREATE (s)-[:RELATION {{
                predicate: $predicate,
                valid_from: {now_expr},
                valid_to: null,
                confidence: $confidence,
                source_memory_id: $source_id
            }}]->(o)
            """,
            subject=triple.subject,
            object=triple.object,
            predicate=triple.predicate,
            confidence=triple.confidence,
            source_id=source_memory_id or "",
            now_iso=now_iso,
            user_id=user_id,
        )

    async def get_user_graph(
        self, user_id: str, include_invalidated: bool = False
    ) -> Dict[str, Any]:
        """
        Retrieve the full knowledge graph for a user.
        Returns nodes and edges for visualization.
        """
        where_clause = "" if include_invalidated else "AND r.valid_to IS NULL"

        async with self.driver.session() as session:
            result = await session.run(
                f"""
                MATCH (s:Entity {{user_id: $user_id}})
                      -[r:RELATION]->
                      (o:Entity {{user_id: $user_id}})
                WHERE TRUE {where_clause}
                RETURN s.name AS subject, s.entity_type AS subject_type,
                       r.predicate AS predicate,
                       r.valid_from AS valid_from, r.valid_to AS valid_to,
                       r.confidence AS confidence,
                       o.name AS object, o.entity_type AS object_type
                ORDER BY r.valid_from DESC
                """,
                user_id=user_id,
            )

            nodes = set()
            edges = []
            async for record in result:
                nodes.add(record["subject"])
                nodes.add(record["object"])
                edges.append({
                    "subject": record["subject"],
                    "predicate": record["predicate"],
                    "object": record["object"],
                    "valid_from": str(record["valid_from"]) if record["valid_from"] else None,
                    "valid_to": str(record["valid_to"]) if record["valid_to"] else None,
                    "confidence": record["confidence"],
                    "is_active": record["valid_to"] is None,
                })

        return {
            "nodes": [{"id": n, "label": n} for n in nodes],
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

    async def query_context(
        self, user_id: str, query_entities: List[str], max_hops: int = 2
    ) -> List[str]:
        """
        Retrieve relevant graph context for given entities.
        Traverses up to max_hops from the query entities.
        Returns human-readable fact strings.
        """
        facts = []
        async with self.driver.session() as session:
            for entity in query_entities:
                result = await session.run(
                    """
                    MATCH (s:Entity {user_id: $user_id})
                          -[r:RELATION]->(o:Entity)
                    WHERE (s.name = $entity OR o.name = $entity)
                      AND r.valid_to IS NULL
                    RETURN s.name AS subject, r.predicate AS predicate, o.name AS object
                    LIMIT 10
                    """,
                    user_id=user_id, entity=entity,
                )
                async for record in result:
                    fact = f"{record['subject']} {record['predicate'].replace('_', ' ')} {record['object']}"
                    facts.append(fact)
        return list(set(facts))

    async def delete_user_graph(self, user_id: str) -> int:
        """GDPR: Delete all entities and relations for a user."""
        async with self.driver.session() as session:
            result = await session.run(
                """
                MATCH (e:Entity {user_id: $user_id})
                DETACH DELETE e
                RETURN count(e) AS deleted
                """,
                user_id=user_id,
            )
            record = await result.single()
            count = record["deleted"] if record else 0
            logger.info("graph_deleted", extra={"user_id": user_id, "nodes_deleted": count})
            return count
