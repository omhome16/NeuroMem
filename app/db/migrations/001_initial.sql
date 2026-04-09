-- NeuroMem — Simplified Schema (Single-User, No Encryption)
-- ==========================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Episodic Memories ─────────────────────────────────────────────────
-- Stores LLM-extracted facts from conversations (Tier 2: Mid-term)
CREATE TABLE IF NOT EXISTS episodic_memories (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         VARCHAR(255) NOT NULL DEFAULT 'default',
    content         TEXT NOT NULL,
    memory_type     VARCHAR(50) NOT NULL CHECK (
                        memory_type IN ('fact', 'preference', 'event', 'relationship')
                    ),
    importance      FLOAT NOT NULL DEFAULT 0.5 CHECK (importance BETWEEN 0 AND 1),
    recall_count    INTEGER NOT NULL DEFAULT 0,
    tags            TEXT[] DEFAULT '{}',
    source_turn     INTEGER,
    session_id      UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_recalled   TIMESTAMPTZ,
    consolidated    BOOLEAN NOT NULL DEFAULT FALSE,
    decay_score     FLOAT NOT NULL DEFAULT 1.0,
    expires_at      TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 days')
);

CREATE INDEX IF NOT EXISTS idx_episodic_user_id ON episodic_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_created_at ON episodic_memories(created_at);
CREATE INDEX IF NOT EXISTS idx_episodic_type ON episodic_memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_episodic_decay ON episodic_memories(decay_score);
CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_memories(consolidated);
CREATE INDEX IF NOT EXISTS idx_episodic_expires ON episodic_memories(expires_at);

-- ── Consolidation Log ─────────────────────────────────────────────────
-- Tracks nightly consolidation runs (episodic → semantic promotion)
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             VARCHAR(255) NOT NULL DEFAULT 'default',
    episodes_processed  INTEGER NOT NULL DEFAULT 0,
    facts_written       INTEGER NOT NULL DEFAULT 0,
    memories_deleted    INTEGER NOT NULL DEFAULT 0,
    duration_ms         INTEGER,
    status              VARCHAR(20) NOT NULL DEFAULT 'success'
                            CHECK (status IN ('success', 'failed', 'partial')),
    run_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
