from __future__ import annotations
import os

# Backend connection
BACKEND_URL = os.getenv("BACKEND_URL", "https://axiom-backend-production-dfba.up.railway.app")

# Server
PORT = int(os.getenv("PORT", 8080))

# Device — set DEVICE env var to skip torch import at config load time.
# On Railway: DEVICE=cpu. On RunPod: DEVICE=cuda.
# If not set, auto-detected (triggers torch import).
_DEVICE_ENV = os.getenv("DEVICE", "")
if _DEVICE_ENV:
    DEVICE = _DEVICE_ENV
else:
    try:
        import torch as _torch
        DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"
    except ImportError:
        DEVICE = "cpu"

# World Model (GPU-scaled)
WORLD_MODEL_HIDDEN_DIM = 512
WORLD_MODEL_LATENT_DIM = 128
WORLD_MODEL_OBS_DIM = 1024
WORLD_MODEL_ACTION_DIM = 27
WORLD_MODEL_LR = 3e-4
WORLD_MODEL_BATCH_SIZE = 64
WORLD_MODEL_SEQUENCE_LENGTH = 32
WORLD_MODEL_KL_BETA = 1.0
WORLD_MODEL_TRAIN_EVERY = 10  # Train after every N new experiences

# Continual Learning
EWC_LAMBDA = 5000.0
REPLAY_BUFFER_SIZE = 10000
REPLAY_RATIO = 0.1

# Curiosity
RND_EMBEDDING_DIM = 512
CURIOSITY_DECAY = 0.995
PREDICTION_ERROR_THRESHOLD = 0.5

# Abstraction
META_LESSON_MIN_EXAMPLES = 5
PRINCIPLE_CONFIDENCE_THRESHOLD = 0.7

# Reasoning
MAX_THOUGHT_NODES = 500
THOUGHT_TTL_HOURS = 72

# Self-Model
SELF_STATE_DIM = 64
CAPABILITY_UPDATE_INTERVAL = 3600

# Embeddings - GPU uses bge-large, CPU falls back to MiniLM
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-large-en-v1.5" if DEVICE == "cuda" else "all-MiniLM-L6-v2"
)

# Action types AXIOM uses
ACTION_TYPES = [
    "research", "read_codebase", "propose_change", "build_and_test",
    "email", "text", "runpod", "reflect", "plan", "analyze",
    "debug", "refactor", "deploy", "monitor", "review",
    "document", "test", "optimize", "migrate", "configure",
    "integrate", "automate", "learn", "teach", "collaborate",
    "create", "delete"
]

# Database
DB_PATH = os.getenv("DB_PATH", "cogcore.db")

# ──────────────────────────────────────────────
# Phase 2 Configuration
# ──────────────────────────────────────────────

# β-VAE
BETA_VAE_LATENT_DIM = 32
BETA_VAE_HIDDEN_DIM = 256
BETA_VAE_BETA = 4.0
BETA_VAE_CAPACITY_MAX = 25.0
BETA_VAE_LR = 1e-4

# Active Inference
AI_PRECISION_INIT = 1.0
AI_PRECISION_LR = 0.01
AI_GAMMA = 1.0
AI_EPISTEMIC_WEIGHT = 0.5
AI_PLANNING_HORIZON = 3

# Modern Hopfield
HOPFIELD_PATTERN_DIM = 384
HOPFIELD_MAX_PATTERNS = 10000
HOPFIELD_BETA = 1.0
HOPFIELD_CONSOLIDATION_THRESHOLD = 0.95

# Meta-Learning (Reptile)
REPTILE_INNER_LR = 0.01
REPTILE_OUTER_LR = 0.001
REPTILE_INNER_STEPS = 5
REPTILE_TASKS_PER_UPDATE = 4

# ──────────────────────────────────────────────
# Phase 3 Configuration
# ──────────────────────────────────────────────

# DreamCoder
DREAMCODER_MIN_SOLUTIONS = 5
DREAMCODER_MIN_PATTERN_FREQ = 3
DREAMCODER_MIN_DOMAINS = 2
DREAMCODER_MAX_LIBRARY_SIZE = 100

# Global Workspace
GW_SALIENCE_THRESHOLD = 0.3
GW_BROADCAST_HISTORY_SIZE = 100
GW_COMPETITION_INTERVAL = 1.0

# Causal Reasoner
CAUSAL_MIN_EVIDENCE = 3
CAUSAL_STRENGTH_THRESHOLD = 0.15
CAUSAL_MAX_NODES = 200

# ──────────────────────────────────────────────
# Phase 4 Configuration
# ──────────────────────────────────────────────

# Attention Schema
ATTENTION_SWITCH_THRESHOLD = 0.15
ATTENTION_HISTORY_SIZE = 50
ATTENTION_DECAY_RATE = 0.95

# Predictive Hierarchy
HIERARCHY_LEVELS = 4
HIERARCHY_PROPAGATION_RATE = 0.3
HIERARCHY_PRECISION_LR = 0.05

# Liquid Network (CfC)
LTC_HIDDEN_SIZE = 128
LTC_INPUT_SIZE = 384
LTC_TIME_CONSTANT_MIN = 0.1
LTC_TIME_CONSTANT_MAX = 10.0
LTC_LR = 1e-4
