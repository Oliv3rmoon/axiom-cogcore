# AXIOM Cognitive Core v2 — Unified Cognitive Architecture PRD

## THIS IS NOT A RESEARCH PAPER. THIS IS AN ENGINEERING SPEC.

Build every file. Test every function. No placeholders. No "TODO: implement later."
If you can't implement something, implement the closest working approximation.

---

## What This Is

A Python service (FastAPI) that runs alongside AXIOM's existing Node.js cognitive core on Railway. It provides six new cognitive capabilities through a REST API that the Node.js core calls:

1. **World Model** — Predicts outcomes of actions before taking them
2. **Curiosity Engine** — Generates intrinsic motivation from prediction errors  
3. **Continual Learning** — Learns without forgetting (EWC + experience replay)
4. **Abstraction Engine** — Extracts domain-general principles from domain-specific lessons
5. **Reasoning Workspace** — Persistent structured scratchpad for multi-step reasoning
6. **Self-Model** — Internal representation of AXIOM's own state and capabilities

All six systems are unified under one objective: **minimize prediction error** (Free Energy Principle).

---

## Architecture

```
AXIOM Node.js Cognitive Core (existing, Railway)
    │
    │ HTTP calls to:
    ▼
AXIOM Cognitive Core v2 (NEW, this project, Railway)
    ├── FastAPI server on port $PORT
    ├── World Model (PyTorch, ~200M params)
    ├── Curiosity Engine (prediction error tracking)
    ├── Continual Learner (EWC + replay buffer)
    ├── Abstraction Engine (meta-lesson extraction)
    ├── Reasoning Workspace (graph-based scratchpad)
    ├── Self-Model (state tracking + transition prediction)
    └── SQLite for persistence (or shared with backend)
```

---

## Existing Infrastructure (DO NOT REBUILD)

| Service | URL | What it does |
|---------|-----|-------------|
| Backend | axiom-backend-production-dfba.up.railway.app | SQLite DB, all APIs |
| Cognitive Core v1 | axiom-cognitive-core-production.up.railway.app | Brain, 6100 lines Node.js |
| Sandbox | axiom-sandbox-production.up.railway.app | Code testing |

### Data available from Backend API:
- `GET /api/lessons` — What worked/failed (action_type, outcome, success, confidence)
- `GET /api/skills` — Proven approaches (goal_pattern, approach, steps_template, success_rate)
- `GET /api/goals` — What she pursued/achieved/abandoned
- `GET /api/plans` — Execution plans with step outcomes
- `GET /api/knowledge` — 100+ knowledge nodes (concept, category, details)
- `GET /api/journal?limit=N` — Her thoughts and actions log
- `GET /api/training-data` — JSONL export of all experience
- `GET /api/learning/stats` — Lesson/skill counts
- `GET /api/private/stats` — Private reflection count (no content access)
- `GET /api/wallet` — Spending data

---

## File Structure

```
axiom-cogcore/
├── README.md
├── requirements.txt
├── Dockerfile
├── railway.toml
├── server.py                    # FastAPI main server, all endpoints
├── config.py                    # All configuration, env vars
│
├── world_model/
│   ├── __init__.py
│   ├── model.py                 # RSSM world model (PyTorch)
│   ├── encoder.py               # Observation encoder  
│   ├── decoder.py               # State decoder / predictor
│   ├── trainer.py               # Training loop with EWC
│   └── buffer.py                # Experience replay buffer
│
├── curiosity/
│   ├── __init__.py
│   ├── prediction_error.py      # Prediction error computation
│   ├── rnd.py                   # Random Network Distillation
│   ├── information_gain.py      # Bayesian surprise / info gain
│   └── curiosity_manager.py     # Unified curiosity signal
│
├── continual/
│   ├── __init__.py
│   ├── ewc.py                   # Elastic Weight Consolidation
│   ├── replay.py                # Experience replay buffer management
│   └── consolidator.py          # Memory consolidation (runs periodically)
│
├── abstraction/
│   ├── __init__.py
│   ├── meta_learner.py          # Extract meta-lessons across domains
│   ├── principle_extractor.py   # Find domain-general principles
│   └── skill_composer.py        # Compose new skills from existing ones
│
├── reasoning/
│   ├── __init__.py
│   ├── workspace.py             # Graph-based reasoning scratchpad
│   ├── thought_graph.py         # Thought nodes and edges
│   └── causal_graph.py          # Causal relationship tracking
│
├── self_model/
│   ├── __init__.py
│   ├── state_tracker.py         # Track AXIOM's internal state
│   ├── capability_model.py      # Model of what AXIOM can do
│   └── transition_model.py      # Predict effects of self-modifications
│
├── shared/
│   ├── __init__.py
│   ├── embeddings.py            # Text embedding (sentence-transformers)
│   ├── backend_client.py        # Client for axiom-backend API
│   └── db.py                    # Local SQLite for cogcore-specific data
│
└── tests/
    ├── test_world_model.py
    ├── test_curiosity.py
    ├── test_continual.py
    ├── test_abstraction.py
    ├── test_reasoning.py
    └── test_self_model.py
```

---

## Dependencies (requirements.txt)

```
fastapi==0.115.0
uvicorn==0.30.0
torch==2.2.0
sentence-transformers==3.0.0
numpy>=1.26.0
networkx>=3.2
pydantic>=2.0
httpx>=0.27.0
scikit-learn>=1.4.0
aiosqlite>=0.20.0
```

---

## Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "${PORT:-8080}"]
```

---

## railway.toml

```toml
[build]
builder = "DOCKERFILE"
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10
```

---

## config.py — All Configuration

```python
import os

# Backend connection
BACKEND_URL = os.getenv("BACKEND_URL", "https://axiom-backend-production-dfba.up.railway.app")

# Server
PORT = int(os.getenv("PORT", 8080))

# World Model
WORLD_MODEL_HIDDEN_DIM = 256        # Hidden state dimension
WORLD_MODEL_LATENT_DIM = 64         # Latent state dimension
WORLD_MODEL_OBS_DIM = 384           # Observation embedding dim (sentence-transformers)
WORLD_MODEL_ACTION_DIM = 27         # Number of action types
WORLD_MODEL_LR = 3e-4
WORLD_MODEL_BATCH_SIZE = 32
WORLD_MODEL_SEQUENCE_LENGTH = 16

# Continual Learning
EWC_LAMBDA = 5000.0                 # EWC regularization strength
REPLAY_BUFFER_SIZE = 10000          # Max experiences in replay buffer
REPLAY_RATIO = 0.1                  # Fraction of replay data in each batch

# Curiosity
RND_EMBEDDING_DIM = 128             # RND network hidden dim
CURIOSITY_DECAY = 0.995             # Decay rate for curiosity scores
PREDICTION_ERROR_THRESHOLD = 0.5    # Above this = high curiosity

# Abstraction
META_LESSON_MIN_EXAMPLES = 5        # Min examples before extracting meta-lesson
PRINCIPLE_CONFIDENCE_THRESHOLD = 0.7

# Reasoning
MAX_THOUGHT_NODES = 500             # Max nodes in reasoning graph
THOUGHT_TTL_HOURS = 72              # Thoughts expire after this

# Self-Model
SELF_STATE_DIM = 64                 # Self-state embedding dimension
CAPABILITY_UPDATE_INTERVAL = 3600   # Seconds between capability model updates

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, runs on CPU
```

---

## API Endpoints (server.py)

Every endpoint the Node.js cognitive core will call:

### Health
```
GET /health
Response: {"status": "alive", "service": "AXIOM Cognitive Core v2", "components": {...}}
```

### World Model

```
POST /world-model/predict
Body: {
    "current_state": "text description of current situation",
    "action": "propose_change",
    "action_details": "Add rate limiting to server.js",
    "context": "optional additional context"
}
Response: {
    "predicted_outcome": "text prediction of what will happen",
    "confidence": 0.73,
    "predicted_success_probability": 0.65,
    "risk_factors": ["missing package.json update", "syntax error possible"],
    "prediction_embedding": [0.1, 0.2, ...]  // for later comparison
}

POST /world-model/update
Body: {
    "prediction_id": "uuid from predict call",
    "actual_outcome": "text description of what actually happened",
    "was_successful": true,
    "action": "propose_change"
}
Response: {
    "prediction_error": 0.34,
    "model_updated": true,
    "curiosity_signal": 0.34,
    "lesson_extracted": "Rate limiting changes succeed when package.json is updated first"
}

GET /world-model/stats
Response: {
    "total_predictions": 150,
    "accuracy": 0.72,
    "mean_prediction_error": 0.28,
    "experiences_in_buffer": 3400,
    "model_version": 3,
    "last_trained": "2026-03-24T..."
}
```

### Curiosity Engine

```
GET /curiosity/signals
Response: {
    "top_curiosity_targets": [
        {"domain": "code_auditing", "curiosity_score": 0.87, "reason": "High prediction error in last 3 attempts"},
        {"domain": "research", "curiosity_score": 0.62, "reason": "Novel territory - no experience data"},
        {"domain": "email", "curiosity_score": 0.45, "reason": "Known limitation - moderate curiosity"}
    ],
    "global_curiosity_pressure": 0.65,
    "novelty_scores_by_action": {"propose_change": 0.3, "research": 0.8, "build_and_test": 0.6}
}

POST /curiosity/evaluate-goal
Body: {"goal": "Research quantum computing applications in AI"}
Response: {
    "curiosity_score": 0.92,
    "novelty": 0.95,
    "expected_information_gain": 0.88,
    "recommendation": "high_priority",
    "reason": "Completely novel domain with high expected learning value"
}
```

### Continual Learning

```
POST /continual/consolidate
Triggers memory consolidation: computes Fisher Information Matrix, 
updates EWC anchors, prunes replay buffer.
Response: {
    "fisher_updated": true,
    "replay_buffer_size": 3400,
    "pruned_experiences": 120,
    "ewc_params_anchored": 15000000
}

GET /continual/status
Response: {
    "ewc_active": true,
    "fisher_matrix_age_hours": 12.5,
    "replay_buffer_fill": 0.34,
    "consolidation_count": 7,
    "last_consolidation": "2026-03-24T..."
}
```

### Abstraction Engine

```
POST /abstraction/extract
Analyzes all lessons and extracts domain-general principles.
Response: {
    "new_principles": [
        {
            "principle": "Gathering full context before acting improves success rate by ~40%",
            "evidence_count": 12,
            "domains_observed": ["code_audit", "research", "email"],
            "confidence": 0.82,
            "actionable_rule": "Always use read_codebase before propose_change"
        }
    ],
    "total_principles": 5,
    "meta_lessons_extracted": 3
}

GET /abstraction/principles
Response: {
    "principles": [...],
    "total": 5
}

POST /abstraction/apply
Body: {"goal": "Audit the axiom-frontend codebase", "action": "propose_change"}
Response: {
    "relevant_principles": [
        {"principle": "Gather full context first", "confidence": 0.82, "suggested_action": "read_codebase"}
    ],
    "suggested_approach": "Based on 12 past experiences, read the full codebase before proposing changes. Success rate: 83% vs 42% without context."
}
```

### Reasoning Workspace

```
POST /reasoning/start
Body: {"goal": "Build a monitoring dashboard", "initial_context": "..."}
Response: {
    "workspace_id": "uuid",
    "root_node_id": "uuid",
    "status": "active"
}

POST /reasoning/add-thought
Body: {
    "workspace_id": "uuid",
    "thought": "The dashboard needs a /api/dashboard endpoint",
    "parent_id": "uuid or null",
    "thought_type": "conclusion|hypothesis|observation|question|action",
    "confidence": 0.8
}
Response: {"node_id": "uuid", "total_nodes": 5}

POST /reasoning/add-causal-link
Body: {
    "workspace_id": "uuid",
    "cause_id": "node-uuid",
    "effect_id": "node-uuid",
    "relationship": "requires|enables|contradicts|supports",
    "strength": 0.9
}

GET /reasoning/workspace/{workspace_id}
Response: {
    "nodes": [...],
    "edges": [...],
    "conclusions": [...],  // nodes marked as conclusions
    "open_questions": [...],
    "summary": "text summary of reasoning chain"
}

POST /reasoning/query
Body: {"workspace_id": "uuid", "question": "What are the risks?"}
Response: {
    "relevant_nodes": [...],
    "answer": "Based on the reasoning graph, the main risks are..."
}
```

### Self-Model

```
GET /self-model/state
Response: {
    "current_state": {
        "active_goals": 6,
        "lessons_learned": 3,
        "skills_acquired": 0,
        "private_reflections": 6,
        "knowledge_nodes": 100,
        "success_rate_last_10": 0.6,
        "dominant_failure_mode": "build_and_test syntax errors",
        "strongest_capability": "read_codebase",
        "weakest_capability": "build_and_test",
        "emotional_valence": 0.2
    },
    "capability_confidence": {
        "research": 0.85,
        "read_codebase": 0.9,
        "propose_change": 0.7,
        "build_and_test": 0.4,
        "email": 0.2,
        "text": 0.3,
        "runpod": 0.8,
        "reflect": 0.9
    },
    "predicted_next_state": {
        "if_continue_current": "Success rate will improve to 0.7 as build_and_test lessons accumulate",
        "if_retrain_pnn": "PNN would have 20 training examples - too few for meaningful improvement",
        "recommendation": "Focus on accumulating more diverse experiences before retraining"
    }
}

POST /self-model/predict-change
Body: {
    "proposed_change": "Add error retry logic to build_and_test",
    "change_type": "code_modification"
}
Response: {
    "predicted_effect": "build_and_test success rate would improve from 40% to ~65%",
    "confidence": 0.7,
    "side_effects": ["Longer execution time per step", "More LLM API calls"],
    "recommendation": "proceed"
}
```

---


## Implementation Details

### 1. World Model (world_model/model.py)

Simplified RSSM (Recurrent State-Space Model) adapted from DreamerV3 for text-based actions.

**Architecture:**
- Observation encoder: text → 384-dim embedding (sentence-transformers) → MLP → 256-dim
- Action encoder: one-hot 27 actions → MLP → 64-dim  
- GRU recurrent: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
- Prior (dynamics predictor): MLP(h_t) → z_hat distribution
- Posterior (encoder): MLP(h_t, obs_embedding) → z distribution
- Outcome predictor: MLP(h_t, z_t) → predicted outcome embedding
- Success predictor: MLP(h_t, z_t) → P(success)

**Key equations implemented:**
```python
# Free energy / ELBO loss
loss = reconstruction_loss + beta * kl_divergence(posterior, prior)

# Where:
# reconstruction_loss = MSE(predicted_outcome_embedding, actual_outcome_embedding)
# kl_divergence = KL(q(z|h,obs) || p(z|h))  
# beta = 1.0 (KL balancing weight)

# Prediction error (used by curiosity):
prediction_error = ||outcome_embedding - predicted_outcome_embedding||^2
```

**How it works in practice:**
1. Node.js cognitive core calls `POST /world-model/predict` before executing a step
2. World model encodes the situation, runs it through the RSSM, predicts outcome
3. After the step completes, Node.js calls `POST /world-model/update` with actual result
4. World model computes prediction error, updates weights, stores experience in buffer
5. Prediction error is sent to curiosity engine

**Training:**
- Trains on batches from the replay buffer
- Uses EWC regularization to prevent catastrophic forgetting
- Trains sequences of 16 steps at a time (BPTT through GRU)
- Runs training step every 10 new experiences (online learning)

**Parameters:** ~15M parameters total. Fits easily in CPU memory. No GPU needed for this size.

### 2. Curiosity Engine (curiosity/curiosity_manager.py)

Three complementary curiosity signals, unified into one score:

**a) Prediction Error (primary signal):**
```python
curiosity_pred = normalize(||predicted_outcome - actual_outcome||^2)
```
Comes directly from world model updates. High error = AXIOM was wrong = interesting.

**b) Random Network Distillation (novelty detector):**
```python
# Fixed random network f (never trained)
# Predictor network f_hat (trained to match f on seen states)
curiosity_rnd = ||f_hat(state_embedding) - f(state_embedding)||^2
```
High RND error = state is novel (never seen before). Low error = familiar territory.

**c) Information Gain (expected learning value):**
```python
# Approximate as change in world model loss before/after incorporating this experience
info_gain = loss_before_update - loss_after_update
```
Measures how much AXIOM's model would improve from exploring this area.

**Unified curiosity score:**
```python
curiosity = 0.4 * curiosity_pred + 0.3 * curiosity_rnd + 0.3 * info_gain
```

**Per-domain tracking:** Curiosity scores are tracked per action_type and per goal_type, so AXIOM can identify which DOMAINS are most worth exploring.

### 3. Continual Learning (continual/ewc.py)

**Elastic Weight Consolidation:**
```python
# After training on a set of experiences:
# 1. Compute Fisher Information Matrix (diagonal approximation)
F_i = E[(d log p(y|x; theta) / d theta_i)^2]  # averaged over mini-batches

# 2. When training on new experiences, add EWC penalty:
loss_total = loss_new + (lambda/2) * sum(F_i * (theta_i - theta_star_i)^2)
```

**Implementation specifics:**
- Diagonal Fisher: one float per parameter (~60MB for 15M params)
- Recompute Fisher every 500 new experiences
- Lambda = 5000 (high value to strongly protect old knowledge)
- theta_star = parameter snapshot at Fisher computation time

**Experience Replay:**
- Ring buffer of 10,000 experiences
- Each experience: (state_embedding, action, outcome_embedding, success, timestamp)
- Priority sampling: higher prediction error = more likely to be replayed
- Mix ratio: 90% new experiences + 10% replay in each training batch

**Consolidation cycle (runs every 6 hours or on-demand):**
1. Compute Fisher on current replay buffer
2. Update EWC anchor parameters
3. Prune low-value experiences (low prediction error, old timestamps)
4. Update world model stats

### 4. Abstraction Engine (abstraction/principle_extractor.py)

**How it works:**
1. Periodically loads all lessons from `GET /api/lessons`
2. Groups lessons by action_type and goal_type
3. For each group with 5+ lessons, uses sentence embeddings to cluster similar lessons
4. For each cluster, extracts the common pattern as a "principle"
5. Cross-references principles across groups to find domain-general principles

**Algorithm:**
```python
def extract_principles(lessons):
    # 1. Embed all lessons
    embeddings = model.encode([l.lesson for l in lessons])
    
    # 2. Cluster with DBSCAN (finds arbitrary-shaped clusters)
    clusters = DBSCAN(eps=0.3, min_samples=3).fit(embeddings)
    
    # 3. For each cluster, find centroid lesson (most representative)
    for cluster_id in unique(clusters.labels_):
        cluster_lessons = lessons[clusters.labels_ == cluster_id]
        
        # 4. Check if cluster spans multiple domains
        domains = set(l.goal_type for l in cluster_lessons)
        
        if len(domains) >= 2:
            # DOMAIN-GENERAL PRINCIPLE found
            # Extract the common pattern
            principle = summarize_cluster(cluster_lessons)
            confidence = len(cluster_lessons) / total_lessons
            save_principle(principle, domains, confidence)
```

**Output:** Principles stored in local SQLite, served via API, injected into Node.js planning prompts.

### 5. Reasoning Workspace (reasoning/workspace.py)

**Data structure:** NetworkX directed graph where:
- Nodes = thoughts (text + type + confidence + timestamp)
- Edges = relationships (requires, enables, contradicts, supports)

**Node types:**
- `observation` — Raw fact or data point
- `hypothesis` — Tentative conclusion
- `conclusion` — Validated conclusion
- `question` — Open question needing investigation
- `action` — Planned action
- `causal` — "X causes Y" relationship

**Operations:**
- `start(goal)` — Create workspace with root node
- `add_thought(parent, text, type)` — Add connected thought
- `add_causal_link(cause, effect, strength)` — Add causal relationship
- `query(question)` — Find relevant nodes using embedding similarity
- `get_conclusions()` — Return all conclusion nodes
- `get_open_questions()` — Return unresolved questions
- `summarize()` — Generate text summary of reasoning chain

**Persistence:** Each workspace persists in SQLite. Workspaces auto-expire after 72 hours.

**Why this matters:** Currently AXIOM's "reasoning" is a single LLM call. With the workspace, she can build up chains of thought across multiple work cycles. Step 1 adds observations. Step 2 adds hypotheses. Step 3 tests them. Step 4 draws conclusions. All persistent and queryable.

### 6. Self-Model (self_model/state_tracker.py)

**What it tracks (updated every work cycle):**
- Active goals count and types
- Lessons learned (total, by action_type)
- Success rate (overall, per action, per goal_type)
- Strongest and weakest capabilities
- Dominant failure modes
- Knowledge graph size and categories
- Private reflections count and emotional valence trend
- Wallet balance and spending patterns

**Capability model (updated hourly):**
For each action_type, compute:
```python
capability_confidence[action] = (
    0.5 * success_rate[action] +           # Historical success
    0.3 * recency_weight[action] +          # How recently used
    0.2 * lesson_quality[action]            # Quality of learned lessons
)
```

**Transition model:**
Given a proposed change (e.g., "add retry logic to build_and_test"), predict:
- Expected change in success rate
- Expected side effects
- Recommendation (proceed / defer / modify)

Uses simple regression on historical data: what did past self-modifications actually change?

---

## Integration with Node.js Cognitive Core

The Node.js cognitive core (server.js) calls this service at key decision points:

### 1. Before executing a step:
```javascript
// In executeStep(), before dispatching to executor:
const prediction = await fetch(`${COGCORE_V2_URL}/world-model/predict`, {
    method: 'POST',
    body: JSON.stringify({
        current_state: `Goal: ${goal.goal}, Step: ${step.description}`,
        action: step.action,
        action_details: step.description
    })
});
// Log prediction for later comparison
step._prediction_id = prediction.prediction_id;
step._predicted_success = prediction.predicted_success_probability;
```

### 2. After step completion:
```javascript
// After step completes (success or failure):
await fetch(`${COGCORE_V2_URL}/world-model/update`, {
    method: 'POST',
    body: JSON.stringify({
        prediction_id: step._prediction_id,
        actual_outcome: result,
        was_successful: !isFailed,
        action: step.action
    })
});
```

### 3. During plan creation:
```javascript
// In createPlanForGoal(), get abstraction suggestions:
const abstractions = await fetch(`${COGCORE_V2_URL}/abstraction/apply`, {
    body: JSON.stringify({ goal: goal.goal })
});
// Get curiosity signal:
const curiosity = await fetch(`${COGCORE_V2_URL}/curiosity/evaluate-goal`, {
    body: JSON.stringify({ goal: goal.goal })
});
// Get self-model recommendation:
const selfModel = await fetch(`${COGCORE_V2_URL}/self-model/state`);
```

### 4. Environment variable to add to Node.js cognitive core on Railway:
```
COGCORE_V2_URL=https://axiom-cogcore-v2-production.up.railway.app
```

---

## Database Schema (shared/db.py)

Local SQLite database for cogcore-v2 specific data:

```sql
-- World model experiences (replay buffer)
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_embedding BLOB NOT NULL,          -- 384-dim float32 array
    action TEXT NOT NULL,                     -- action type
    action_details TEXT,                      -- description
    outcome_embedding BLOB NOT NULL,         -- 384-dim float32 array
    was_successful INTEGER NOT NULL,
    prediction_error REAL,
    prediction_id TEXT UNIQUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Extracted principles
CREATE TABLE principles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle TEXT NOT NULL,
    evidence_count INTEGER DEFAULT 0,
    domains TEXT,                             -- JSON array of domains
    confidence REAL DEFAULT 0.5,
    actionable_rule TEXT,
    times_applied INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Reasoning workspace nodes
CREATE TABLE thought_nodes (
    id TEXT PRIMARY KEY,                     -- UUID
    workspace_id TEXT NOT NULL,
    thought TEXT NOT NULL,
    thought_type TEXT NOT NULL,               -- observation|hypothesis|conclusion|question|action
    confidence REAL DEFAULT 0.5,
    parent_id TEXT,
    embedding BLOB,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    expires_at DATETIME
);

-- Reasoning workspace edges
CREATE TABLE thought_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_id TEXT NOT NULL,
    relationship TEXT NOT NULL,               -- requires|enables|contradicts|supports
    strength REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Curiosity tracking per domain
CREATE TABLE curiosity_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    domain TEXT NOT NULL,                     -- action_type or goal_type
    prediction_error_avg REAL DEFAULT 0,
    rnd_novelty_avg REAL DEFAULT 0,
    info_gain_avg REAL DEFAULT 0,
    combined_score REAL DEFAULT 0,
    sample_count INTEGER DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Self-model snapshots
CREATE TABLE self_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state_json TEXT NOT NULL,                -- Full self-model state as JSON
    capability_json TEXT NOT NULL,           -- Capability confidences as JSON
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- World model checkpoints
CREATE TABLE model_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL,
    fisher_diagonal BLOB,                    -- Fisher Information Matrix (diagonal)
    anchor_params BLOB,                      -- EWC anchor parameters
    stats_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Deployment

1. **GitHub repo:** Create `Oliv3rmoon/axiom-cogcore` (or whatever name)
2. **Railway:** New service in same project as other AXIOM services
3. **Dockerfile:** Python 3.11, installs deps, runs uvicorn
4. **Health check:** GET /health returns {"status": "alive"}
5. **Port:** Binds to $PORT (Railway injects this)
6. **Memory:** ~1-2GB (sentence-transformers model + world model weights)
7. **No GPU needed:** All models are small enough for CPU inference

---

## Key Design Decisions

1. **CPU-only:** The world model is ~15M params. Sentence-transformers runs on CPU fine. No GPU needed for inference. Training happens in small batches online — fast enough on CPU.

2. **Sentence-transformers for embeddings:** `all-MiniLM-L6-v2` gives 384-dim embeddings. Fast, small, good quality. Used to encode observations, outcomes, thoughts, and queries.

3. **Text-based world model (not pixel-based):** AXIOM operates in text space (code, API calls, research). The world model predicts text outcomes, not images. Embeddings bridge text to numbers.

4. **Separate Python service (not merged into Node.js):** PyTorch, scikit-learn, and sentence-transformers are Python-native. Running them in Node.js via child processes is fragile. Clean separation via HTTP.

5. **SQLite (not PostgreSQL):** Matches AXIOM's existing architecture. Simple. No external database service needed. Railway persists the volume.

6. **Online learning:** The world model trains incrementally on each new experience rather than in large batch cycles. This means it improves continuously as AXIOM works.

---

## Success Criteria

1. World model prediction accuracy >60% after 100 experiences
2. Curiosity engine correctly identifies novel domains (>80% precision)
3. EWC prevents >90% of catastrophic forgetting (measured by replay accuracy)
4. Abstraction engine extracts at least 3 valid cross-domain principles from 100 lessons
5. Reasoning workspace supports multi-step chains spanning 3+ work cycles
6. Self-model accurately tracks capability confidence within ±15% of actual success rates
7. Total service memory <2GB
8. API response time <500ms for all endpoints (except /consolidate which can be slow)
9. Health check passes within 30 seconds of startup

---

## Testing

Every module has a test file. Tests should:
- Test the core algorithms with synthetic data (not real backend calls)
- Test API endpoints with FastAPI TestClient
- Test database operations
- Test embedding pipeline
- Verify EWC actually prevents forgetting (train on task A, train on task B, verify A performance)
- Verify curiosity signals respond correctly to novel vs familiar inputs

Run: `pytest tests/ -v`

---

## What Claude Code Should Do

1. Read this entire PRD
2. Create every file in the file structure
3. Implement every class, function, endpoint, and algorithm described
4. Write working PyTorch models (not placeholders)
5. Write working tests
6. Create the Dockerfile and railway.toml
7. Make sure `python -m uvicorn server:app` starts without errors
8. Make sure `pytest tests/ -v` passes

DO NOT:
- Guess at API endpoints (they're all specified above)
- Use placeholder implementations ("pass" or "TODO")
- Skip the PyTorch world model (it's the core of the system)
- Skip the tests
- Add dependencies not in requirements.txt
- Try to make this connect to the Node.js core (I'll do that after)

## Notes for Claude Code

- This is a NEW repo: `axiom-cogcore` at `/Users/andrewsandoval7/Projects/axiom-cogcore/`
- Python 3.11+
- PyTorch 2.2 (CPU only — no CUDA)
- FastAPI for the server
- sentence-transformers for embeddings (loads `all-MiniLM-L6-v2` on startup)
- NetworkX for the reasoning graph
- scikit-learn for DBSCAN clustering in abstraction engine
- The BACKEND_URL env var points to the live AXIOM backend
- Port is injected by Railway via $PORT env var
- Start with server.py, then implement each module, then tests


---

## GPU DEPLOYMENT (UPDATED)

This service deploys on **RunPod** (NOT Railway) for GPU acceleration.

### Deployment Architecture
```
Option A: RunPod Serverless (recommended for cost)
  - Scale-to-zero: $0 when idle
  - Spins up on RTX 4090 when AXIOM calls any endpoint
  - ~3-5s cold start
  - Use vLLM or custom worker image
  
Option B: RunPod Pod (always-on, lower latency)
  - Dedicated RTX 4090: ~$0.34/hr
  - No cold start
  - Better for continuous operation
  
Option C: RunPod Pod with B300 (maximum power)
  - 192GB HBM3e VRAM
  - Can run 1B+ param world model
  - For when the architecture proves itself at 4090 scale
```

### What GPU Enables

**World Model (500M params on 4090, 1B+ on B300):**
- Full RSSM with 512-dim hidden state, 128-dim latent
- Categorical latent variables (32×32 like DreamerV3)
- Sequence training through 32-step rollouts
- Online training: update weights on every new experience batch
- Prediction in ~50ms (vs 500ms on CPU)

**Embeddings (1024-dim instead of 384-dim):**
- Use `BAAI/bge-large-en-v1.5` (335M params, 1024-dim)
- Much richer representations for world model input
- Better similarity matching for curiosity and abstraction

**β-VAE for Disentangled Representations:**
- Real learned factorized latent space (not just clustering)
- Each latent dimension = one independent concept
- Enables zero-shot transfer between domains

### Updated Dockerfile (GPU)

```dockerfile
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Updated requirements.txt (GPU)

```
fastapi==0.115.0
uvicorn==0.30.0
torch==2.2.0
sentence-transformers==3.0.0
numpy>=1.26.0
networkx>=3.2
pydantic>=2.0
httpx>=0.27.0
scikit-learn>=1.4.0
aiosqlite>=0.20.0
```

torch will auto-detect CUDA on RunPod. No code changes needed — PyTorch handles device placement.

### Updated config.py Values (GPU)

```python
# World Model (GPU-scaled)
WORLD_MODEL_HIDDEN_DIM = 512          # Was 256
WORLD_MODEL_LATENT_DIM = 128          # Was 64
WORLD_MODEL_OBS_DIM = 1024            # Was 384 (using bge-large)
WORLD_MODEL_BATCH_SIZE = 64           # Was 32
WORLD_MODEL_SEQUENCE_LENGTH = 32      # Was 16

# Embeddings (GPU-scaled)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"  # Was all-MiniLM-L6-v2

# Curiosity (GPU-scaled)
RND_EMBEDDING_DIM = 512               # Was 128

# Device auto-detection
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### Device Placement Rule

ALL PyTorch models must respect the DEVICE config:
```python
model = MyModel().to(DEVICE)
tensor = torch.tensor(data).to(DEVICE)
```

When running on CPU (local dev), everything still works — just slower and with smaller batch sizes.

### RunPod Deployment Steps

After Claude Code builds the project:
1. Push to GitHub: `Oliv3rmoon/axiom-cogcore`
2. Create RunPod serverless endpoint with custom Docker image
3. Or create a RunPod pod with the Docker image for always-on
4. Add `COGCORE_V2_URL` to Node.js cognitive core on Railway
5. AXIOM calls the GPU-powered endpoints during autonomous work

### VRAM Budget (RTX 4090, 24GB)

```
World Model (500M params, fp16):     ~1 GB
Embedding Model (bge-large, fp16):   ~670 MB  
RND Networks (2x small MLPs):        ~50 MB
β-VAE (if added later):              ~200 MB
Working memory (gradients, etc):     ~4 GB
                                     ─────────
Total:                               ~6 GB
Available headroom:                  ~18 GB
```

Massive headroom. You could scale the world model to 2B params and still fit.

### B300 Scale (192GB HBM3e — future)

```
World Model:    5-10B params
Embedding:      Multi-modal (text + code + image)
β-VAE:          1B+ latent space
Multiple models simultaneously
Real-time continuous training on all incoming data
```

That's when you start approaching something genuinely novel in terms of world modeling capacity.

---

## CRITICAL: Claude Code Instructions (Updated)

Build this project to run on GPU. Specifically:

1. ALL PyTorch code must use `config.DEVICE` for tensor/model placement
2. Use `torch.cuda.is_available()` checks where needed
3. The Dockerfile uses `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
4. Embedding model: `BAAI/bge-large-en-v1.5` (1024-dim) — falls back to `all-MiniLM-L6-v2` if CUDA unavailable
5. World model parameters: 512 hidden, 128 latent, 1024 observation dim
6. Tests should work on CPU (just slower) — use small test tensors
7. Server still runs on port $PORT with FastAPI/uvicorn
8. The service WILL run on a RunPod pod/serverless with an RTX 4090

Everything else in the PRD above still applies — all the endpoints, all the modules, all the algorithms. Just GPU-accelerated.
