# AXIOM Cognitive Core v2 — Phase 2 PRD

## THIS EXTENDS THE EXISTING CODEBASE. DO NOT REBUILD PHASE 1.

The axiom-cogcore project already exists at `/Users/andrewsandoval7/Projects/axiom-cogcore/` with 46 files, 60 passing tests, and a running FastAPI server. Phase 2 ADDS new modules and MODIFIES existing ones. Do NOT delete or rewrite Phase 1 code.

---

## What Phase 2 Adds

4 new capabilities that upgrade Phase 1's foundation:

| New System | What It Does | Core Math |
|-----------|-------------|-----------|
| **β-VAE** | Learns factorized representations — each latent dimension = one independent concept | L = E[log p(x|z)] - β·KL(q(z|x) ‖ p(z)) |
| **Active Inference** | Unifies curiosity + planning under one equation — Expected Free Energy | G(π) = ambiguity + risk = epistemic + pragmatic |
| **Modern Hopfield Memory** | Exponential-capacity episodic memory — stores/retrieves experience patterns | E(ξ) = -lse(β, X^T ξ) + ½ξ^T ξ |
| **Reptile Meta-Learning** | Rapid adaptation to new domains — learns HOW to learn | Φ ← Φ + ε(W̃ - Φ) |

---

## New Files to Create

```
axiom-cogcore/                          (EXISTING PROJECT)
├── ... (all existing files stay)       
│
├── beta_vae/                           (NEW)
│   ├── __init__.py
│   ├── model.py                        # β-VAE encoder/decoder (PyTorch)
│   ├── trainer.py                      # Training loop with β annealing
│   └── representations.py             # Extract/query disentangled representations
│
├── active_inference/                   (NEW)
│   ├── __init__.py
│   ├── generative_model.py            # P(observations, states) — the world as AXIOM models it
│   ├── expected_free_energy.py         # G(π) computation — unified curiosity + planning
│   ├── policy_selection.py            # Select actions via softmax over -G(π)
│   └── precision.py                   # Learned confidence/attention weighting
│
├── hopfield/                           (NEW)
│   ├── __init__.py
│   ├── modern_hopfield.py             # Modern Hopfield network (exponential capacity)
│   ├── episodic_store.py              # Store/retrieve experience episodes
│   └── memory_manager.py             # Manage memory lifecycle, consolidation
│
├── meta_learning/                      (NEW)
│   ├── __init__.py
│   ├── reptile.py                     # Reptile meta-learning algorithm
│   ├── task_sampler.py                # Sample learning tasks from experience
│   └── adaptation.py                  # Fast adaptation to new domains
│
└── tests/                              (ADD TO EXISTING)
    ├── test_beta_vae.py               (NEW)
    ├── test_active_inference.py       (NEW)
    ├── test_hopfield.py               (NEW)
    └── test_meta_learning.py          (NEW)
```

---

## Existing Files to MODIFY

### server.py — Add new endpoints (APPEND, don't rewrite)

Add these endpoint groups to the existing FastAPI app:

```python
# ============================================================
# β-VAE — Disentangled Representations
# ============================================================

@app.post("/beta-vae/encode")
# Takes text, returns disentangled latent vector
# Body: {"text": "AXIOM tried to email Andrew but Resend blocked it"}
# Response: {"latent_vector": [0.1, -0.3, ...], "dimensions": {"action_type": 0.82, "success": -0.91, "domain": 0.45, ...}}

@app.post("/beta-vae/similarity")
# Compare two experiences in disentangled space
# Body: {"text_a": "...", "text_b": "..."}
# Response: {"similarity": 0.73, "shared_factors": ["action_type", "domain"], "different_factors": ["outcome"]}

@app.post("/beta-vae/generate")
# Generate new experience descriptions from latent manipulation
# Body: {"base_text": "...", "modify": {"success": 1.0}}
# Response: {"generated": "If this had succeeded, the outcome would be..."}

@app.get("/beta-vae/stats")
# Response: {"total_encoded": 500, "latent_dim": 32, "beta": 4.0, "reconstruction_loss": 0.12}

# ============================================================
# Active Inference — Unified Curiosity + Planning
# ============================================================

@app.post("/active-inference/evaluate-policy")
# Evaluate a proposed action using Expected Free Energy
# Body: {"current_state": "...", "proposed_action": "read_codebase", "goal": "Audit axiom-backend"}
# Response: {
#   "expected_free_energy": -2.3,
#   "epistemic_value": 1.8,        (how much AXIOM would learn — curiosity)
#   "pragmatic_value": 0.5,        (how much it serves the goal — utility)
#   "ambiguity": 0.7,              (uncertainty about outcomes)
#   "risk": 0.3,                   (distance from preferred outcomes)
#   "recommendation": "proceed",
#   "confidence": 0.82
# }

@app.post("/active-inference/compare-policies")
# Compare multiple action options using EFE
# Body: {"current_state": "...", "policies": ["read_codebase", "propose_change", "research"], "goal": "..."}
# Response: {
#   "ranked_policies": [
#     {"action": "read_codebase", "efe": -2.3, "epistemic": 1.8, "pragmatic": 0.5},
#     {"action": "research", "efe": -1.1, "epistemic": 0.8, "pragmatic": 0.3},
#     {"action": "propose_change", "efe": 0.5, "epistemic": 0.1, "pragmatic": 0.4}
#   ],
#   "best_action": "read_codebase",
#   "exploration_exploitation_ratio": 0.78
# }

@app.post("/active-inference/update-beliefs")
# Update internal beliefs after observing outcome
# Body: {"action_taken": "read_codebase", "observation": "Found 3 performance issues", "was_expected": false}
# Response: {"belief_update_magnitude": 0.45, "surprise": 0.67, "precision_updated": true}

@app.get("/active-inference/status")
# Response: {"precision": 0.7, "exploration_tendency": 0.65, "total_inferences": 200}

# ============================================================
# Modern Hopfield Memory
# ============================================================

@app.post("/hopfield/store")
# Store an experience episode
# Body: {"content": "description of experience", "context": "goal/action context", "importance": 0.8}
# Response: {"pattern_id": "uuid", "total_patterns": 150}

@app.post("/hopfield/retrieve")
# Content-based retrieval — find most similar stored experiences
# Body: {"query": "code audit that found performance issues", "top_k": 5}
# Response: {
#   "retrieved": [
#     {"content": "...", "similarity": 0.92, "context": "...", "stored_at": "..."},
#     ...
#   ]
# }

@app.post("/hopfield/associate")
# Find associative connections — what experiences are linked?
# Body: {"pattern_id": "uuid"}
# Response: {"associations": [{"content": "...", "association_strength": 0.85}, ...]}

@app.get("/hopfield/stats")
# Response: {"total_patterns": 150, "capacity": 10000, "utilization": 0.015, "temperature": 1.0}

# ============================================================
# Meta-Learning (Reptile)
# ============================================================

@app.post("/meta-learning/adapt")
# Rapidly adapt to a new domain using Reptile-style few-shot learning
# Body: {"domain": "code_auditing", "examples": [...], "query": "what approach to use?"}
# Response: {"adapted_prediction": "...", "confidence": 0.78, "adaptation_steps": 5}

@app.post("/meta-learning/train-step")
# Run one Reptile meta-learning step
# Body: {"task_type": "action_selection"}
# Response: {"meta_loss": 0.23, "inner_steps": 5, "outer_step": 47}

@app.get("/meta-learning/status")
# Response: {"meta_steps": 47, "domains_seen": 8, "adaptation_speed": 0.85}
```

### config.py — Add new configuration (APPEND)

```python
# β-VAE
BETA_VAE_LATENT_DIM = 32              # Disentangled latent dimensions
BETA_VAE_HIDDEN_DIM = 256             # Encoder/decoder hidden size
BETA_VAE_BETA = 4.0                   # β > 1 forces disentanglement
BETA_VAE_CAPACITY_MAX = 25.0          # Max capacity for controlled β annealing (nats)
BETA_VAE_LR = 1e-4

# Active Inference
AI_PRECISION_INIT = 1.0               # Initial precision (inverse temperature)
AI_PRECISION_LR = 0.01                # Precision learning rate
AI_GAMMA = 1.0                        # Policy precision (softmax temperature)
AI_EPISTEMIC_WEIGHT = 0.5             # Weight for information gain vs utility
AI_PLANNING_HORIZON = 3               # How many steps ahead to evaluate

# Modern Hopfield
HOPFIELD_PATTERN_DIM = 384            # Must match embedding dim
HOPFIELD_MAX_PATTERNS = 10000         # Maximum stored patterns
HOPFIELD_BETA = 1.0                   # Inverse temperature (sharpness)
HOPFIELD_CONSOLIDATION_THRESHOLD = 0.95  # Similarity threshold for merging

# Meta-Learning (Reptile)
REPTILE_INNER_LR = 0.01              # Learning rate for task-specific adaptation
REPTILE_OUTER_LR = 0.001             # Learning rate for meta-update
REPTILE_INNER_STEPS = 5              # SGD steps per task
REPTILE_TASKS_PER_UPDATE = 4         # Tasks sampled per meta-update
```

---

## Implementation Details

### 1. β-VAE (beta_vae/model.py)

Variational Autoencoder with β > 1 to force disentangled latent representations.

**Architecture:**
```
Encoder: embedding(384) → Linear(384, 256) → ReLU → Linear(256, 256) → ReLU
         → Linear(256, 32) [μ]
         → Linear(256, 32) [log σ²]
         
z = μ + σ · ε,  where ε ~ N(0, I)    (reparameterization trick)

Decoder: Linear(32, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 384)
```

**Loss function (the core equation):**
```python
def loss(x, x_recon, mu, log_var, beta=4.0, capacity=25.0):
    # Reconstruction: how well can we reconstruct the input?
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence: how much does the posterior deviate from the prior?
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Controlled capacity: gradually increase allowed information
    # |KL - C| ensures KL stays near target capacity C
    total = recon_loss + beta * torch.abs(kl - capacity)
    return total, recon_loss, kl
```

**What each latent dimension learns to represent:**
After training on AXIOM's experiences, each of the 32 dimensions should capture one independent factor. We label them post-hoc by computing the correlation between each latent dimension and known features:
- Dimension 3 might correlate with "action_type" 
- Dimension 7 might correlate with "success/failure"
- Dimension 12 might correlate with "domain" (code vs research vs communication)

**Training data:** Encode all experiences from the world model buffer using the sentence-transformers embedding, then train the β-VAE on these 384-dim embeddings.

**Capacity annealing:** Start with capacity C=0, gradually increase to 25 nats over training. This progressively allows more information through the bottleneck while maintaining disentanglement.

### 2. Active Inference (active_inference/expected_free_energy.py)

This is the central equation that unifies curiosity and goal-directed behavior.

**Expected Free Energy for policy π:**
```python
def expected_free_energy(world_model, current_state, action, goal_embedding, precision):
    """
    G(π) = AMBIGUITY + RISK
         = -E[info_gain]  -  E[log p(preferred_outcomes)]
         = epistemic_value + pragmatic_value
    
    Lower G = better policy (more informative AND more goal-aligned)
    """
    # 1. Predict what would happen if we take this action
    predicted_state, prediction_uncertainty = world_model.predict_with_uncertainty(
        current_state, action
    )
    
    # 2. EPISTEMIC VALUE (curiosity) — how much would we learn?
    # = expected reduction in uncertainty about world state
    # Approximated as the entropy of the predicted state distribution
    epistemic = prediction_uncertainty  # High uncertainty = high learning value
    
    # 3. PRAGMATIC VALUE (goal-directed) — does it serve our goals?
    # = expected log-probability of preferred outcomes under this policy
    # Approximated as similarity between predicted outcome and goal
    goal_similarity = F.cosine_similarity(predicted_state, goal_embedding, dim=-1)
    pragmatic = goal_similarity
    
    # 4. PRECISION-WEIGHTED combination
    # High precision → exploit (favor pragmatic)
    # Low precision → explore (favor epistemic)
    G = -(precision * pragmatic + (1 - precision) * epistemic)
    
    return G, epistemic, pragmatic
```

**Policy selection via softmax:**
```python
def select_action(world_model, current_state, available_actions, goal, gamma=1.0):
    """Select action by sampling from softmax over negative EFE."""
    efes = []
    for action in available_actions:
        G, eps, prag = expected_free_energy(world_model, current_state, action, goal, precision)
        efes.append(G)
    
    # Softmax: P(π) = exp(-γ·G(π)) / Σ exp(-γ·G(π'))
    probs = F.softmax(-gamma * torch.tensor(efes), dim=0)
    return available_actions[torch.argmax(probs)], probs, efes
```

**Precision learning:**
Precision (γ) adapts based on prediction accuracy. When predictions are accurate → increase precision (exploit more). When predictions are poor → decrease precision (explore more).
```python
def update_precision(precision, prediction_error, lr=0.01):
    """Precision increases when predictions are accurate, decreases when wrong."""
    # Precision ~ inverse prediction error (smoothed)
    target_precision = 1.0 / (1.0 + prediction_error)
    precision = precision + lr * (target_precision - precision)
    return max(0.1, min(2.0, precision))  # Clamp to [0.1, 2.0]
```

**Integration with existing world model:**
The active inference module USES the existing world model from Phase 1 (`world_model/model.py`). It wraps the world model's predict method and adds uncertainty estimation + EFE computation.

### 3. Modern Hopfield Network (hopfield/modern_hopfield.py)

Exponential-capacity associative memory.

**Energy function:**
```python
def energy(xi, X, beta=1.0):
    """
    E(ξ) = -lse(β, X^T ξ) + ½||ξ||² + β⁻¹ log N + ½M²
    
    where lse = log-sum-exp, X = stored patterns matrix, ξ = query
    """
    similarities = beta * X @ xi  # (N,) — similarity to each stored pattern
    lse = torch.logsumexp(similarities, dim=0) / beta
    return -lse + 0.5 * xi @ xi
```

**Retrieval (one-step update):**
```python
def retrieve(xi, X, beta=1.0):
    """
    ξ_new = X · softmax(β · X^T · ξ)
    
    This IS the transformer attention mechanism with a theoretical grounding.
    """
    similarities = beta * X @ xi        # (N,)
    attention = F.softmax(similarities, dim=0)  # (N,)
    retrieved = X.T @ attention          # (D,)
    return retrieved, attention
```

**Storage:**
- Patterns stored as rows of matrix X (N × D)
- D = 384 (embedding dimension from sentence-transformers)
- N grows up to HOPFIELD_MAX_PATTERNS (10,000)
- When full, consolidate: merge similar patterns (cosine similarity > 0.95)

**Why this is better than the current replay buffer:**
- Current replay buffer: linear scan, random sampling
- Hopfield: content-addressed retrieval, one-step convergence, exponential capacity in dimension d
- Given d=384, theoretical capacity is astronomical — far more than 10,000 patterns

### 4. Reptile Meta-Learning (meta_learning/reptile.py)

Learns a meta-initialization so the world model can rapidly adapt to new domains.

**Algorithm:**
```python
def reptile_step(model, task_batch, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
    """
    One Reptile meta-update step.
    
    Φ ← Φ + ε(W̃ - Φ)
    
    where W̃ = result of k SGD steps on task T starting from Φ
    """
    meta_model_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    accumulated_diff = {k: torch.zeros_like(v) for k, v in meta_model_state.items()}
    
    for task in task_batch:
        # Clone current parameters
        model.load_state_dict({k: v.clone() for k, v in meta_model_state.items()})
        
        # Run k SGD steps on this task
        optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr)
        for step in range(inner_steps):
            loss = compute_task_loss(model, task)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Accumulate (W̃ - Φ)
        for k, v in model.state_dict().items():
            accumulated_diff[k] += (v - meta_model_state[k])
    
    # Average over tasks
    n_tasks = len(task_batch)
    for k in accumulated_diff:
        accumulated_diff[k] /= n_tasks
    
    # Meta-update: Φ ← Φ + ε · avg(W̃ - Φ)
    new_state = {}
    for k in meta_model_state:
        new_state[k] = meta_model_state[k] + outer_lr * accumulated_diff[k]
    
    model.load_state_dict(new_state)
    return model
```

**Task sampling:**
Each "task" is a subset of AXIOM's experiences filtered by domain (action_type or goal_type). Tasks are:
- "action_selection_code_audit" — predict best action for code audit goals
- "action_selection_research" — predict best action for research goals
- "outcome_prediction_propose_change" — predict outcomes of code proposals
- "outcome_prediction_build_and_test" — predict outcomes of build attempts

The meta-learned initialization allows rapid adaptation when AXIOM encounters a new domain she hasn't seen before.

---

## Database Schema Additions (shared/db.py)

ADD these tables to the existing cogcore.db (don't drop existing tables):

```sql
-- β-VAE training data
CREATE TABLE IF NOT EXISTS vae_training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB NOT NULL,            -- 384-dim float32
    source_text TEXT,
    action_type TEXT,
    was_successful INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Modern Hopfield patterns
CREATE TABLE IF NOT EXISTS hopfield_patterns (
    id TEXT PRIMARY KEY,                -- UUID
    pattern BLOB NOT NULL,              -- 384-dim float32
    content TEXT NOT NULL,              -- Human-readable description
    context TEXT,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Active inference state
CREATE TABLE IF NOT EXISTS inference_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    precision REAL NOT NULL,
    exploration_tendency REAL NOT NULL,
    total_inferences INTEGER DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Meta-learning checkpoints
CREATE TABLE IF NOT EXISTS meta_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meta_step INTEGER NOT NULL,
    domains_seen TEXT,                  -- JSON array
    meta_loss REAL,
    adaptation_speed REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Integration with Existing Phase 1 Code

### World Model integration:
- Active inference WRAPS the existing world model — calls `world_model.predict()` and adds uncertainty estimation
- β-VAE trains on embeddings from the world model's experience buffer
- Hopfield stores experiences that the world model has processed

### Curiosity integration:
- Active inference's epistemic value REPLACES the current curiosity score
- The existing `curiosity_manager.py` should have a new method that calls `/active-inference/evaluate-policy` and uses the epistemic component as the curiosity signal
- Keep the existing ICM/RND as fallbacks when active inference data is insufficient

### Self-model integration:
- The self-model should track precision (exploration vs exploitation tendency)
- Hopfield memory should be accessible from the self-model for capability assessment

---

## How Phase 2 Endpoints Get Called

The Node.js cognitive core ALREADY calls cogcore v2 at these points. Phase 2 adds MORE calls at the same integration points:

### During plan creation (existing integration point):
```
CURRENT: POST /abstraction/apply + POST /curiosity/evaluate-goal + GET /self-model/state
ADD: POST /active-inference/compare-policies  ← evaluates which actions to include in plan
```

### Before step execution (existing integration point):
```
CURRENT: POST /world-model/predict
ADD: POST /active-inference/evaluate-policy  ← should AXIOM proceed or try a different action?
```

### After step completion (existing integration point):
```
CURRENT: POST /world-model/update
ADD: POST /active-inference/update-beliefs   ← update precision based on surprise
ADD: POST /hopfield/store                    ← store experience in episodic memory
ADD: POST /beta-vae/encode                   ← encode experience for representation learning
```

---

## Testing

Add test files for each new module. Tests should:
- Test β-VAE encodes and decodes (reconstruction loss decreases)
- Test β-VAE produces disentangled representations (varying one latent dim changes one factor)
- Test Active Inference correctly ranks actions (epistemic vs pragmatic tradeoff)
- Test precision updates in the right direction (good predictions → higher precision)
- Test Hopfield stores and retrieves patterns correctly
- Test Hopfield retrieval is content-addressed (similar queries → similar results)
- Test Reptile meta-update moves parameters toward task solution
- Test all new API endpoints return correct response shapes

Run: `pytest tests/ -v` (should pass ALL tests including Phase 1's 60)

---

## What Claude Code Should Do

1. Read this PRD
2. Read the existing codebase first: `config.py`, `server.py`, `world_model/model.py`, `curiosity/curiosity_manager.py`, `shared/embeddings.py`, `shared/db.py`
3. Add new config values to `config.py` (APPEND, don't rewrite)
4. Create all new module directories and files
5. Add new database tables to `shared/db.py` init function (APPEND)
6. Add new endpoints to `server.py` (APPEND to existing app)
7. Write tests for all new modules
8. Run `pytest tests/ -v` — ALL tests must pass (Phase 1 + Phase 2)
9. Run `python -m uvicorn server:app --port 8080` — server must start clean with all components

DO NOT:
- Delete or rewrite any Phase 1 code
- Change existing endpoint signatures
- Remove existing tests
- Modify the world model architecture (Phase 2 wraps it, doesn't replace it)
- Break any existing functionality

CRITICAL: All PyTorch code must use `config.DEVICE` for GPU/CPU placement. The service runs on Railway (CPU) now but will move to RunPod GPU later.
