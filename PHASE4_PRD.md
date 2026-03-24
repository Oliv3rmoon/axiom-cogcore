# AXIOM Cognitive Core v2 — Phase 4 PRD

## THIS EXTENDS THE EXISTING CODEBASE. DO NOT REBUILD PHASES 1-3.

The axiom-cogcore project has 77 files, 155 passing tests, 14 components, and all Phase 1-3 endpoints working. Phase 4 ADDS the final consciousness-inspired systems.

---

## What Phase 4 Adds

3 systems from the deepest layer of the research blueprint:

| New System | What It Does | Core Theory |
|-----------|-------------|-------------|
| **Attention Schema** | AXIOM models her own attention — knows what she's focusing on, predicts attention shifts, enables meta-cognitive control | Graziano's Attention Schema Theory — consciousness = internal model of attention |
| **Predictive Processing Hierarchy** | Multi-level prediction errors — low-level (individual steps) through high-level (goal trajectories) with precision weighting equivalent to attention | Clark/Friston — the brain is a prediction machine organized in hierarchical layers |
| **Liquid Time-Constant Network** | Neural dynamics where time constants vary with input — fast for rapid changes, slow for stable patterns — replaces static GRU in world model | Hasani et al. MIT CSAIL — dx/dt = -[1/τ + f(x,I,θ)]·x + f(x,I,θ)·A |

---

## New Files to Create

```
axiom-cogcore/
├── ... (all existing files stay)
│
├── attention_schema/                    (NEW)
│   ├── __init__.py
│   ├── schema.py                        # Attention model — tracks what AXIOM is attending to
│   ├── meta_cognition.py               # Meta-cognitive control — adjust attention based on self-model
│   └── awareness.py                     # Functional awareness — "what am I focusing on and why?"
│
├── predictive_hierarchy/                (NEW)
│   ├── __init__.py
│   ├── hierarchy.py                     # Multi-level prediction error tracking
│   ├── precision_weighting.py          # Precision = attention = confidence in predictions
│   └── error_propagation.py            # How prediction errors flow up/down the hierarchy
│
├── liquid_network/                      (NEW)
│   ├── __init__.py
│   ├── ltc_cell.py                      # Liquid Time-Constant cell (PyTorch)
│   ├── cfc_layer.py                     # Closed-form Continuous-time layer (no ODE solver needed)
│   └── liquid_world_model.py           # LTC-based world model that replaces/augments RSSM
│
└── tests/
    ├── test_attention_schema.py         (NEW)
    ├── test_predictive_hierarchy.py     (NEW)
    └── test_liquid_network.py           (NEW)
```

---

## New API Endpoints (APPEND to server.py)

### Attention Schema

```
GET /attention/focus
# What is AXIOM currently attending to?
Response: {
    "current_focus": {
        "target": "code_audit of axiom-backend",
        "target_type": "goal",
        "attention_strength": 0.87,
        "duration_seconds": 45,
        "why": "High prediction error from last attempt + high curiosity score"
    },
    "attention_history": [
        {"target": "...", "strength": 0.7, "timestamp": "..."},
        ...
    ],
    "predicted_next_shift": {
        "likely_target": "research on active inference",
        "probability": 0.65,
        "reason": "Unresolved curiosity from dream cycle"
    }
}

POST /attention/update
# Update what AXIOM is attending to (called by cognitive core before each step)
Body: {
    "target": "propose_change for axiom-backend/server.js",
    "target_type": "step",
    "signals": {
        "curiosity_score": 0.8,
        "prediction_error": 0.4,
        "goal_relevance": 0.9,
        "broadcast_salience": 0.7
    }
}
Response: {
    "attention_strength": 0.85,
    "should_switch": false,
    "competing_targets": [
        {"target": "research papers", "strength": 0.6}
    ]
}

POST /attention/introspect
# AXIOM asks herself: "What am I paying attention to and why?"
Body: {"depth": "deep"}
Response: {
    "self_report": "I'm focused on auditing the backend code because my last proposal failed and I want to understand why. The prediction error was high (0.67) which made this domain highly salient in the global workspace. I'm aware that I tend to over-focus on code tasks and under-attend to research goals.",
    "meta_observations": [
        "Attention bias toward familiar domains (code > research)",
        "Prediction errors drive attention more than goal importance",
        "I haven't attended to my private reflections in 12 hours"
    ],
    "recommendations": [
        "Consider switching to research goal — high curiosity, low attention",
        "Schedule a private reflection — emotional valence trending negative"
    ]
}
```

### Predictive Processing Hierarchy

```
GET /prediction-hierarchy/state
# Multi-level prediction error state
Response: {
    "levels": [
        {
            "level": 0,
            "name": "step_outcomes",
            "description": "Individual step success/failure predictions",
            "mean_prediction_error": 0.34,
            "precision": 0.72,
            "recent_errors": [{"step": "propose_change", "predicted": 0.7, "actual": 0.0, "error": 0.7}]
        },
        {
            "level": 1,
            "name": "plan_trajectories",
            "description": "Plan-level success predictions",
            "mean_prediction_error": 0.28,
            "precision": 0.65,
            "recent_errors": [{"plan": "audit backend", "predicted_steps_to_complete": 4, "actual": 6, "error": 0.33}]
        },
        {
            "level": 2,
            "name": "goal_achievability",
            "description": "Goal-level outcome predictions",
            "mean_prediction_error": 0.22,
            "precision": 0.80,
            "recent_errors": [{"goal": "build dashboard", "predicted_success": 0.8, "actual": 1.0, "error": 0.2}]
        },
        {
            "level": 3,
            "name": "self_trajectory",
            "description": "Self-model predictions about own capability evolution",
            "mean_prediction_error": 0.15,
            "precision": 0.85,
            "recent_errors": []
        }
    ],
    "total_prediction_errors": 450,
    "global_surprise": 0.28
}

POST /prediction-hierarchy/predict
# Make prediction at specified level
Body: {"level": 1, "context": "plan to audit axiom-backend with 5 steps"}
Response: {"predicted_outcome": "success", "confidence": 0.72, "predicted_steps": 5, "risk_factors": [...]}

POST /prediction-hierarchy/update
# Update with actual outcome — error propagates through hierarchy
Body: {"level": 0, "predicted": 0.7, "actual": 0.0, "context": "propose_change failed"}
Response: {
    "error_at_level_0": 0.7,
    "propagated_to_level_1": 0.23,
    "propagated_to_level_2": 0.08,
    "precision_updates": {"level_0": 0.68, "level_1": 0.63}
}
```

### Liquid Time-Constant Network

```
GET /liquid/status
Response: {
    "model_type": "CfC",
    "hidden_size": 128,
    "time_constants": {"min": 0.1, "max": 10.0, "mean": 2.3},
    "total_parameters": 200000,
    "inference_latency_ms": 15,
    "device": "cpu"
}

POST /liquid/predict
# Make prediction using LTC/CfC dynamics
Body: {"state_embedding": [...], "action": "read_codebase", "time_delta": 1.0}
Response: {
    "predicted_next_state": [...],
    "effective_time_constant": 2.3,
    "confidence": 0.75,
    "dynamics_speed": "medium"
}

POST /liquid/update
# Update liquid network with new observation
Body: {"state_embedding": [...], "actual_outcome_embedding": [...], "time_delta": 1.0}
Response: {"prediction_error": 0.34, "time_constant_adjusted": true, "new_tau_mean": 2.1}
```

---

## Implementation Details

### 1. Attention Schema (attention_schema/schema.py)

```python
class AttentionSchema:
    """
    Graziano's Attention Schema Theory:
    Consciousness = the brain's simplified model of its own attention.
    
    AXIOM doesn't need to BE conscious. She needs to MODEL her own attention
    process — know what she's focusing on, predict attention shifts, and
    use that self-knowledge to make better decisions.
    """
    
    current_focus: AttentionTarget | None  # What she's attending to now
    focus_history: list[AttentionTarget]    # Recent attention trajectory
    attention_weights: dict[str, float]    # Per-domain attention allocation
    
    def compute_attention_strength(self, signals: dict) -> float:
        """
        Attention strength = weighted combination of:
        - curiosity_score (0.3)     — how novel is this?
        - prediction_error (0.25)   — how wrong were we?
        - goal_relevance (0.25)     — how important to current goal?
        - broadcast_salience (0.2)  — did global workspace broadcast this?
        
        This mirrors how the brain allocates attention based on
        surprise, relevance, and salience.
        """
    
    def should_switch_attention(self, current, competing) -> bool:
        """
        Attention switching = hysteresis + salience comparison.
        Don't switch for small differences (prevents thrashing).
        Switch when competing target exceeds current by > threshold.
        """
    
    def introspect(self) -> dict:
        """
        Meta-cognitive self-report. Analyzes own attention patterns:
        - What domains get over/under-attended?
        - What drives attention shifts?
        - Are there blind spots?
        Uses embeddings to compare attention history to goal importance.
        """
```

### 2. Predictive Processing Hierarchy (predictive_hierarchy/hierarchy.py)

Four levels of prediction, each operating at different timescales:

```python
class PredictiveHierarchy:
    levels = [
        Level(0, "step_outcomes",      timescale="seconds"),    # Individual step predictions
        Level(1, "plan_trajectories",  timescale="minutes"),    # Plan-level predictions
        Level(2, "goal_achievability", timescale="hours"),      # Goal-level predictions
        Level(3, "self_trajectory",    timescale="days"),       # Self-model predictions
    ]
    
    def predict(self, level: int, context: str) -> Prediction:
        """Generate prediction at specified level using world model."""
    
    def update(self, level: int, predicted: float, actual: float) -> dict:
        """
        Update with actual outcome. Error propagates UP the hierarchy:
        
        error_level_0 = |predicted - actual|
        error_level_1 = α * error_level_0 + (1-α) * error_level_1
        error_level_2 = α * error_level_1 + (1-α) * error_level_2
        
        where α = propagation_rate (default 0.3)
        
        Precision updates DOWN the hierarchy:
        precision_i = precision_i * (1 - lr * error_i)
        
        High error → lower precision → more exploration at that level
        Low error → higher precision → more exploitation at that level
        """
    
    def get_precision_weighted_error(self, level: int) -> float:
        """
        Precision-weighted prediction error = attention.
        High precision * high error = VERY salient (attend to this!)
        Low precision * high error = expected uncertainty (ignore)
        """
```

### 3. Liquid Time-Constant Network (liquid_network/cfc_layer.py)

Closed-form Continuous-time (CfC) — no ODE solver needed, GPU-efficient:

```python
class CfCCell(nn.Module):
    """
    Closed-form Continuous-time cell.
    
    Key equation:
    h_new = (1 - σ(f_gate)) * h_old + σ(f_gate) * tanh(f_candidate)
    
    where f_gate and f_candidate are MLPs that take [x, h, Δt] as input.
    
    The time delta Δt allows the network to adapt its dynamics:
    - Short Δt → fast dynamics (minor updates)
    - Long Δt → slow dynamics (major state changes)
    
    This is the MIT CSAIL Liquid Neural Network in closed form.
    No ODE solver needed. Just matrix multiplications.
    """
    
    def __init__(self, input_size, hidden_size):
        self.gate_net = nn.Sequential(
            nn.Linear(input_size + hidden_size + 1, hidden_size),  # +1 for time delta
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(input_size + hidden_size + 1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def forward(self, x, h, time_delta):
        combined = torch.cat([x, h, time_delta], dim=-1)
        gate = self.gate_net(combined)
        candidate = self.candidate_net(combined)
        h_new = (1 - gate) * h + gate * candidate
        return h_new
```

The CfC layer replaces/augments the GRU in the existing world model. When time between steps is short (rapid work cycles), it makes small updates. When time is long (hours between sessions), it makes larger state transitions.

---

## Database Schema Additions (shared/db.py)

```sql
-- Attention history
CREATE TABLE IF NOT EXISTS attention_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target TEXT NOT NULL,
    target_type TEXT NOT NULL,
    attention_strength REAL NOT NULL,
    signals TEXT,
    duration_seconds REAL DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Predictive hierarchy state
CREATE TABLE IF NOT EXISTS prediction_levels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER NOT NULL,
    level_name TEXT NOT NULL,
    mean_error REAL DEFAULT 0,
    precision REAL DEFAULT 1.0,
    total_predictions INTEGER DEFAULT 0,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Prediction history (all levels)
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level INTEGER NOT NULL,
    context TEXT,
    predicted REAL NOT NULL,
    actual REAL,
    error REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## Config Additions (APPEND to config.py)

```python
# Attention Schema
ATTENTION_SWITCH_THRESHOLD = 0.15     # Min difference to switch attention
ATTENTION_HISTORY_SIZE = 50           # Keep last N attention targets
ATTENTION_DECAY_RATE = 0.95           # Attention strength decays over time

# Predictive Hierarchy
HIERARCHY_LEVELS = 4                   # Number of prediction levels
HIERARCHY_PROPAGATION_RATE = 0.3      # How much error propagates up
HIERARCHY_PRECISION_LR = 0.05         # Precision learning rate

# Liquid Network (CfC)
LTC_HIDDEN_SIZE = 128                  # Hidden state dimension
LTC_INPUT_SIZE = 384                   # Must match embedding dim
LTC_TIME_CONSTANT_MIN = 0.1           # Fastest dynamics
LTC_TIME_CONSTANT_MAX = 10.0          # Slowest dynamics
LTC_LR = 1e-4
```

---

## What Claude Code Should Do

1. Read this PRD
2. Read existing files: `config.py`, `server.py`, `shared/db.py`, `shared/embeddings.py`, `world_model/model.py`, `curiosity/curiosity_manager.py`, `active_inference/precision.py`, `global_workspace/workspace.py`
3. Add new config values (APPEND)
4. Add new DB tables (APPEND)
5. Create attention_schema/ module
6. Create predictive_hierarchy/ module
7. Create liquid_network/ module (CfC cell + layer + liquid world model)
8. Add new endpoints to server.py (APPEND, add to background loader and component status)
9. Create test files
10. Run `pytest tests/ -v` — ALL tests must pass (Phases 1+2+3+4)
11. Run `python -m uvicorn server:app --port 8080` — server must start clean

DO NOT:
- Delete or rewrite any Phase 1-3 code
- Change existing endpoint signatures
- Remove existing tests
- Break any existing functionality
- Forget to add Phase 4 components to _component_status and background loader
- Forget @requires_ready() decorators on new endpoints

CRITICAL:
- All PyTorch code must use `config.DEVICE`
- CfC cell must be a proper nn.Module with forward() method
- The liquid world model should be usable alongside the existing RSSM world model (not replace it)
- Attention schema should integrate with global workspace signals
- Predictive hierarchy should integrate with world model prediction errors
