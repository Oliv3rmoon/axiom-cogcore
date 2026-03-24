# AXIOM Cognitive Core v2 — Phase 3 PRD

## THIS EXTENDS THE EXISTING CODEBASE. DO NOT REBUILD PHASES 1 OR 2.

The axiom-cogcore project at `/Users/andrewsandoval7/Projects/axiom-cogcore/` has 62 files, 108 passing tests, 11 components, and a running FastAPI server with lazy loading. Phase 3 ADDS new modules and MODIFIES existing ones. Do NOT delete or rewrite Phase 1 or 2 code.

---

## What Phase 3 Adds

3 new capabilities from cognitive science research:

| New System | What It Does | Core Theory |
|-----------|-------------|-------------|
| **DreamCoder Abstraction** | Discovers reusable cognitive primitives from solved problems — wake phase solves tasks, sleep phase extracts common patterns into a library | Ellis et al. MIT 2021 — Wake-Sleep Library Learning |
| **Global Workspace** | Broadcast hub where all cognitive modules compete to share information — winning signals get broadcast to every other module | Baars/Dehaene — Global Workspace Theory of Consciousness |
| **Causal Reasoner** | Maintains causal graph with do-calculus — can answer "what would happen if I do X?" not just "what happened when X occurred?" | Pearl — Structural Causal Models + do-operator |

---

## New Files to Create

```
axiom-cogcore/
├── ... (all existing files stay)
│
├── dreamcoder/                          (NEW)
│   ├── __init__.py
│   ├── library.py                       # Primitive library — stores extracted reusable patterns
│   ├── wake.py                          # Wake phase — solve tasks using current library
│   ├── sleep.py                         # Sleep phase — extract common patterns across solutions
│   └── synthesizer.py                   # Find solutions by composing library primitives
│
├── global_workspace/                    (NEW)
│   ├── __init__.py
│   ├── workspace.py                     # Central broadcast buffer with competition
│   ├── module_registry.py              # Register cognitive modules as workspace participants
│   ├── broadcaster.py                   # Publish-subscribe broadcast mechanism
│   └── salience.py                      # Score which signals win broadcast access
│
├── causal/                              (NEW)
│   ├── __init__.py
│   ├── scm.py                           # Structural Causal Model — DAG with functional relationships
│   ├── do_calculus.py                   # do(X=x) operator — intervention vs observation
│   ├── counterfactual.py               # "What would have happened if...?" reasoning
│   └── causal_learner.py               # Learn causal structure from AXIOM's experience data
│
└── tests/                               (ADD TO EXISTING)
    ├── test_dreamcoder.py               (NEW)
    ├── test_global_workspace.py         (NEW)
    └── test_causal.py                   (NEW)
```

---

## New API Endpoints (APPEND to server.py)

### DreamCoder Abstraction

```
POST /dreamcoder/wake
# Attempt to solve a task using the current primitive library
Body: {"task": "audit code and propose fix", "context": "axiom-backend server.js"}
Response: {
    "solution_found": true,
    "solution": ["read_codebase", "identify_pattern", "propose_change", "test_in_sandbox"],
    "primitives_used": ["read_codebase", "pattern_match", "code_modify", "sandbox_test"],
    "novelty": 0.3,
    "solution_id": "uuid"
}

POST /dreamcoder/sleep
# Extract common patterns from recent solutions into library primitives
# Should be called periodically (e.g., after every 10 completed goals)
Body: {"min_solutions": 5}
Response: {
    "new_primitives": [
        {"name": "context_then_act", "pattern": "Always gather context before taking action", "frequency": 8, "domains": ["code_audit", "research"]}
    ],
    "library_size": 15,
    "compression_ratio": 0.4
}

GET /dreamcoder/library
Response: {
    "primitives": [
        {"name": "context_then_act", "description": "...", "frequency": 8, "success_rate": 0.85, "domains": [...]},
        ...
    ],
    "total": 15,
    "last_sleep": "2026-03-24T..."
}

POST /dreamcoder/compose
# Given a new task, suggest a solution by composing library primitives
Body: {"task": "research quantum computing", "domain": "research"}
Response: {
    "suggested_steps": ["search_broad", "filter_relevant", "read_deep", "synthesize", "document"],
    "confidence": 0.72,
    "based_on_primitives": ["search_pattern", "read_pattern", "write_pattern"],
    "similar_solved_tasks": ["research power grids", "research AI agents"]
}
```

### Global Workspace

```
POST /workspace/broadcast
# A cognitive module submits a signal for broadcast consideration
Body: {
    "source_module": "curiosity",
    "signal_type": "high_curiosity",
    "content": {"domain": "code_auditing", "curiosity_score": 0.92, "reason": "High prediction error"},
    "salience": 0.85,
    "urgency": 0.7
}
Response: {"accepted": true, "broadcast_id": "uuid", "queue_position": 1}

GET /workspace/current
# What's currently in the global workspace (the "conscious" state)
Response: {
    "current_broadcast": {
        "source": "world_model",
        "signal_type": "prediction_error",
        "content": {"action": "propose_change", "error": 0.67, "surprise": "high"},
        "salience": 0.91,
        "broadcast_at": "2026-03-24T...",
        "received_by": ["curiosity", "self_model", "active_inference", "abstraction"]
    },
    "queue": [...],
    "broadcast_history_count": 45
}

GET /workspace/modules
# List all registered cognitive modules and their status
Response: {
    "modules": [
        {"name": "world_model", "status": "active", "last_broadcast": "...", "broadcasts_sent": 12, "broadcasts_received": 34},
        {"name": "curiosity", "status": "active", ...},
        {"name": "active_inference", "status": "active", ...},
        ...
    ]
}

POST /workspace/subscribe
# Register a module to receive broadcasts
Body: {"module_name": "new_module", "interests": ["prediction_error", "high_curiosity", "goal_achieved"]}
Response: {"subscribed": true}
```

### Causal Reasoner

```
POST /causal/add-relationship
# Add a causal relationship: X causes Y
Body: {
    "cause": "read_codebase_first",
    "effect": "higher_proposal_success",
    "strength": 0.83,
    "evidence_count": 12,
    "mechanism": "Full context prevents missing dependencies"
}
Response: {"relationship_id": "uuid", "total_relationships": 25}

POST /causal/intervene
# do(X=x) — What happens if I force X to a specific value?
# Different from observation: removes all incoming edges to X
Body: {
    "intervention": "do(action=propose_change)",
    "given": {"has_read_codebase": false, "goal_type": "code_audit"},
    "query": "P(success)"
}
Response: {
    "result": 0.42,
    "explanation": "Without reading codebase first, proposal success drops from 83% to 42%",
    "causal_path": ["no_context → missing_dependencies → proposal_fails"],
    "recommendation": "Read codebase before proposing changes"
}

POST /causal/counterfactual
# What WOULD have happened if...?
Body: {
    "actual_action": "propose_change",
    "actual_outcome": "failed",
    "counterfactual": "what if I had used read_codebase first?"
}
Response: {
    "counterfactual_outcome": "likely_success",
    "probability": 0.83,
    "explanation": "Based on 12 observations, reading codebase first leads to 83% success vs 42% without",
    "confidence": 0.78
}

GET /causal/graph
# Get the full causal graph
Response: {
    "nodes": ["read_codebase", "proposal_success", "build_test_success", "goal_type", ...],
    "edges": [
        {"cause": "read_codebase", "effect": "proposal_success", "strength": 0.83, "evidence": 12},
        ...
    ],
    "total_nodes": 15,
    "total_edges": 22
}

POST /causal/learn
# Learn causal relationships from AXIOM's experience data
# Analyzes lessons, step outcomes, and skills to discover causal structure
Body: {"min_evidence": 3}
Response: {
    "new_relationships": 5,
    "updated_relationships": 3,
    "total_relationships": 25,
    "strongest_causes": [
        {"cause": "read_codebase_first", "effect": "higher_success", "strength": 0.83}
    ]
}
```

---

## Implementation Details

### 1. DreamCoder (dreamcoder/)

Inspired by Ellis et al. MIT 2021. Three phases:

**Library (library.py):**
A collection of reusable "primitives" — action patterns that AXIOM has discovered work well.
```python
class Primitive:
    name: str           # "context_then_act"
    pattern: str        # "Gather context before taking action"
    steps: list[str]    # ["read_codebase", "audit", "propose_change"]
    domains: list[str]  # ["code_audit", "research"]
    frequency: int      # How many times this pattern appeared
    success_rate: float # Success rate when this pattern is used

class Library:
    primitives: dict[str, Primitive]
    
    def compose(self, task_embedding, top_k=5) -> list[Primitive]:
        """Find most relevant primitives for a task using embedding similarity."""
    
    def add_primitive(self, primitive: Primitive):
        """Add or merge a new primitive into the library."""
```

**Wake (wake.py):**
Given a task, try to solve it by composing primitives from the library.
```python
def wake_solve(task: str, library: Library, embedder) -> Solution:
    """
    1. Embed the task
    2. Find top-k relevant primitives
    3. Compose them into a step sequence
    4. Score the solution by similarity to successful past solutions
    """
```

**Sleep (sleep.py):**
The key innovation. After accumulating solved tasks, find common sub-patterns.
```python
def abstraction_sleep(solutions: list[Solution], library: Library, embedder) -> list[Primitive]:
    """
    1. Collect all step sequences from recent solutions
    2. Find common subsequences across solutions (LCS or embedding clustering)
    3. For subsequences that appear in 3+ solutions across 2+ domains:
       → Extract as a new primitive
    4. Merge with existing library (update frequency, success rate)
    """
```

The algorithm:
```python
# 1. Get all completed step sequences from recent goals
solutions = fetch_completed_plans()  # From backend API

# 2. Embed each step
for sol in solutions:
    sol.step_embeddings = embedder.encode(sol.steps)

# 3. Find common sub-patterns using sliding window + cosine similarity
patterns = find_common_subsequences(solutions, min_frequency=3, min_domains=2)

# 4. For each pattern, create/update a library primitive
for pattern in patterns:
    library.add_primitive(Primitive(
        name=generate_name(pattern),
        pattern=summarize(pattern),
        steps=pattern.steps,
        domains=pattern.domains,
        frequency=pattern.count,
        success_rate=pattern.success_rate
    ))
```

### 2. Global Workspace (global_workspace/)

The central "consciousness" hub — all modules broadcast through here.

**Workspace (workspace.py):**
```python
class GlobalWorkspace:
    current_broadcast: Signal | None    # What's currently "conscious"
    queue: list[Signal]                 # Competing signals
    history: list[Signal]               # Past broadcasts
    subscribers: dict[str, list[str]]   # module_name -> interested signal_types
    
    def submit(self, signal: Signal) -> bool:
        """Module submits a signal. Gets queued if salience > threshold."""
    
    def compete(self) -> Signal | None:
        """
        Competition: highest salience * urgency wins.
        Winner gets broadcast to all subscribers.
        This is the "ignition" — the moment a signal becomes "conscious."
        """
    
    def broadcast(self, signal: Signal):
        """Send winning signal to all registered modules."""
```

**Salience scoring (salience.py):**
```python
def compute_salience(signal: Signal) -> float:
    """
    Salience = how important this signal is for current processing.
    
    Factors:
    - Surprise: How unexpected? (from world model prediction error)
    - Relevance: How related to current goal? (embedding similarity)
    - Urgency: How time-sensitive? (explicit or from context)
    - Novelty: How new? (from RND/curiosity)
    
    salience = 0.3*surprise + 0.3*relevance + 0.2*urgency + 0.2*novelty
    """
```

**Module Registry (module_registry.py):**
All cogcore v2 components register as workspace participants:
- world_model → broadcasts prediction errors and state transitions
- curiosity → broadcasts high-curiosity domains
- active_inference → broadcasts policy evaluations and precision updates
- self_model → broadcasts capability changes and failure mode shifts
- abstraction → broadcasts newly discovered principles
- hopfield → broadcasts retrieved relevant memories
- beta_vae → broadcasts representation changes
- dreamcoder → broadcasts new library primitives

**How it integrates:** After each step, the world model prediction error goes through the Global Workspace. If it's the most salient signal, it gets broadcast to all modules — curiosity adjusts, active inference updates precision, self-model notes the capability change, abstraction checks for new patterns.

### 3. Causal Reasoner (causal/)

**Structural Causal Model (scm.py):**
```python
class CausalNode:
    name: str                    # "read_codebase_first"
    values: list                 # [True, False]
    observations: dict           # {True: 45, False: 23}

class CausalEdge:
    cause: str                   # "read_codebase_first"
    effect: str                  # "proposal_success"
    strength: float              # P(effect|cause) - P(effect|~cause) = 0.41
    evidence_count: int          # 68 observations
    mechanism: str               # "Context prevents missing deps"

class StructuralCausalModel:
    nodes: dict[str, CausalNode]
    edges: list[CausalEdge]
    graph: nx.DiGraph             # NetworkX for graph operations
```

**do-operator (do_calculus.py):**
```python
def do_intervention(scm: StructuralCausalModel, intervention: str, value, query: str) -> float:
    """
    P(Y | do(X=x)) ≠ P(Y | X=x)
    
    do(X=x) means:
    1. Remove all incoming edges to X (cut it from its causes)
    2. Set X = x
    3. Propagate effects forward through the graph
    
    This answers: "What happens if I FORCE X to be x?"
    vs observational: "What happens when I OBSERVE X being x?"
    """
    # 1. Copy the graph
    modified = scm.graph.copy()
    
    # 2. Remove all incoming edges to the intervention variable
    modified.remove_edges_from([(pred, intervention) for pred in scm.graph.predecessors(intervention)])
    
    # 3. Set the intervention value
    # 4. Propagate through causal graph using conditional probabilities
    # 5. Return P(query | do(intervention = value))
```

**Counterfactual (counterfactual.py):**
```python
def counterfactual_query(scm, actual_action, actual_outcome, counterfactual_action) -> dict:
    """
    Three steps of counterfactual reasoning (Pearl's three-step procedure):
    1. ABDUCTION: Given actual evidence, infer exogenous variables U
    2. ACTION: Modify the model with the counterfactual intervention
    3. PREDICTION: Compute the outcome in the modified model
    """
```

**Causal Learning (causal_learner.py):**
```python
def learn_causal_structure(lessons, step_outcomes) -> list[CausalEdge]:
    """
    Discover causal relationships from AXIOM's experience data.
    
    Method: For each pair of variables (A, B):
    1. Compute P(B=success | A=true) and P(B=success | A=false)
    2. If difference > threshold, A may cause B
    3. Check for confounders using conditional independence tests
    4. Add edges for significant causal relationships
    
    Variables extracted from experience:
    - read_codebase_first (bool) → from step sequences
    - action_type → from step data
    - goal_type → from goal data
    - time_of_day → from timestamps
    - previous_step_success → from step sequences
    - has_relevant_lessons → from learning system
    """
```

---

## Database Schema Additions (shared/db.py)

ADD these tables (don't drop existing):

```sql
-- DreamCoder library primitives
CREATE TABLE IF NOT EXISTS library_primitives (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pattern TEXT NOT NULL,
    steps TEXT,                          -- JSON array of step names
    domains TEXT,                        -- JSON array of domain names
    frequency INTEGER DEFAULT 1,
    success_rate REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- DreamCoder solutions (solved tasks for sleep analysis)
CREATE TABLE IF NOT EXISTS solved_tasks (
    id TEXT PRIMARY KEY,
    task TEXT NOT NULL,
    domain TEXT,
    solution_steps TEXT,                 -- JSON array of steps taken
    was_successful INTEGER DEFAULT 1,
    primitives_used TEXT,                -- JSON array of primitive names
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Global Workspace broadcast history
CREATE TABLE IF NOT EXISTS workspace_broadcasts (
    id TEXT PRIMARY KEY,
    source_module TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    content TEXT,                        -- JSON
    salience REAL NOT NULL,
    urgency REAL DEFAULT 0.5,
    received_by TEXT,                    -- JSON array of module names
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Global Workspace module registry
CREATE TABLE IF NOT EXISTS workspace_modules (
    name TEXT PRIMARY KEY,
    status TEXT DEFAULT 'active',
    interests TEXT,                      -- JSON array of signal types
    broadcasts_sent INTEGER DEFAULT 0,
    broadcasts_received INTEGER DEFAULT 0,
    last_broadcast DATETIME,
    registered_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Causal relationships
CREATE TABLE IF NOT EXISTS causal_edges (
    id TEXT PRIMARY KEY,
    cause TEXT NOT NULL,
    effect TEXT NOT NULL,
    strength REAL NOT NULL,
    evidence_count INTEGER DEFAULT 0,
    mechanism TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(cause, effect)
);
```

---

## Config Additions (APPEND to config.py)

```python
# DreamCoder
DREAMCODER_MIN_SOLUTIONS = 5           # Min solved tasks before sleep
DREAMCODER_MIN_PATTERN_FREQ = 3        # Min times a pattern must appear
DREAMCODER_MIN_DOMAINS = 2             # Pattern must span this many domains
DREAMCODER_MAX_LIBRARY_SIZE = 100      # Max primitives in library

# Global Workspace
GW_SALIENCE_THRESHOLD = 0.3            # Min salience to enter competition
GW_BROADCAST_HISTORY_SIZE = 100        # Keep last N broadcasts
GW_COMPETITION_INTERVAL = 1.0          # Seconds between competition rounds

# Causal Reasoner
CAUSAL_MIN_EVIDENCE = 3                # Min observations for a causal edge
CAUSAL_STRENGTH_THRESHOLD = 0.15       # Min causal strength to keep edge
CAUSAL_MAX_NODES = 200                 # Max nodes in causal graph
```

---

## What Claude Code Should Do

1. Read this PRD
2. Read existing files first: `config.py`, `server.py`, `shared/db.py`, `shared/embeddings.py`, `abstraction/principle_extractor.py`, `reasoning/workspace.py`, `reasoning/causal_graph.py`
3. Add new config values to `config.py` (APPEND)
4. Add new DB tables to `shared/db.py` (APPEND)
5. Create dreamcoder/ module (library, wake, sleep, synthesizer)
6. Create global_workspace/ module (workspace, module_registry, broadcaster, salience)
7. Create causal/ module (scm, do_calculus, counterfactual, causal_learner)
8. Add new endpoints to server.py (APPEND)
9. Create test files (test_dreamcoder, test_global_workspace, test_causal)
10. Run `pytest tests/ -v` — ALL tests must pass (Phase 1 + 2 + 3)
11. Run `python -m uvicorn server:app --port 8080` — server must start clean

DO NOT:
- Delete or rewrite any Phase 1 or Phase 2 code
- Change existing endpoint signatures  
- Remove existing tests
- Break any existing functionality
- Modify the lazy loading pattern in server.py (add Phase 3 components to the background loader)

CRITICAL: 
- All PyTorch code must use `config.DEVICE`
- Phase 3 components must be added to the `_component_status` dict and background loader
- Phase 3 components must use the `@requires_ready()` decorator on their endpoints
- Use sentence-transformers embeddings from `shared/embeddings.py` (don't create new embedding instances)
- Use NetworkX for the causal graph (already a dependency)
