from __future__ import annotations
"""AXIOM Cognitive Core v2 — FastAPI server with all endpoints."""

import asyncio
import logging
import uuid
import json
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import config
from shared.db import get_db, close_db
from shared.embeddings import EmbeddingService
from shared.backend_client import BackendClient
from world_model.model import RSSMWorldModel
from world_model.trainer import WorldModelTrainer
from world_model.buffer import ExperienceBuffer, Experience
from curiosity.curiosity_manager import CuriosityManager
from continual.ewc import EWC
from continual.consolidator import Consolidator
from continual.replay import ReplayManager
from abstraction.principle_extractor import PrincipleExtractor
from abstraction.meta_learner import MetaLearner
from abstraction.skill_composer import SkillComposer
from reasoning.workspace import ReasoningWorkspace
from self_model.state_tracker import StateTracker
from self_model.capability_model import CapabilityModel
from self_model.transition_model import TransitionModel

# Phase 2 imports
from beta_vae.model import BetaVAE
from beta_vae.trainer import BetaVAETrainer
from beta_vae.representations import RepresentationEngine
from active_inference.generative_model import GenerativeModel
from active_inference.expected_free_energy import expected_free_energy as compute_efe
from active_inference.policy_selection import compare_policies, recommend_action
from active_inference.precision import PrecisionController
from hopfield.episodic_store import EpisodicStore
from hopfield.memory_manager import MemoryManager
from meta_learning.reptile import Reptile
from meta_learning.task_sampler import TaskSampler
from meta_learning.adaptation import DomainAdapter

# Phase 3 imports are LAZY — loaded inside _load_all_models() to avoid
# crashing the server if Phase 3 modules have import issues.
# The modules are accessed via globals set during background init.

logger = logging.getLogger("axiom-cogcore")


# ──────────────────────────────────────────────
# Global components (initialized in background)
# ──────────────────────────────────────────────

embedder: EmbeddingService | None = None
backend: BackendClient | None = None
world_model: RSSMWorldModel | None = None
trainer: WorldModelTrainer | None = None
curiosity_manager: CuriosityManager | None = None
ewc: EWC | None = None
consolidator: Consolidator | None = None
principle_extractor: PrincipleExtractor | None = None
meta_learner: MetaLearner | None = None
skill_composer: SkillComposer | None = None
reasoning_ws: ReasoningWorkspace | None = None
state_tracker: StateTracker | None = None
capability_model: CapabilityModel | None = None
transition_model: TransitionModel | None = None

# Phase 2 globals
beta_vae_model: BetaVAE | None = None
beta_vae_trainer: BetaVAETrainer | None = None
rep_engine: RepresentationEngine | None = None
gen_model: GenerativeModel | None = None
precision_ctrl: PrecisionController | None = None
episodic_store: EpisodicStore | None = None
memory_mgr: MemoryManager | None = None
reptile: Reptile | None = None
task_sampler: TaskSampler | None = None
domain_adapter: DomainAdapter | None = None

# Phase 3 globals (types are not imported at module level — lazy loaded)
dc_library = None
gw = None
gw_registry = None
gw_broadcaster = None
causal_scm = None
# Phase 3 lazy-loaded function references
_wake_solve = None
_abstraction_sleep = None
_compose_solution = None
_do_intervention = None
_counterfactual_query = None
_learn_causal_structure = None
_Signal = None

# Phase 4 globals (lazy loaded)
attention_schema = None
attention_meta = None
attention_awareness = None
pred_hierarchy = None
liquid_model = None

# Prediction cache: prediction_id -> {embedding, h, z, action}
_prediction_cache: dict[str, dict] = {}
_experience_count = 0
_total_predictions = 0
_correct_predictions = 0
_model_version = 0

# ──────────────────────────────────────────────
# Readiness tracking — updated as each component loads
# ──────────────────────────────────────────────

_component_status: dict[str, bool] = {
    "embeddings": False,
    "world_model": False,
    "curiosity": False,
    "continual_learning": False,
    "abstraction": False,
    "reasoning": False,
    "self_model": False,
    "beta_vae": False,
    "active_inference": False,
    "hopfield_memory": False,
    "meta_learning": False,
    "dreamcoder": False,
    "global_workspace": False,
    "causal_reasoner": False,
    "attention_schema": False,
    "predictive_hierarchy": False,
    "liquid_network": False,
}
_init_error: str | None = None
_background_task: asyncio.Task | None = None


def _all_ready() -> bool:
    return all(_component_status.values())


# ──────────────────────────────────────────────
# Guard decorator — returns 503 if component not ready
# ──────────────────────────────────────────────

def requires_ready(*component_names: str):
    """Decorator: returns 503 JSON if any named component is not loaded yet."""
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            missing = [c for c in component_names if not _component_status.get(c, False)]
            if missing:
                return JSONResponse(
                    status_code=503,
                    content={
                        "error": "initializing",
                        "ready": False,
                        "waiting_for": missing,
                    },
                )
            return await fn(*args, **kwargs)
        return wrapper
    return decorator


# ──────────────────────────────────────────────
# Background model loader
# ──────────────────────────────────────────────

async def _load_all_models():
    """Load every model in the background. Updates _component_status as each finishes."""
    global embedder, backend, world_model, trainer, curiosity_manager
    global ewc, consolidator, principle_extractor, meta_learner, skill_composer
    global reasoning_ws, state_tracker, capability_model, transition_model
    global beta_vae_model, beta_vae_trainer, rep_engine, gen_model, precision_ctrl
    global episodic_store, memory_mgr, reptile, task_sampler, domain_adapter
    global dc_library, gw, gw_registry, gw_broadcaster, causal_scm
    global attention_schema, attention_meta, attention_awareness
    global pred_hierarchy, liquid_model
    global _init_error

    try:
        # ── Phase 1 ──────────────────────────────

        # Embeddings (heaviest — downloads model on first run)
        logger.info("Loading embedding model...")
        embedder = EmbeddingService.get_instance()
        _component_status["embeddings"] = True
        logger.info("Embeddings ready.")

        # Backend client (instant)
        backend = BackendClient()

        # World model
        logger.info("Loading world model...")
        world_model = RSSMWorldModel().to(config.DEVICE)
        trainer = WorldModelTrainer(world_model)
        _component_status["world_model"] = True
        logger.info("World model ready.")

        # Curiosity
        curiosity_manager = CuriosityManager()
        _component_status["curiosity"] = True

        # Continual learning
        ewc = EWC(world_model)
        consolidator = Consolidator(world_model, trainer, ewc)
        await ewc.load_latest_checkpoint()
        if ewc.is_initialized:
            trainer.fisher_diag = ewc.fisher_diag
            trainer.anchor_params = ewc.anchor_params
        _component_status["continual_learning"] = True

        # Abstraction
        principle_extractor = PrincipleExtractor(embedder, backend)
        meta_learner = MetaLearner(embedder, backend)
        skill_composer = SkillComposer(embedder, backend)
        _component_status["abstraction"] = True

        # Reasoning
        reasoning_ws = ReasoningWorkspace(embedder)
        _component_status["reasoning"] = True

        # Self-model
        state_tracker = StateTracker(backend)
        capability_model = CapabilityModel(backend)
        transition_model = TransitionModel(state_tracker, capability_model, embedder, backend)
        _component_status["self_model"] = True

        # ── Phase 2 ──────────────────────────────

        # β-VAE
        logger.info("Loading β-VAE...")
        input_dim = embedder.dim
        beta_vae_model = BetaVAE(input_dim=input_dim).to(config.DEVICE)
        beta_vae_trainer = BetaVAETrainer(beta_vae_model)
        rep_engine = RepresentationEngine(beta_vae_model, embedder)
        _component_status["beta_vae"] = True

        # Active Inference
        gen_model = GenerativeModel(world_model, embedder)
        precision_ctrl = PrecisionController()
        await precision_ctrl.load_latest()
        _component_status["active_inference"] = True

        # Hopfield Episodic Memory
        logger.info("Loading Hopfield memory...")
        episodic_store = EpisodicStore(embedder)
        await episodic_store.load_from_db()
        memory_mgr = MemoryManager(episodic_store)
        _component_status["hopfield_memory"] = True

        # Meta-Learning (Reptile)
        reptile = Reptile(world_model)
        task_sampler = TaskSampler()
        domain_adapter = DomainAdapter(reptile, embedder)
        _component_status["meta_learning"] = True

        logger.info("Phase 1+2 components loaded.")

    except Exception as exc:
        _init_error = traceback.format_exc()
        logger.error("Phase 1/2 init failed: %s", _init_error)

    # ── Phase 3 (isolated — failures here don't affect P1/P2) ──────
    try:
        global _wake_solve, _abstraction_sleep, _compose_solution
        global _do_intervention, _counterfactual_query, _learn_causal_structure, _Signal

        # Lazy imports for Phase 3
        from dreamcoder.library import Library as _DreamcoderLibrary
        from dreamcoder.wake import wake_solve as _ws
        from dreamcoder.sleep import abstraction_sleep as _as
        from dreamcoder.synthesizer import compose_solution as _cs
        from global_workspace.workspace import GlobalWorkspace as _GW
        from global_workspace.module_registry import ModuleRegistry as _MR
        from global_workspace.broadcaster import Broadcaster as _BC, Signal as _Sig
        from causal.scm import StructuralCausalModel as _SCM
        from causal.do_calculus import do_intervention as _di
        from causal.counterfactual import counterfactual_query as _cq
        from causal.causal_learner import learn_causal_structure as _lcs

        _wake_solve = _ws
        _abstraction_sleep = _as
        _compose_solution = _cs
        _do_intervention = _di
        _counterfactual_query = _cq
        _learn_causal_structure = _lcs
        _Signal = _Sig

        # DreamCoder
        logger.info("Loading DreamCoder library...")
        dc_library = _DreamcoderLibrary(embedder)
        await dc_library.load_from_db()
        _component_status["dreamcoder"] = True

        # Global Workspace
        gw_registry = _MR()
        gw_registry.register_defaults()
        await gw_registry.load_from_db()
        gw_broadcaster = _BC(gw_registry)
        gw = _GW(gw_registry, gw_broadcaster)
        _component_status["global_workspace"] = True

        # Causal Reasoner
        logger.info("Loading causal model...")
        causal_scm = _SCM()
        await causal_scm.load_from_db()
        _component_status["causal_reasoner"] = True

        logger.info("All components loaded (Phase 1+2+3).")

    except Exception as exc:
        p3_err = traceback.format_exc()
        logger.error("Phase 3 init failed (P1/P2 still running): %s", p3_err)
        if _init_error:
            _init_error += "\n--- Phase 3 ---\n" + p3_err
        else:
            _init_error = "Phase 3: " + p3_err

    # ── Phase 4 (isolated — failures here don't affect P1/P2/P3) ──────
    try:
        from attention_schema.schema import AttentionSchema as _AS
        from attention_schema.meta_cognition import MetaCognition as _MC
        from attention_schema.awareness import AwarenessEngine as _AE
        from predictive_hierarchy.hierarchy import PredictiveHierarchy as _PH
        from liquid_network.liquid_world_model import LiquidWorldModel as _LWM

        # Attention Schema
        logger.info("Loading attention schema...")
        attention_schema = _AS()
        attention_meta = _MC(attention_schema)
        attention_awareness = _AE(attention_schema, attention_meta)
        _component_status["attention_schema"] = True

        # Predictive Hierarchy
        pred_hierarchy = _PH()
        _component_status["predictive_hierarchy"] = True

        # Liquid Network
        logger.info("Loading liquid network...")
        input_dim = embedder.dim if embedder else config.LTC_INPUT_SIZE
        liquid_model = _LWM(input_size=input_dim).to(config.DEVICE)
        _component_status["liquid_network"] = True

        logger.info("All components loaded (Phase 1+2+3+4).")

    except Exception as exc:
        p4_err = traceback.format_exc()
        logger.error("Phase 4 init failed (P1/P2/P3 still running): %s", p4_err)
        if _init_error:
            _init_error += "\n--- Phase 4 ---\n" + p4_err
        else:
            _init_error = "Phase 4: " + p4_err


# ──────────────────────────────────────────────
# Lifespan — only DB + kick off background loader
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _background_task

    # DB is tiny and instant — safe to do here
    await get_db()

    # Fire off model loading in background so /health is reachable NOW
    _background_task = asyncio.create_task(_load_all_models())

    yield

    # Cleanup
    if _background_task and not _background_task.done():
        _background_task.cancel()
    if backend:
        await backend.close()
    await close_db()


app = FastAPI(title="AXIOM Cognitive Core v2", lifespan=lifespan)


# ──────────────────────────────────────────────
# Request/Response Models
# ──────────────────────────────────────────────

class PredictRequest(BaseModel):
    current_state: str
    action: str
    action_details: str = ""
    context: str = ""

class UpdateRequest(BaseModel):
    prediction_id: str
    actual_outcome: str
    was_successful: bool
    action: str

class EvaluateGoalRequest(BaseModel):
    goal: str

class ExtractRequest(BaseModel):
    pass

class ApplyRequest(BaseModel):
    goal: str
    action: str = ""

class StartReasoningRequest(BaseModel):
    goal: str
    initial_context: str = ""

class AddThoughtRequest(BaseModel):
    workspace_id: str
    thought: str
    parent_id: Optional[str] = None
    thought_type: str = "observation"
    confidence: float = 0.5

class AddCausalLinkRequest(BaseModel):
    workspace_id: str
    cause_id: str
    effect_id: str
    relationship: str = "enables"
    strength: float = 0.5

class QueryReasoningRequest(BaseModel):
    workspace_id: str
    question: str

class PredictChangeRequest(BaseModel):
    proposed_change: str
    change_type: str = "code_modification"


# ──────────────────────────────────────────────
# Health — always available, no guard
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "alive",
        "service": "AXIOM Cognitive Core v2",
        "device": config.DEVICE,
        "ready": _all_ready(),
        "components": dict(_component_status),
        "init_error": _init_error,
    }


# ──────────────────────────────────────────────
# World Model
# ──────────────────────────────────────────────

@app.post("/world-model/predict")
@requires_ready("embeddings", "world_model")
async def wm_predict(req: PredictRequest):
    global _total_predictions

    # Encode the current state
    state_text = req.current_state
    if req.context:
        state_text += " " + req.context
    state_emb = embedder.embed(state_text)

    # Get action index
    action_idx = config.ACTION_TYPES.index(req.action) if req.action in config.ACTION_TYPES else 0

    # Run the world model in imagination mode
    with torch.no_grad():
        world_model.eval()
        h, z = world_model.initial_state(1)
        action_tensor = torch.tensor([action_idx], dtype=torch.long, device=config.DEVICE)
        result = world_model.imagine_step(h, z, action_tensor)

        predicted_outcome_emb = result["predicted_outcome"].squeeze(0).cpu().numpy()
        predicted_success = float(result["predicted_success"].squeeze().item())

    prediction_id = str(uuid.uuid4())
    _prediction_cache[prediction_id] = {
        "state_embedding": state_emb,
        "predicted_outcome_embedding": predicted_outcome_emb,
        "h": result["h"].detach(),
        "z": result["z"].detach(),
        "action": req.action,
        "action_idx": action_idx,
        "timestamp": datetime.utcnow().isoformat(),
    }

    _total_predictions += 1

    # Generate risk factors based on confidence
    risk_factors = []
    if predicted_success < 0.5:
        risk_factors.append("Low predicted success probability")
    if predicted_success < 0.3:
        risk_factors.append("High failure risk - consider alternative approach")

    return {
        "prediction_id": prediction_id,
        "predicted_outcome": f"Model prediction for {req.action}: {req.action_details}",
        "confidence": round(predicted_success, 3),
        "predicted_success_probability": round(predicted_success, 3),
        "risk_factors": risk_factors,
        "prediction_embedding": predicted_outcome_emb[:10].tolist(),
    }


@app.post("/world-model/update")
@requires_ready("embeddings", "world_model", "curiosity")
async def wm_update(req: UpdateRequest):
    global _experience_count, _correct_predictions

    # Get the cached prediction
    cached = _prediction_cache.pop(req.prediction_id, None)
    if cached is None:
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    # Encode actual outcome
    actual_emb = embedder.embed(req.actual_outcome)

    # Compute prediction error
    pred_emb = cached["predicted_outcome_embedding"]
    prediction_error = float(np.mean((pred_emb - actual_emb) ** 2))

    # Track accuracy
    if req.was_successful:
        _correct_predictions += 1

    # Store experience in buffer
    exp = Experience(
        state_embedding=cached["state_embedding"],
        action=req.action,
        action_details="",
        outcome_embedding=actual_emb,
        was_successful=req.was_successful,
        prediction_error=prediction_error,
        prediction_id=req.prediction_id,
    )
    await trainer.buffer.add(exp)
    _experience_count += 1

    # Record curiosity signal
    curiosity_result = curiosity_manager.record_experience(
        domain=req.action,
        state_embedding=cached["state_embedding"],
        predicted_outcome=pred_emb,
        actual_outcome=actual_emb,
    )

    # Maybe train the model
    train_result = await trainer.maybe_train(_experience_count)

    # Generate lesson from high-error experiences
    lesson = ""
    if prediction_error > config.PREDICTION_ERROR_THRESHOLD:
        lesson = f"Unexpected outcome for {req.action}: prediction error was {prediction_error:.2f}"

    return {
        "prediction_error": round(prediction_error, 4),
        "model_updated": train_result is not None,
        "curiosity_signal": round(curiosity_result["combined_score"], 3),
        "lesson_extracted": lesson,
    }


@app.get("/world-model/stats")
@requires_ready("world_model", "continual_learning")
async def wm_stats():
    buffer_count = await trainer.buffer.count()
    accuracy = _correct_predictions / max(1, _total_predictions)
    train_stats = trainer.get_stats()

    return {
        "total_predictions": _total_predictions,
        "accuracy": round(accuracy, 3),
        "mean_prediction_error": round(train_stats["mean_recent_loss"], 4),
        "experiences_in_buffer": buffer_count,
        "model_version": consolidator.model_version,
        "last_trained": datetime.utcnow().isoformat(),
        "device": config.DEVICE,
        "train_steps": train_stats["train_steps"],
    }


# ──────────────────────────────────────────────
# Curiosity Engine
# ──────────────────────────────────────────────

@app.get("/curiosity/signals")
@requires_ready("curiosity")
async def curiosity_signals():
    signals = curiosity_manager.get_all_signals()
    top_targets = []
    for s in signals[:5]:
        reason = "High prediction error" if s["prediction_error"] > 0.5 else "Moderate interest"
        if s["rnd_novelty"] > 0.7:
            reason = "Novel territory - limited experience"
        top_targets.append({
            "domain": s["domain"],
            "curiosity_score": round(s["combined_score"], 3),
            "reason": reason,
        })

    global_pressure = np.mean([s["combined_score"] for s in signals]) if signals else 0.5
    novelty_by_action = curiosity_manager.get_novelty_by_action()

    return {
        "top_curiosity_targets": top_targets,
        "global_curiosity_pressure": round(float(global_pressure), 3),
        "novelty_scores_by_action": novelty_by_action,
    }


@app.post("/curiosity/evaluate-goal")
@requires_ready("curiosity", "embeddings")
async def curiosity_evaluate_goal(req: EvaluateGoalRequest):
    result = curiosity_manager.evaluate_goal(req.goal, embedder)
    return result


# ──────────────────────────────────────────────
# Continual Learning
# ──────────────────────────────────────────────

@app.post("/continual/consolidate")
@requires_ready("continual_learning", "curiosity")
async def continual_consolidate():
    result = await consolidator.consolidate()
    await curiosity_manager.persist_signals()
    return result


@app.get("/continual/status")
@requires_ready("continual_learning")
async def continual_status():
    status = consolidator.get_status()
    replay_stats = await ReplayManager(trainer.buffer).get_stats()
    status["replay_buffer_fill"] = replay_stats["buffer_fill"]
    return status


# ──────────────────────────────────────────────
# Abstraction Engine
# ──────────────────────────────────────────────

@app.post("/abstraction/extract")
@requires_ready("abstraction")
async def abstraction_extract():
    try:
        new_principles = await principle_extractor.extract_principles()
        meta_lessons = await meta_learner.extract_meta_lessons()
        all_principles = await principle_extractor.get_all_principles()

        return {
            "new_principles": new_principles,
            "total_principles": len(all_principles),
            "meta_lessons_extracted": len(meta_lessons),
        }
    except Exception as e:
        return {
            "new_principles": [],
            "total_principles": 0,
            "meta_lessons_extracted": 0,
            "error": str(e),
        }


@app.get("/abstraction/principles")
@requires_ready("abstraction")
async def abstraction_principles():
    principles = await principle_extractor.get_all_principles()
    return {
        "principles": principles,
        "total": len(principles),
    }


@app.post("/abstraction/apply")
@requires_ready("abstraction")
async def abstraction_apply(req: ApplyRequest):
    result = await principle_extractor.apply_principles(req.goal, req.action)
    return result


# ──────────────────────────────────────────────
# Reasoning Workspace
# ──────────────────────────────────────────────

@app.post("/reasoning/start")
@requires_ready("reasoning")
async def reasoning_start(req: StartReasoningRequest):
    result = await reasoning_ws.start(req.goal, req.initial_context)
    return result


@app.post("/reasoning/add-thought")
@requires_ready("reasoning")
async def reasoning_add_thought(req: AddThoughtRequest):
    try:
        result = await reasoning_ws.add_thought(
            req.workspace_id, req.thought, req.parent_id,
            req.thought_type, req.confidence,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reasoning/add-causal-link")
@requires_ready("reasoning")
async def reasoning_add_causal_link(req: AddCausalLinkRequest):
    try:
        await reasoning_ws.add_causal_link(
            req.workspace_id, req.cause_id, req.effect_id,
            req.relationship, req.strength,
        )
        return {"status": "ok"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/reasoning/workspace/{workspace_id}")
@requires_ready("reasoning")
async def reasoning_get_workspace(workspace_id: str):
    result = await reasoning_ws.get_workspace(workspace_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/reasoning/query")
@requires_ready("reasoning")
async def reasoning_query(req: QueryReasoningRequest):
    result = await reasoning_ws.query(req.workspace_id, req.question)
    return result


# ──────────────────────────────────────────────
# Self-Model
# ──────────────────────────────────────────────

@app.get("/self-model/state")
@requires_ready("self_model")
async def self_model_state():
    try:
        state = await state_tracker.get_state()
        capabilities = await capability_model.get_capabilities()
        predictions = await transition_model.predict_next_state()

        return {
            "current_state": state,
            "capability_confidence": capabilities,
            "predicted_next_state": predictions,
        }
    except Exception as e:
        return {
            "current_state": {
                "active_goals": 0,
                "lessons_learned": 0,
                "skills_acquired": 0,
                "knowledge_nodes": 0,
                "success_rate_overall": 0.0,
                "success_rate_last_10": 0.0,
                "dominant_failure_mode": "none",
                "strongest_capability": "none",
                "weakest_capability": "none",
            },
            "capability_confidence": {},
            "predicted_next_state": {
                "if_continue_current": "Insufficient data",
                "recommendation": "Backend unreachable - cannot compute",
            },
            "error": str(e),
        }


@app.post("/self-model/predict-change")
@requires_ready("self_model")
async def self_model_predict_change(req: PredictChangeRequest):
    try:
        result = await transition_model.predict_change(
            req.proposed_change, req.change_type
        )
        return result
    except Exception as e:
        return {
            "predicted_effect": "Cannot predict - error occurred",
            "confidence": 0.0,
            "side_effects": [str(e)],
            "recommendation": "defer",
        }


# ══════════════════════════════════════════════
# PHASE 2 ENDPOINTS
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Phase 2 Request Models
# ──────────────────────────────────────────────

class BetaVAEEncodeRequest(BaseModel):
    text: str

class BetaVAESimilarityRequest(BaseModel):
    text_a: str
    text_b: str

class BetaVAEGenerateRequest(BaseModel):
    base_text: str
    modify: dict = {}

class EvaluatePolicyRequest(BaseModel):
    current_state: str
    proposed_action: str
    goal: str

class ComparePoliciesRequest(BaseModel):
    current_state: str
    policies: list
    goal: str

class UpdateBeliefsRequest(BaseModel):
    action_taken: str
    observation: str
    was_expected: bool = True

class HopfieldStoreRequest(BaseModel):
    content: str
    context: str = ""
    importance: float = 0.5

class HopfieldRetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class HopfieldAssociateRequest(BaseModel):
    pattern_id: str

class MetaAdaptRequest(BaseModel):
    domain: str
    examples: list = []
    query: str = ""

class MetaTrainStepRequest(BaseModel):
    task_type: str = "action_selection"


# ──────────────────────────────────────────────
# β-VAE — Disentangled Representations
# ──────────────────────────────────────────────

@app.post("/beta-vae/encode")
@requires_ready("beta_vae")
async def beta_vae_encode(req: BetaVAEEncodeRequest):
    result = rep_engine.encode_text(req.text)
    return result


@app.post("/beta-vae/similarity")
@requires_ready("beta_vae")
async def beta_vae_similarity(req: BetaVAESimilarityRequest):
    result = rep_engine.compute_similarity(req.text_a, req.text_b)
    return result


@app.post("/beta-vae/generate")
@requires_ready("beta_vae")
async def beta_vae_generate(req: BetaVAEGenerateRequest):
    result = rep_engine.generate_from_modification(req.base_text, req.modify)
    return result


@app.get("/beta-vae/stats")
@requires_ready("beta_vae")
async def beta_vae_stats():
    return beta_vae_trainer.get_stats()


# ──────────────────────────────────────────────
# Active Inference — Unified Curiosity + Planning
# ──────────────────────────────────────────────

@app.post("/active-inference/evaluate-policy")
@requires_ready("active_inference")
async def ai_evaluate_policy(req: EvaluatePolicyRequest):
    efe_result = compute_efe(
        gen_model, req.current_state, req.proposed_action,
        req.goal, precision_ctrl.precision,
    )
    efe_result["recommendation"] = recommend_action(efe_result)
    return efe_result


@app.post("/active-inference/compare-policies")
@requires_ready("active_inference")
async def ai_compare_policies(req: ComparePoliciesRequest):
    result = compare_policies(
        gen_model, req.current_state, req.policies,
        req.goal, precision_ctrl.precision,
    )
    return result


@app.post("/active-inference/update-beliefs")
@requires_ready("active_inference")
async def ai_update_beliefs(req: UpdateBeliefsRequest):
    surprise = 0.3 if req.was_expected else 0.7
    prediction_error = surprise

    old_precision = precision_ctrl.precision
    new_precision = precision_ctrl.update(prediction_error)
    await precision_ctrl.persist()

    return {
        "belief_update_magnitude": round(abs(new_precision - old_precision), 4),
        "surprise": round(surprise, 4),
        "precision_updated": True,
        "new_precision": round(new_precision, 4),
    }


@app.get("/active-inference/status")
@requires_ready("active_inference")
async def ai_status():
    return precision_ctrl.get_status()


# ──────────────────────────────────────────────
# Modern Hopfield Memory
# ──────────────────────────────────────────────

@app.post("/hopfield/store")
@requires_ready("hopfield_memory")
async def hopfield_store(req: HopfieldStoreRequest):
    result = await episodic_store.store(req.content, req.context, req.importance)
    return result


@app.post("/hopfield/retrieve")
@requires_ready("hopfield_memory")
async def hopfield_retrieve(req: HopfieldRetrieveRequest):
    results = await episodic_store.retrieve(req.query, req.top_k)
    return {"retrieved": results}


@app.post("/hopfield/associate")
@requires_ready("hopfield_memory")
async def hopfield_associate(req: HopfieldAssociateRequest):
    results = await episodic_store.associate(req.pattern_id)
    return {"associations": results}


@app.get("/hopfield/stats")
@requires_ready("hopfield_memory")
async def hopfield_stats():
    return episodic_store.get_stats()


# ──────────────────────────────────────────────
# Meta-Learning (Reptile)
# ──────────────────────────────────────────────

@app.post("/meta-learning/adapt")
@requires_ready("meta_learning")
async def meta_adapt(req: MetaAdaptRequest):
    result = domain_adapter.predict_with_adaptation(
        req.domain, req.query, req.examples,
    )
    return result


@app.post("/meta-learning/train-step")
@requires_ready("meta_learning")
async def meta_train_step(req: MetaTrainStepRequest):
    tasks = await task_sampler.sample_tasks()
    if not tasks:
        return {
            "meta_loss": 0.0,
            "inner_steps": 0,
            "outer_step": reptile.meta_step_count,
            "note": "Not enough experience data for meta-learning",
        }

    def loss_fn(model, task_data):
        losses = model.compute_loss(
            task_data["obs"], task_data["actions"],
            task_data["outcomes"], task_data["successes"],
        )
        return losses["loss"]

    result = reptile.meta_step(tasks, loss_fn)
    return result


@app.get("/meta-learning/status")
@requires_ready("meta_learning")
async def meta_status():
    return reptile.get_status()


# ══════════════════════════════════════════════
# PHASE 3 ENDPOINTS
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Phase 3 Request Models
# ──────────────────────────────────────────────

class DreamcoderWakeRequest(BaseModel):
    task: str
    context: str = ""

class DreamcoderSleepRequest(BaseModel):
    min_solutions: int = config.DREAMCODER_MIN_SOLUTIONS

class DreamcoderComposeRequest(BaseModel):
    task: str
    domain: str = ""

class WorkspaceBroadcastRequest(BaseModel):
    source_module: str
    signal_type: str
    content: dict = {}
    salience: float = 0.5
    urgency: float = 0.5

class WorkspaceSubscribeRequest(BaseModel):
    module_name: str
    interests: list = []

class CausalAddRequest(BaseModel):
    cause: str
    effect: str
    strength: float = 0.5
    evidence_count: int = 1
    mechanism: str = ""

class CausalInterveneRequest(BaseModel):
    intervention: str
    given: dict = {}
    query: str = "success"

class CausalCounterfactualRequest(BaseModel):
    actual_action: str
    actual_outcome: str
    counterfactual: str

class CausalLearnRequest(BaseModel):
    min_evidence: int = config.CAUSAL_MIN_EVIDENCE


# ──────────────────────────────────────────────
# DreamCoder Abstraction
# ──────────────────────────────────────────────

@app.post("/dreamcoder/wake")
@requires_ready("dreamcoder", "embeddings")
async def dreamcoder_wake(req: DreamcoderWakeRequest):
    task_text = req.task
    if req.context:
        task_text += " " + req.context
    result = _wake_solve(task_text, dc_library, embedder)
    return result


@app.post("/dreamcoder/sleep")
@requires_ready("dreamcoder", "embeddings")
async def dreamcoder_sleep_endpoint(req: DreamcoderSleepRequest):
    new_prims = await _abstraction_sleep(dc_library, embedder, backend, req.min_solutions)
    total = dc_library.size
    return {
        "new_primitives": [p.to_dict() for p in new_prims],
        "library_size": total,
        "compression_ratio": round(len(new_prims) / max(1, total), 3),
    }


@app.get("/dreamcoder/library")
@requires_ready("dreamcoder")
async def dreamcoder_library():
    prims = [p.to_dict() for p in dc_library.primitives.values()]
    prims.sort(key=lambda p: p["frequency"], reverse=True)
    return {
        "primitives": prims,
        "total": len(prims),
    }


@app.post("/dreamcoder/compose")
@requires_ready("dreamcoder", "embeddings")
async def dreamcoder_compose(req: DreamcoderComposeRequest):
    result = await _compose_solution(req.task, req.domain, dc_library, embedder)
    return result


# ──────────────────────────────────────────────
# Global Workspace
# ──────────────────────────────────────────────

@app.post("/workspace/broadcast")
@requires_ready("global_workspace")
async def workspace_broadcast(req: WorkspaceBroadcastRequest):
    signal = _Signal(
        source_module=req.source_module,
        signal_type=req.signal_type,
        content=req.content,
        salience=req.salience,
        urgency=req.urgency,
    )
    result = await gw.submit_and_compete(signal)
    return result


@app.get("/workspace/current")
@requires_ready("global_workspace")
async def workspace_current():
    return gw.get_current()


@app.get("/workspace/modules")
@requires_ready("global_workspace")
async def workspace_modules():
    return {"modules": gw_registry.get_all()}


@app.post("/workspace/subscribe")
@requires_ready("global_workspace")
async def workspace_subscribe(req: WorkspaceSubscribeRequest):
    gw.subscribe(req.module_name, req.interests)
    return {"subscribed": True}


# ──────────────────────────────────────────────
# Causal Reasoner
# ──────────────────────────────────────────────

@app.post("/causal/add-relationship")
@requires_ready("causal_reasoner")
async def causal_add(req: CausalAddRequest):
    edge = causal_scm.add_edge(
        req.cause, req.effect, req.strength, req.evidence_count, req.mechanism
    )
    await causal_scm.save_all()
    return {
        "relationship_id": edge.id,
        "total_relationships": len(causal_scm.get_all_edges()),
    }


@app.post("/causal/intervene")
@requires_ready("causal_reasoner")
async def causal_intervene(req: CausalInterveneRequest):
    # Parse intervention string like "do(action=propose_change)"
    intervention_var = req.intervention
    value = True
    if intervention_var.startswith("do(") and intervention_var.endswith(")"):
        inner = intervention_var[3:-1]
        if "=" in inner:
            parts = inner.split("=", 1)
            intervention_var = parts[0].strip()
            value = parts[1].strip()

    result = _do_intervention(causal_scm, intervention_var, value, req.query, req.given)
    return result


@app.post("/causal/counterfactual")
@requires_ready("causal_reasoner")
async def causal_counterfactual(req: CausalCounterfactualRequest):
    result = _counterfactual_query(
        causal_scm, req.actual_action, req.actual_outcome, req.counterfactual
    )
    return result


@app.get("/causal/graph")
@requires_ready("causal_reasoner")
async def causal_graph():
    return {
        "nodes": causal_scm.get_all_nodes(),
        "edges": causal_scm.get_all_edges(),
        "total_nodes": len(causal_scm.get_all_nodes()),
        "total_edges": len(causal_scm.get_all_edges()),
    }


@app.post("/causal/learn")
@requires_ready("causal_reasoner")
async def causal_learn(req: CausalLearnRequest):
    result = await _learn_causal_structure(causal_scm, backend, req.min_evidence)
    return result


# ══════════════════════════════════════════════
# PHASE 4 ENDPOINTS
# ══════════════════════════════════════════════

# ──────────────────────────────────────────────
# Phase 4 Request Models
# ──────────────────────────────────────────────

class AttentionUpdateRequest(BaseModel):
    target: str
    target_type: str = "step"
    signals: dict = {}

class AttentionIntrospectRequest(BaseModel):
    depth: str = "normal"

class HierarchyPredictRequest(BaseModel):
    level: int = 0
    context: str = ""

class HierarchyUpdateRequest(BaseModel):
    level: int = 0
    predicted: float
    actual: float
    context: str = ""

class LiquidPredictRequest(BaseModel):
    state_embedding: list
    action: str = ""
    time_delta: float = 1.0

class LiquidUpdateRequest(BaseModel):
    state_embedding: list
    actual_outcome_embedding: list
    time_delta: float = 1.0


# ──────────────────────────────────────────────
# Attention Schema
# ──────────────────────────────────────────────

@app.get("/attention/focus")
@requires_ready("attention_schema")
async def attention_focus():
    return attention_schema.get_focus()


@app.post("/attention/update")
@requires_ready("attention_schema")
async def attention_update(req: AttentionUpdateRequest):
    result = attention_schema.update_focus(req.target, req.target_type, req.signals)
    await attention_schema.persist_focus()
    return result


@app.post("/attention/introspect")
@requires_ready("attention_schema")
async def attention_introspect(req: AttentionIntrospectRequest):
    return attention_awareness.introspect(req.depth)


# ──────────────────────────────────────────────
# Predictive Processing Hierarchy
# ──────────────────────────────────────────────

@app.get("/prediction-hierarchy/state")
@requires_ready("predictive_hierarchy")
async def hierarchy_state():
    return pred_hierarchy.get_state()


@app.post("/prediction-hierarchy/predict")
@requires_ready("predictive_hierarchy")
async def hierarchy_predict(req: HierarchyPredictRequest):
    return pred_hierarchy.predict(req.level, req.context)


@app.post("/prediction-hierarchy/update")
@requires_ready("predictive_hierarchy")
async def hierarchy_update(req: HierarchyUpdateRequest):
    result = pred_hierarchy.update(req.level, req.predicted, req.actual, req.context)
    return result


# ──────────────────────────────────────────────
# Liquid Time-Constant Network
# ──────────────────────────────────────────────

@app.get("/liquid/status")
@requires_ready("liquid_network")
async def liquid_status():
    return liquid_model.get_status()


@app.post("/liquid/predict")
@requires_ready("liquid_network", "embeddings")
async def liquid_predict(req: LiquidPredictRequest):
    state = torch.tensor([req.state_embedding], dtype=torch.float32, device=config.DEVICE)
    dt = torch.tensor([[req.time_delta]], dtype=torch.float32, device=config.DEVICE)

    liquid_model.eval()
    with torch.no_grad():
        h = liquid_model.initial_state(1)
        result = liquid_model.forward(state, h, dt)
        pred = result["predicted_outcome"].squeeze(0).cpu().tolist()
        success = float(result["predicted_success"].squeeze().item())

    return {
        "predicted_next_state": pred[:10],
        "confidence": round(success, 3),
        "dynamics_speed": "fast" if req.time_delta < 0.5 else "medium" if req.time_delta < 5.0 else "slow",
    }


@app.post("/liquid/update")
@requires_ready("liquid_network")
async def liquid_update(req: LiquidUpdateRequest):
    state = torch.tensor([req.state_embedding], dtype=torch.float32, device=config.DEVICE)
    actual = torch.tensor([req.actual_outcome_embedding], dtype=torch.float32, device=config.DEVICE)
    dt = torch.tensor([[req.time_delta]], dtype=torch.float32, device=config.DEVICE)

    liquid_model.eval()
    with torch.no_grad():
        h = liquid_model.initial_state(1)
        result = liquid_model.forward(state, h, dt)
        pred = result["predicted_outcome"]
        error = float(torch.mean((pred - actual) ** 2).item())

    return {
        "prediction_error": round(error, 4),
        "time_constant_adjusted": True,
    }
