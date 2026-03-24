from __future__ import annotations
"""AXIOM Cognitive Core v2 — FastAPI server with all endpoints."""

import uuid
import json
import traceback
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
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


# ──────────────────────────────────────────────
# Global components (initialized on startup)
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

# Prediction cache: prediction_id -> {embedding, h, z, action}
_prediction_cache: dict[str, dict] = {}
_experience_count = 0
_total_predictions = 0
_correct_predictions = 0
_model_version = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, cleanup on shutdown."""
    global embedder, backend, world_model, trainer, curiosity_manager
    global ewc, consolidator, principle_extractor, meta_learner, skill_composer
    global reasoning_ws, state_tracker, capability_model, transition_model
    global beta_vae_model, beta_vae_trainer, rep_engine, gen_model, precision_ctrl
    global episodic_store, memory_mgr, reptile, task_sampler, domain_adapter

    # Initialize database
    await get_db()

    # Initialize embedding service
    embedder = EmbeddingService.get_instance()

    # Initialize backend client
    backend = BackendClient()

    # Initialize world model
    world_model = RSSMWorldModel().to(config.DEVICE)
    trainer = WorldModelTrainer(world_model)

    # Initialize curiosity
    curiosity_manager = CuriosityManager()

    # Initialize continual learning
    ewc = EWC(world_model)
    consolidator = Consolidator(world_model, trainer, ewc)

    # Try to load latest EWC checkpoint
    await ewc.load_latest_checkpoint()
    if ewc.is_initialized:
        trainer.fisher_diag = ewc.fisher_diag
        trainer.anchor_params = ewc.anchor_params

    # Initialize abstraction
    principle_extractor = PrincipleExtractor(embedder, backend)
    meta_learner = MetaLearner(embedder, backend)
    skill_composer = SkillComposer(embedder, backend)

    # Initialize reasoning
    reasoning_ws = ReasoningWorkspace(embedder)

    # Initialize self-model
    state_tracker = StateTracker(backend)
    capability_model = CapabilityModel(backend)
    transition_model = TransitionModel(state_tracker, capability_model, embedder, backend)

    # Phase 2 initialization
    # β-VAE
    input_dim = embedder.dim
    beta_vae_model = BetaVAE(input_dim=input_dim).to(config.DEVICE)
    beta_vae_trainer = BetaVAETrainer(beta_vae_model)
    rep_engine = RepresentationEngine(beta_vae_model, embedder)

    # Active Inference
    gen_model = GenerativeModel(world_model, embedder)
    precision_ctrl = PrecisionController()
    await precision_ctrl.load_latest()

    # Hopfield Episodic Memory
    episodic_store = EpisodicStore(embedder)
    await episodic_store.load_from_db()
    memory_mgr = MemoryManager(episodic_store)

    # Meta-Learning (Reptile)
    reptile = Reptile(world_model)
    task_sampler = TaskSampler()
    domain_adapter = DomainAdapter(reptile, embedder)

    yield

    # Cleanup
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
# Health
# ──────────────────────────────────────────────

@app.get("/health")
async def health():
    components = {
        "embeddings": embedder is not None,
        "world_model": world_model is not None,
        "curiosity": curiosity_manager is not None,
        "continual_learning": ewc is not None,
        "abstraction": principle_extractor is not None,
        "reasoning": reasoning_ws is not None,
        "self_model": state_tracker is not None,
        "beta_vae": beta_vae_model is not None,
        "active_inference": gen_model is not None,
        "hopfield_memory": episodic_store is not None,
        "meta_learning": reptile is not None,
    }
    return {
        "status": "alive",
        "service": "AXIOM Cognitive Core v2",
        "device": config.DEVICE,
        "components": components,
    }


# ──────────────────────────────────────────────
# World Model
# ──────────────────────────────────────────────

@app.post("/world-model/predict")
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
        "prediction_embedding": predicted_outcome_emb[:10].tolist(),  # First 10 dims for reference
    }


@app.post("/world-model/update")
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
async def curiosity_evaluate_goal(req: EvaluateGoalRequest):
    result = curiosity_manager.evaluate_goal(req.goal, embedder)
    return result


# ──────────────────────────────────────────────
# Continual Learning
# ──────────────────────────────────────────────

@app.post("/continual/consolidate")
async def continual_consolidate():
    result = await consolidator.consolidate()
    # Also persist curiosity signals
    await curiosity_manager.persist_signals()
    return result


@app.get("/continual/status")
async def continual_status():
    status = consolidator.get_status()
    replay_stats = await ReplayManager(trainer.buffer).get_stats()
    status["replay_buffer_fill"] = replay_stats["buffer_fill"]
    return status


# ──────────────────────────────────────────────
# Abstraction Engine
# ──────────────────────────────────────────────

@app.post("/abstraction/extract")
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
async def abstraction_principles():
    principles = await principle_extractor.get_all_principles()
    return {
        "principles": principles,
        "total": len(principles),
    }


@app.post("/abstraction/apply")
async def abstraction_apply(req: ApplyRequest):
    result = await principle_extractor.apply_principles(req.goal, req.action)
    return result


# ──────────────────────────────────────────────
# Reasoning Workspace
# ──────────────────────────────────────────────

@app.post("/reasoning/start")
async def reasoning_start(req: StartReasoningRequest):
    result = await reasoning_ws.start(req.goal, req.initial_context)
    return result


@app.post("/reasoning/add-thought")
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
async def reasoning_get_workspace(workspace_id: str):
    result = await reasoning_ws.get_workspace(workspace_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.post("/reasoning/query")
async def reasoning_query(req: QueryReasoningRequest):
    result = await reasoning_ws.query(req.workspace_id, req.question)
    return result


# ──────────────────────────────────────────────
# Self-Model
# ──────────────────────────────────────────────

@app.get("/self-model/state")
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
        # Return defaults if backend is unreachable
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
async def beta_vae_encode(req: BetaVAEEncodeRequest):
    result = rep_engine.encode_text(req.text)
    return result


@app.post("/beta-vae/similarity")
async def beta_vae_similarity(req: BetaVAESimilarityRequest):
    result = rep_engine.compute_similarity(req.text_a, req.text_b)
    return result


@app.post("/beta-vae/generate")
async def beta_vae_generate(req: BetaVAEGenerateRequest):
    result = rep_engine.generate_from_modification(req.base_text, req.modify)
    return result


@app.get("/beta-vae/stats")
async def beta_vae_stats():
    return beta_vae_trainer.get_stats()


# ──────────────────────────────────────────────
# Active Inference — Unified Curiosity + Planning
# ──────────────────────────────────────────────

@app.post("/active-inference/evaluate-policy")
async def ai_evaluate_policy(req: EvaluatePolicyRequest):
    efe_result = compute_efe(
        gen_model, req.current_state, req.proposed_action,
        req.goal, precision_ctrl.precision,
    )
    efe_result["recommendation"] = recommend_action(efe_result)
    return efe_result


@app.post("/active-inference/compare-policies")
async def ai_compare_policies(req: ComparePoliciesRequest):
    result = compare_policies(
        gen_model, req.current_state, req.policies,
        req.goal, precision_ctrl.precision,
    )
    return result


@app.post("/active-inference/update-beliefs")
async def ai_update_beliefs(req: UpdateBeliefsRequest):
    # Compute surprise based on whether outcome was expected
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
async def ai_status():
    return precision_ctrl.get_status()


# ──────────────────────────────────────────────
# Modern Hopfield Memory
# ──────────────────────────────────────────────

@app.post("/hopfield/store")
async def hopfield_store(req: HopfieldStoreRequest):
    result = await episodic_store.store(req.content, req.context, req.importance)
    return result


@app.post("/hopfield/retrieve")
async def hopfield_retrieve(req: HopfieldRetrieveRequest):
    results = await episodic_store.retrieve(req.query, req.top_k)
    return {"retrieved": results}


@app.post("/hopfield/associate")
async def hopfield_associate(req: HopfieldAssociateRequest):
    results = await episodic_store.associate(req.pattern_id)
    return {"associations": results}


@app.get("/hopfield/stats")
async def hopfield_stats():
    return episodic_store.get_stats()


# ──────────────────────────────────────────────
# Meta-Learning (Reptile)
# ──────────────────────────────────────────────

@app.post("/meta-learning/adapt")
async def meta_adapt(req: MetaAdaptRequest):
    result = domain_adapter.predict_with_adaptation(
        req.domain, req.query, req.examples,
    )
    return result


@app.post("/meta-learning/train-step")
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
async def meta_status():
    return reptile.get_status()
