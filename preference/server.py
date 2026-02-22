from __future__ import annotations

import json
import os
import random
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .clip_embedder import CLIPEmbedder
from .features import extract_features, get_feature_names
from .model import PreferenceModel

DATA_PATH = Path(__file__).resolve().parent / "preference_data.json"
STATIC_DIR = Path(__file__).resolve().parent.parent / "public"
data_lock = Lock()

clip_embedder = CLIPEmbedder()
FEATURE_DIM = len(get_feature_names())

app = FastAPI(title="Armoria Preference API")

# CORS origins are configurable via the PREFERENCE_CORS_ORIGINS environment variable.
# Provide a comma-separated list of allowed origins.
# Default: http://localhost:5000
_cors_origins_env = os.environ.get("PREFERENCE_CORS_ORIGINS", "http://localhost:5000")
_cors_origins = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GridSelection(BaseModel):
    coa: Dict[str, Any]
    selected: bool
    image: Optional[str] = None


class GridPreferenceRequest(BaseModel):
    selections: List[GridSelection]


class PairwisePreferenceRequest(BaseModel):
    winner: Dict[str, Any]
    loser: Dict[str, Any]
    winner_image: Optional[str] = None
    loser_image: Optional[str] = None


class ScoreRequest(BaseModel):
    coas: List[Dict[str, Any]]
    images: Optional[List[str]] = None


class SuggestPairRequest(BaseModel):
    candidates: List[Dict[str, Any]]


def load_preferences() -> List[Dict[str, Any]]:
    if not DATA_PATH.exists():
        return []
    try:
        with DATA_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                return data.get("preferences", [])
            return []
    except (OSError, json.JSONDecodeError):
        return []


def save_preferences(preferences: List[Dict[str, Any]]) -> None:
    with data_lock:
        with DATA_PATH.open("w", encoding="utf-8") as handle:
            json.dump({"preferences": preferences}, handle, ensure_ascii=False, indent=2)


def _is_valid_image(image: Optional[str]) -> bool:
    return bool(image and image.strip())


def _get_embedding_length(value: Any) -> Optional[int]:
    if isinstance(value, np.ndarray):
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _pref_has_embedding_dim(pref: Dict[str, Any], dim: int) -> bool:
    pref_type = pref.get("type")
    if pref_type == "grid":
        return _get_embedding_length(pref.get("embedding")) == dim
    if pref_type == "pairwise":
        return (
            _get_embedding_length(pref.get("winner_embedding")) == dim
            or _get_embedding_length(pref.get("loser_embedding")) == dim
        )
    return False


def _select_embedding_dim(preferences: List[Dict[str, Any]]) -> int:
    clip_dim = clip_embedder.embedding_dim
    if any(_pref_has_embedding_dim(pref, clip_dim) for pref in preferences):
        return clip_dim
    if any(_pref_has_embedding_dim(pref, FEATURE_DIM) for pref in preferences):
        return FEATURE_DIM
    return clip_dim


def _ensure_feature_embeddings(preferences: List[Dict[str, Any]]) -> bool:
    updated = False
    for pref in preferences:
        pref_type = pref.get("type")
        if pref_type == "grid":
            if pref.get("embedding") is None and pref.get("coa"):
                pref["embedding"] = extract_features(pref["coa"])
                pref["embedding_type"] = "features"
                updated = True
        elif pref_type == "pairwise":
            if pref.get("winner_embedding") is None and pref.get("winner"):
                pref["winner_embedding"] = extract_features(pref["winner"])
                pref["winner_embedding_type"] = "features"
                updated = True
            if pref.get("loser_embedding") is None and pref.get("loser"):
                pref["loser_embedding"] = extract_features(pref["loser"])
                pref["loser_embedding_type"] = "features"
                updated = True
    return updated


def _embed_images_safe(images: List[str]) -> Optional[np.ndarray]:
    if not images:
        return None
    try:
        return clip_embedder.embed_base64_batch(images)
    except Exception as exc:  # pragma: no cover - best effort fallback
        print(f"CLIP embedding failed: {exc}")
        return None


preferences: List[Dict[str, Any]] = load_preferences()
model = PreferenceModel(embedding_dim=clip_embedder.embedding_dim)
model_stats: Dict[str, Any] = {"accuracy": 0.0, "n_samples": 0}


@app.post("/api/preferences/grid")
def add_grid_preferences(payload: GridPreferenceRequest) -> Dict[str, Any]:
    selections = payload.selections
    image_indices = [
        index
        for index, selection in enumerate(selections)
        if _is_valid_image(selection.image)
    ]
    clip_embeddings = None
    if image_indices:
        clip_images = [selections[index].image for index in image_indices]
        clip_embeddings = _embed_images_safe(clip_images)

    clip_map: Dict[int, List[float]] = {}
    if clip_embeddings is not None:
        for index, embedding in zip(image_indices, clip_embeddings):
            clip_map[index] = embedding.tolist()

    new_items = []
    for index, selection in enumerate(selections):
        if index in clip_map:
            embedding = clip_map[index]
            embedding_type = "clip"
        else:
            embedding = extract_features(selection.coa)
            embedding_type = "features"
        new_items.append(
            {
                "type": "grid",
                "coa": selection.coa,
                "selected": selection.selected,
                "embedding": embedding,
                "embedding_type": embedding_type,
            }
        )

    preferences.extend(new_items)
    save_preferences(preferences)
    return {"saved": len(new_items), "total": len(preferences)}


@app.post("/api/preferences/pairwise")
def add_pairwise_preferences(payload: PairwisePreferenceRequest) -> Dict[str, Any]:
    use_clip = _is_valid_image(payload.winner_image) and _is_valid_image(
        payload.loser_image
    )
    winner_embedding: List[float]
    loser_embedding: List[float]
    embedding_type = "features"

    if use_clip:
        clip_embeddings = _embed_images_safe(
            [payload.winner_image, payload.loser_image]
        )
        if clip_embeddings is not None:
            winner_embedding = clip_embeddings[0].tolist()
            loser_embedding = clip_embeddings[1].tolist()
            embedding_type = "clip"
        else:
            winner_embedding = extract_features(payload.winner)
            loser_embedding = extract_features(payload.loser)
    else:
        winner_embedding = extract_features(payload.winner)
        loser_embedding = extract_features(payload.loser)

    preferences.append(
        {
            "type": "pairwise",
            "winner": payload.winner,
            "loser": payload.loser,
            "winner_embedding": winner_embedding,
            "loser_embedding": loser_embedding,
            "embedding_type": embedding_type,
        }
    )
    save_preferences(preferences)
    return {"saved": 1, "total": len(preferences)}


@app.post("/api/train")
def train_model() -> Dict[str, Any]:
    global model, model_stats
    embedding_dim = _select_embedding_dim(preferences)
    if model.embedding_dim != embedding_dim:
        model = PreferenceModel(embedding_dim=embedding_dim)

    if embedding_dim == FEATURE_DIM:
        if _ensure_feature_embeddings(preferences):
            save_preferences(preferences)

    model_stats = model.fit(preferences)
    return {"trained": model.is_trained, **model_stats}


@app.post("/api/score")
def score_coas(payload: ScoreRequest) -> Dict[str, Any]:
    images = payload.images
    if images is not None and len(images) != len(payload.coas):
        raise HTTPException(
            status_code=400, detail="Images length must match COAs length"
        )

    use_images = images is not None and all(_is_valid_image(img) for img in images)
    embeddings: List[List[float]] | np.ndarray

    if use_images:
        clip_embeddings = _embed_images_safe(images)
        if clip_embeddings is not None:
            embeddings = clip_embeddings
        else:
            embeddings = [extract_features(coa) for coa in payload.coas]
    else:
        embeddings = [extract_features(coa) for coa in payload.coas]

    scores = model.score_batch(embeddings)
    return {"scores": scores, "trained": model.is_trained}


@app.post("/api/suggest-pair")
def suggest_pair(payload: SuggestPairRequest) -> Dict[str, Any]:
    candidates = payload.candidates
    if len(candidates) < 2:
        raise HTTPException(status_code=400, detail="At least two candidates required")

    features = [np.array(extract_features(c), dtype=float) for c in candidates]

    if model.is_trained:
        # Minimum L1 feature distance to consider a pair "distinct enough"
        MIN_DISTANCE = 3.0

        best_pair = None
        best_score = -1.0
        best_uncertainty = 0.5

        # Track the most distant pair as a fallback
        max_distance_pair = None
        max_distance = -1.0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                distance = float(np.sum(np.abs(features[i] - features[j])))

                # Track most distant pair for fallback
                if distance > max_distance:
                    max_distance = distance
                    max_distance_pair = (candidates[i], candidates[j])

                # Skip pairs that are too similar
                if distance < MIN_DISTANCE:
                    continue

                uncertainty = model.get_uncertainty(features[i], features[j])
                # Combined score: uncertainty weighted by diversity
                combined = uncertainty * (1.0 + distance)
                if combined > best_score:
                    best_score = combined
                    best_uncertainty = uncertainty
                    best_pair = (candidates[i], candidates[j])

        # Fall back to most distant pair if no pair met the threshold
        if best_pair is None:
            if max_distance_pair is not None:
                best_pair = max_distance_pair
                best_uncertainty = model.get_uncertainty(
                    features[candidates.index(max_distance_pair[0])],
                    features[candidates.index(max_distance_pair[1])],
                )
            else:
                best_pair = tuple(random.sample(candidates, 2))
                best_uncertainty = 0.5

        pair = list(best_pair)
    else:
        pair = random.sample(candidates, 2)
        best_uncertainty = 0.5

    return {
        "pair": pair,
        "uncertainty": best_uncertainty,
        "trained": model.is_trained,
    }


@app.get("/api/stats")
def get_stats() -> Dict[str, Any]:
    grid_count = sum(1 for pref in preferences if pref.get("type") == "grid")
    pairwise_count = sum(1 for pref in preferences if pref.get("type") == "pairwise")
    selected_count = sum(
        1
        for pref in preferences
        if pref.get("type") == "grid" and pref.get("selected")
    )

    return {
        "total_preferences": len(preferences),
        "grid_preferences": grid_count,
        "pairwise_preferences": pairwise_count,
        "grid_selected": selected_count,
        "model_trained": model.is_trained,
        "model_accuracy": model_stats.get("accuracy", 0.0),
        "training_samples": model_stats.get("n_samples", 0),
    }


@app.post("/api/reset")
def reset_preferences() -> Dict[str, Any]:
    global model, model_stats
    preferences.clear()
    save_preferences(preferences)
    model = PreferenceModel(embedding_dim=clip_embedder.embedding_dim)
    model_stats = {"accuracy": 0.0, "n_samples": 0}
    return {"status": "reset", "total": len(preferences)}


@app.get("/api/favorites")
def get_favorites() -> Dict[str, Any]:
    """Return COAs the user liked, grouped by source."""
    grid_selections: List[Dict[str, Any]] = []
    pairwise_winners: List[Dict[str, Any]] = []

    for idx, pref in enumerate(preferences):
        pref_type = pref.get("type")
        if pref_type == "grid" and pref.get("selected"):
            grid_selections.append({"index": idx, "coa": pref["coa"]})
        elif pref_type == "pairwise":
            winner = pref.get("winner")
            if winner:
                pairwise_winners.append({"index": idx, "coa": winner})

    return {
        "grid_selections": grid_selections,
        "pairwise_winners": pairwise_winners,
        "total": len(grid_selections) + len(pairwise_winners),
    }


# Mount static files AFTER API routes so /api/* takes priority.
# This allows the FastAPI server to serve both the API and the Armoria
# frontend from the same origin, eliminating cross-origin issues.
if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("PREFERENCE_HOST", "0.0.0.0")
    port = int(os.environ.get("PREFERENCE_PORT", "8080"))
    uvicorn.run("preference.server:app", host=host, port=port, reload=True)