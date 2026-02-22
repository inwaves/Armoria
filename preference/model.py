from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from joblib import dump, load
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

EmbeddingLike = Union[List[float], np.ndarray]


class PreferenceModel:
    """Pairwise preference model based on a gradient boosted classifier."""

    def __init__(self, embedding_dim: int = 512) -> None:
        self.embedding_dim = embedding_dim
        self.model: Optional[GradientBoostingClassifier] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.reference_embedding = np.zeros(self.embedding_dim, dtype=float)

    def _to_embedding(self, embedding: Optional[EmbeddingLike]) -> Optional[np.ndarray]:
        if embedding is None:
            return None
        array = np.array(embedding, dtype=float).reshape(-1)
        if array.shape[0] != self.embedding_dim:
            return None
        return array

    def _extract_pair_embeddings(
        self, pref: dict
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        winner_embedding = pref.get("winner_embedding")
        if winner_embedding is None:
            winner = pref.get("winner")
            if isinstance(winner, dict):
                winner_embedding = winner.get("embedding")

        loser_embedding = pref.get("loser_embedding")
        if loser_embedding is None:
            loser = pref.get("loser")
            if isinstance(loser, dict):
                loser_embedding = loser.get("embedding")

        winner_array = self._to_embedding(winner_embedding)
        loser_array = self._to_embedding(loser_embedding)
        if winner_array is None or loser_array is None:
            return None
        return winner_array, loser_array

    def _extract_grid_embedding(self, pref: dict) -> Optional[np.ndarray]:
        embedding = pref.get("embedding")
        if embedding is None:
            coa = pref.get("coa")
            if isinstance(coa, dict):
                embedding = coa.get("embedding")
        return self._to_embedding(embedding)

    def _build_pairs(self, preferences: List[dict]) -> List[Tuple[np.ndarray, np.ndarray]]:
        pairwise: List[Tuple[np.ndarray, np.ndarray]] = []
        selected: List[np.ndarray] = []
        rejected: List[np.ndarray] = []

        for pref in preferences:
            pref_type = pref.get("type")
            if pref_type == "pairwise":
                pair = self._extract_pair_embeddings(pref)
                if pair:
                    pairwise.append(pair)
            elif pref_type == "grid":
                embedding = self._extract_grid_embedding(pref)
                if embedding is None:
                    continue
                if pref.get("selected"):
                    selected.append(embedding)
                else:
                    rejected.append(embedding)

        if selected and rejected:
            for winner in selected:
                for loser in rejected:
                    pairwise.append((winner, loser))

        return pairwise

    def _compute_reference(self, preferences: List[dict]) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        for pref in preferences:
            pref_type = pref.get("type")
            if pref_type == "pairwise":
                pair = self._extract_pair_embeddings(pref)
                if pair:
                    embeddings.extend(pair)
            elif pref_type == "grid":
                embedding = self._extract_grid_embedding(pref)
                if embedding is not None:
                    embeddings.append(embedding)

        if not embeddings:
            return np.zeros(self.embedding_dim, dtype=float)

        return np.mean(np.stack(embeddings, axis=0), axis=0)

    def fit(self, preferences: List[dict]) -> Dict[str, float]:
        """Train the model from collected preferences."""
        pairs = self._build_pairs(preferences)
        if not pairs:
            self.model = None
            self.is_trained = False
            return {"accuracy": 0.0, "n_samples": 0}

        X: List[np.ndarray] = []
        y: List[int] = []

        for winner_embedding, loser_embedding in pairs:
            diff = winner_embedding - loser_embedding

            X.append(diff)
            y.append(1)
            X.append(-diff)
            y.append(0)

        X_array = np.array(X, dtype=float)
        y_array = np.array(y, dtype=int)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)

        self.model = GradientBoostingClassifier(random_state=42)
        self.model.fit(X_scaled, y_array)
        self.is_trained = True

        accuracy = float(self.model.score(X_scaled, y_array))
        self.reference_embedding = self._compute_reference(preferences)

        return {"accuracy": accuracy, "n_samples": int(len(y_array))}

    def score(self, embedding: EmbeddingLike) -> float:
        """Return a preference score in [0, 1] for a single embedding."""
        if not self.model or not self.is_trained:
            return 0.5

        vector = self._to_embedding(embedding)
        if vector is None:
            return 0.5

        diff = vector - self.reference_embedding
        X_scaled = self.scaler.transform([diff])
        return float(self.model.predict_proba(X_scaled)[0][1])

    def score_batch(self, embeddings: List[EmbeddingLike]) -> List[float]:
        """Score a batch of embeddings."""
        return [self.score(embedding) for embedding in embeddings]

    def get_uncertainty(self, embedding1: EmbeddingLike, embedding2: EmbeddingLike) -> float:
        """Return uncertainty for the comparison between two embeddings."""
        if not self.model or not self.is_trained:
            return 0.5

        vector1 = self._to_embedding(embedding1)
        vector2 = self._to_embedding(embedding2)
        if vector1 is None or vector2 is None:
            return 0.5

        diff = vector1 - vector2
        X_scaled = self.scaler.transform([diff])
        proba = float(self.model.predict_proba(X_scaled)[0][1])
        return float(1.0 - abs(proba - 0.5) * 2.0)

    def save(self, path: str) -> None:
        """Persist the model to disk."""
        dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "is_trained": self.is_trained,
                "embedding_dim": self.embedding_dim,
                "reference_embedding": self.reference_embedding,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the model from disk."""
        data = load(path)
        self.model = data.get("model")
        self.scaler = data.get("scaler", StandardScaler())
        self.is_trained = data.get("is_trained", False)
        self.embedding_dim = data.get("embedding_dim", self.embedding_dim)
        reference = data.get("reference_embedding")
        if reference is not None:
            reference_array = np.array(reference, dtype=float).reshape(-1)
            if reference_array.shape[0] == self.embedding_dim:
                self.reference_embedding = reference_array
            else:
                self.reference_embedding = np.zeros(self.embedding_dim, dtype=float)
        else:
            self.reference_embedding = np.zeros(self.embedding_dim, dtype=float)

    def get_feature_importance(self) -> Dict[str, float]:
        """Return embedding importance weights."""
        if not self.model or not self.is_trained:
            return {}

        importances = self.model.feature_importances_
        return {
            f"embedding_{idx}": float(value)
            for idx, value in enumerate(importances)
        }