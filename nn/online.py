"""Frozen NN + lightweight per-player adaptive residual with controlled forgetting.

Two distinct mechanisms (do not conflate them):
 * FEATURE DROPOUT is train-time regularization on NEW mini-batches only. It does
   not erase stored knowledge.
 * EPISODIC FORGETTING deliberately zeros a random subset of *adapter* weights
   after every N mini-batch updates. This DOES erase some online calibration,
   while the pretrained model remains untouched. Later fresh ten-sample batches
   can re-learn the residual. Recovery is measured, never guaranteed.

This is parameter-efficient fine-tuning in spirit, but NOT LoRA (no low-rank
matrix factorization). No replay buffer, historical retraining, or base updates.
"""
from __future__ import annotations
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from data.make_data import FEATURES
from nn.model import MLP, sigmoid

ONLINE_DIR = Path(__file__).resolve().parents[1] / 'models' / 'online'

@dataclass
class OnlineAdapter:
    owner: str
    batch_size: int = 10
    learning_rate: float = 0.055
    l2: float = 0.025
    feature_dropout: float = 0.12
    forget_every: int = 3  # number of online ten-example updates between resets
    forget_fraction: float = 0.35
    forgetting_enabled: bool = True
    recovery_boost: float = 1.4
    recovery_updates_remaining: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros(len(FEATURES) + 1))
    updates: int = 0
    observations: int = 0
    pending: list[tuple[list[float], float, int]] = field(default_factory=list)
    forget_events: list[dict] = field(default_factory=list)
    last_update_loss: float | None = None
    prequential_losses: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.batch_size < 1: raise ValueError('batch_size must be positive')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,50}', self.owner):
            raise ValueError('Unsafe adapter name')
        if not (0 <= self.feature_dropout < 1):
            raise ValueError('feature_dropout must be in [0,1)')
        if not (0 <= self.forget_fraction <= 1):
            raise ValueError('forget_fraction must be in [0,1]')
        if self.forget_every < 1: raise ValueError('forget_every must be >=1')
        if self.recovery_boost < 1: raise ValueError('recovery_boost must be >=1')
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.shape != (len(FEATURES) + 1,):
            raise ValueError('Adapter feature schema mismatch')

    @property
    def last_forget_event(self) -> dict | None:
        return self.forget_events[-1] if self.forget_events else None

    def _rng(self, salt: int) -> np.random.Generator:
        # Python's built-in hash randomizes across runs. CRC32 is stable.
        stable = zlib.crc32(self.owner.encode('utf8'))
        return np.random.default_rng((stable + salt * 2654435761) % (2**32))

    def design(self, features: list[float] | np.ndarray, model: MLP) -> np.ndarray:
        x = np.asarray(features, dtype=float).reshape(-1)
        if x.shape != (len(FEATURES),): raise ValueError('Wrong feature vector')
        return np.r_[1., np.clip(model.scale(x), -3., 3.)]

    def predict(self, model: MLP, features: list[float] | np.ndarray) -> float:
        base = float(np.clip(model.predict_proba(features)[0], 1e-5, 1 - 1e-5))
        base_logit = np.log(base / (1 - base))
        return float(sigmoid(base_logit + np.dot(self.weights, self.design(features, model))))

    def forget_now(self, reason: str = 'scheduled') -> dict | None:
        """Erase a fraction of learned *adapter* feature coefficients, never base.

        Choose from currently nonzero features to ensure an actual measurable
        perturbation after learning. Bias and pending examples are preserved.
        """
        eligible = np.flatnonzero(np.abs(self.weights[1:]) > 1e-12) + 1
        if not self.forgetting_enabled or self.forget_fraction == 0 or not len(eligible):
            return None
        count = max(1, round(len(FEATURES) * self.forget_fraction))
        count = min(len(eligible), count)
        rng = self._rng(self.updates * 117 + len(self.forget_events) + 19)
        chosen = rng.choice(eligible, size=count, replace=False)
        before = float(np.linalg.norm(self.weights))
        self.weights[chosen] = 0.0
        after = float(np.linalg.norm(self.weights))
        event = {'update': self.updates, 'observations': self.observations,
                 'reason': reason, 'features_erased': [FEATURES[i-1] for i in sorted(chosen)],
                 'erased_count': count, 'adapter_norm_before': round(before, 6),
                 'adapter_norm_after': round(after, 6),
                 'pretrained_model_modified': False}
        self.forget_events.append(event)
        # Exactly two *future* fresh batches can rebuild calibration faster.
        # This is extra SGD on new evidence, not replay or a base-model update.
        self.recovery_updates_remaining = 2
        return event

    def observe(self, model: MLP, features: list[float], won: bool,
                persist: bool = False) -> bool:
        """Return True when a NEW batch of exactly `batch_size` outcomes updates.

        No older samples are used for gradient steps; only the frozen base
        prediction and latest observation features influence this update.
        """
        x = [float(v) for v in features]
        base = float(np.clip(model.predict_proba(x)[0], 1e-5, 1 - 1e-5))
        base_logit = float(np.log(base / (1 - base)))
        self.pending.append((self.design(x, model).tolist(), base_logit, int(won)))
        self.observations += 1
        if len(self.pending) < self.batch_size:
            return False
        assert len(self.pending) == self.batch_size
        X = np.asarray([e[0] for e in self.pending], dtype=float)
        logits = np.asarray([e[1] for e in self.pending], dtype=float)
        y = np.asarray([e[2] for e in self.pending], dtype=float)
        before_pred = np.clip(sigmoid(logits + X @ self.weights), 1e-7, 1 - 1e-7)
        log_loss_before = float(np.mean(-y * np.log(before_pred) - (1-y) * np.log(1-before_pred)))
        self.prequential_losses.append(round(log_loss_before, 6))
        rng = self._rng(self.updates + 1)
        lr = self.learning_rate * (self.recovery_boost if self.recovery_updates_remaining else 1.)
        for _ in range(55):
            # Training-only inverted dropout: preserve expected feature scale.
            train_x = X.copy()
            if self.feature_dropout:
                keep = (rng.random(train_x[:, 1:].shape) >= self.feature_dropout)
                train_x[:, 1:] *= keep / (1.0 - self.feature_dropout)
            p = sigmoid(logits + train_x @ self.weights)
            grad = (train_x.T @ (p - y)) / len(y)
            grad[1:] += self.l2 * self.weights[1:]
            self.weights -= lr * np.clip(grad, -2.0, 2.0)
            self.weights = np.clip(self.weights, -2.5, 2.5)
        self.updates += 1
        if self.recovery_updates_remaining:
            self.recovery_updates_remaining -= 1
        self.pending.clear()
        self.last_update_loss = round(log_loss_before, 6)
        if self.forgetting_enabled and self.updates % self.forget_every == 0:
            self.forget_now('scheduled')
        if persist: self.save()
        return True

    def save(self, directory: Path = ONLINE_DIR):
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        payload = {'owner': self.owner, 'features': list(FEATURES),
                   'batch_size': self.batch_size, 'weights': self.weights.tolist(),
                   'updates': self.updates, 'observations': self.observations,
                   'pending': self.pending, 'feature_dropout': self.feature_dropout,
                   'forget_every': self.forget_every, 'forget_fraction': self.forget_fraction,
                   'forgetting_enabled': self.forgetting_enabled,
                   'recovery_boost': self.recovery_boost,
                   'recovery_updates_remaining': self.recovery_updates_remaining,
                   'forget_events': self.forget_events,
                   'prequential_losses': self.prequential_losses,
                   'last_update_loss': self.last_update_loss,
                   'method': 'frozen_base_plus_feature_dropout_plus_partial_adapter_forgetting'}
        dest = directory / f'{self.owner}.json'
        tmp = dest.with_suffix('.tmp')
        tmp.write_text(json.dumps(payload, indent=2), encoding='utf8')
        tmp.replace(dest)
        return dest

    @classmethod
    def load(cls, owner: str, directory: Path = ONLINE_DIR,
             batch_size: int = 10) -> 'OnlineAdapter':
        result = cls(owner, batch_size=batch_size)
        path = Path(directory) / f'{owner}.json'
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf8'))
            if payload['features'] != list(FEATURES) or payload['owner'] != owner:
                raise ValueError('Saved adapter schema/owner mismatch')
            if payload['batch_size'] != batch_size:
                raise ValueError('Saved batch size differs')
            result.weights = np.asarray(payload['weights'], dtype=float)
            if result.weights.shape != (len(FEATURES) + 1,):
                raise ValueError('Adapter shape mismatch')
            result.updates = int(payload['updates'])
            result.observations = int(payload['observations'])
            result.pending = [(list(map(float, x)), float(base), int(y))
                              for x, base, y in payload.get('pending', [])]
            for key in ('feature_dropout', 'forget_every', 'forget_fraction', 'forgetting_enabled',
                        'recovery_boost', 'recovery_updates_remaining'):
                if key in payload: setattr(result, key, payload[key])
            result.forget_events = list(payload.get('forget_events', []))
            result.prequential_losses = list(payload.get('prequential_losses', []))
            result.last_update_loss = payload.get('last_update_loss')
            result.__post_init__()
        return result
