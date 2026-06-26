"""replay_engine — train a configurable tiny transformer while recording its *training history*,
so the whole run can be scrubbed and every state faithfully reconstructed and inspected.

This is the technical backbone for the model-builder UI. It implements the locked storage design:

  * DENSE scalar metrics, every step           -> the smooth loss/accuracy curves.
  * SPARSE weight checkpoints, LOG-SPACED       -> the only heavy thing; a `state_dict` (on CPU)
    at each checkpoint step. A "# checkpoints" knob can crank density up to EVERY step (full fidelity).
  * EXACT reconstruction. A checkpoint is the real `state_dict` (bit-for-bit); plus seed + config +
    initial weights, so ANY step is reconstructable exactly via deterministic replay (re-run on CPU).
  * Activations are NOT stored - they are recomputed on demand (`trace_forward`).

This package splits that backbone by cohesion (config / splits / gradients / training / trace / sampling).
The submodules carry the detail; this `__init__` re-exports the public API (declared in `__all__`) so
`from replay_engine import X` works as before. Internal helpers stay in their submodules.
"""
from .config import RunConfig, build_task, build_model, log_spaced_steps, ALL_TASKS
from .splits import is_heldout, sample_train_batch, exact_train_batch
from .gradients import layer_gradients, gradient_scale
from .trace import trace_forward
from .sampling import sample_token, generate_sampled
from .training import TrainingRun, continue_training, per_category_eval, train_eval_curve

__all__ = [
    "RunConfig", "build_task", "build_model", "log_spaced_steps", "ALL_TASKS",
    "is_heldout", "sample_train_batch", "exact_train_batch",
    "layer_gradients", "gradient_scale",
    "trace_forward",
    "sample_token", "generate_sampled",
    "TrainingRun", "continue_training", "per_category_eval", "train_eval_curve",
]
