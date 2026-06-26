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
The submodules carry the detail; this `__init__` re-exports the public API so `from replay_engine import X`
keeps working exactly as before.
"""
from .config import (RunConfig, build_task, build_model, task_kind, _spec_to_layer, log_spaced_steps,
                     ALL_TASKS, TRAIN_STREAM_OFFSET, EVAL_SEED, GRAD_SCALE_SEED_BASE)
from .splits import _split_mask, is_heldout, split_batch, sample_train_batch, exact_train_batch
from .gradients import (param_grad_norm, tensor_norm, _grad_norm, _grad_norms, _avg_grad_dicts,
                        layer_gradients, gradient_scale)
from .trace import trace_forward
from .sampling import sample_token, generate_sampled
from .training import (Checkpoint, TrainingRun, _train_one_step, _run_training, continue_training,
                       _capture_checkpoint, _eval_lm, _eval_pooled, per_category_eval, train_eval_curve,
                       _token_accuracy, _batch_score, _cpu_state)
