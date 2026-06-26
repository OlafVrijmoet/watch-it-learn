"""Tiny end-to-end demo:  python -m replay_engine"""
import json

import torch

from . import RunConfig, TrainingRun

if __name__ == "__main__":
    cfg = RunConfig(task_name="Reverse", task_kwargs={"length": 5},
                    layer_specs=(("attn",), ("ffn", (64,))),
                    d_model=32, n_heads=4, steps=150, n_checkpoints=20, seed=0, device="cpu")
    run = TrainingRun.train(cfg)
    print(f"params={run.n_params}  checkpoints={len(run.checkpoints)} at {run.checkpoint_steps}")
    print(f"first acc={run.checkpoints[0].acc:.2f}  last acc={run.checkpoints[-1].acc:.2f}")
    print(f"checkpoint storage = {run.nbytes()/1e6:.2f} MB")
    task = run.task
    _, _, probe = task.make_batch(1, generator=torch.Generator().manual_seed(3))
    fr = run.frame(run.checkpoint_steps[-1], probe[0])
    print("trace keys:", list(fr["trace"]))
    print("attention layers in trace:", len(fr["trace"]["attention"]))
    print("JSON-serializable:", bool(json.dumps(fr)))
