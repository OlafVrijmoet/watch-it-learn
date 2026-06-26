"""replay_engine.sampling — the Output head's generation knobs (greedy / temperature / top-k / top-p)."""
from __future__ import annotations

import torch


def sample_token(logits, method="greedy", temperature=1.0, top_k=0, top_p=0.0, generator=None):
    """Turn a 1-D logits vector into one chosen token id (int).

    method: 'greedy' (argmax) | 'temperature' | 'top_k' | 'top_p' (nucleus).
    'temperature' alone also applies to the top_k/top_p variants.
    """
    logits = logits.detach().float().clone()
    if method == "greedy":
        return int(logits.argmax().item())

    if temperature and temperature > 0:
        logits = logits / temperature

    if method == "top_k" and top_k and top_k > 0:
        k = min(int(top_k), logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    if method == "top_p" and top_p and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cum > top_p
        cutoff[0] = False                       # always keep the top token
        remove = sorted_idx[cutoff]
        logits[remove] = float("-inf")

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate_sampled(model, task, seq, method="greedy", temperature=1.0, top_k=0, top_p=0.0,
                     seed=0, device="cpu"):
    """Autoregressively generate the output region with the chosen sampling method.

    Works for a batch: each row is sampled independently. With method='greedy' this is
    exactly the argmax decoding in tiny_gpt.generate.
    """
    p = task.prompt_len
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    if not torch.is_tensor(seq):
        seq = torch.tensor(seq, dtype=torch.long)
    ctx = seq[:, :p].to(device)
    B = ctx.shape[0]
    for _ in range(task.gen_len):
        logits, _ = model(ctx)
        last = logits[:, -1, :]                                      # [B, vocab]
        nxt = torch.tensor(
            [sample_token(last[b], method, temperature, top_k, top_p, gen) for b in range(B)],
            device=ctx.device, dtype=torch.long).unsqueeze(1)        # [B, 1]
        ctx = torch.cat([ctx, nxt], dim=1)
    return ctx[:, p:]
