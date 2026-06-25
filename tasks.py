"""
tasks.py - a family of sequence tasks for the tiny GPT, all framed as next-token
prediction with the same layout:

    seq = [ input tokens ... | SEP | output tokens ... ]      (length = block_size)
    x   = seq[:, :-1]                                          (model input)
    y   = seq[:, 1:] with the input region set to -100         (loss only on the output)

Every task exposes the same small interface so the training / eval / replay code can
treat them uniformly:

    .name, .description, .params      (params = {kwarg: (min, max, default)} for the UI)
    .vocab_size, .block_size, .prompt_len, .gen_len
    .id_to_str  and  .decode(ids)
    .make_batch(batch_size, device="cpu", generator=None) -> (x, y, seq)

`prompt_len` = how many tokens are fed before generation starts (input + the SEP);
`gen_len`    = how many tokens the model must produce.

The "Sort" task lives in tiny_gpt.py (the notebook uses it); the others are here.
"""
from __future__ import annotations

import torch

from tiny_gpt import SortTask   # re-exported in the registry below


def _assemble(inp: torch.Tensor, out: torch.Tensor, sep_id, device):
    """Glue input (+ optional SEP) + output into (x, y, seq), masking loss on the prompt.

    Pass sep_id=None for tasks that need no separator (e.g. Lookup, where the answer must
    immediately follow the query so the induction circuit can form).
    """
    B = inp.shape[0]
    parts = [inp]
    if sep_id is not None:
        parts.append(torch.full((B, 1), sep_id, dtype=torch.long))
    parts.append(out)
    seq = torch.cat(parts, dim=1)
    prompt_len = seq.shape[1] - out.shape[1]      # input (+ separator)
    x = seq[:, :-1].contiguous()
    y = seq[:, 1:].clone()
    y[:, :prompt_len - 1] = -100                  # only score the tokens after the prompt
    return x.to(device), y.to(device), seq.to(device)


class SeqTask:
    """Base class: provides a default decode(). Subclasses set the attributes above."""
    name = "task"
    description = ""
    params: dict = {}

    def decode(self, ids) -> str:
        return " ".join(self.id_to_str.get(int(i), "?") for i in ids)


# ---------------------------------------------------------------------------
# Reverse: echo the input sequence backwards.  ->  clean anti-diagonal attention
# ---------------------------------------------------------------------------
class ReverseTask(SeqTask):
    name = "Reverse"
    description = "Echo the input sequence backwards."
    params = {"length": (4, 12, 8)}

    def __init__(self, length=8, n_symbols=10):
        self.length = length
        self.n_symbols = n_symbols
        self.sep_id = n_symbols
        self.vocab_size = n_symbols + 1
        self.block_size = 2 * length + 1
        self.prompt_len = length + 1
        self.gen_len = length
        self.id_to_str = {i: str(i) for i in range(n_symbols)}
        self.id_to_str[self.sep_id] = "|"

    def make_batch(self, batch_size, device="cpu", generator=None):
        inp = torch.randint(0, self.n_symbols, (batch_size, self.length), generator=generator)
        out = torch.flip(inp, dims=[1])
        return _assemble(inp, out, self.sep_id, device)


# ---------------------------------------------------------------------------
# Add: add two n-digit numbers. Output is the sum, LEAST-significant digit first,
# which lines the columns up left-to-right and is far easier to learn (the carry
# flows in the same direction the model generates).  ->  digit-column alignment
# ---------------------------------------------------------------------------
class AddTask(SeqTask):
    name = "Add"
    description = ("Add two n-digit numbers. The answer is written least-significant "
                   "digit first (reversed) — that's the order the carry naturally flows.")
    params = {"n_digits": (1, 4, 2)}

    PLUS = 10
    EQ = 11           # acts as the separator

    def __init__(self, n_digits=2):
        self.n_digits = n_digits
        self.sep_id = self.EQ
        self.vocab_size = 12                       # digits 0..9, '+', '='
        self.out_len = n_digits + 1               # room for a final carry
        self.in_len = 2 * n_digits + 1            # a-digits, '+', b-digits
        self.block_size = self.in_len + 1 + self.out_len
        self.prompt_len = self.in_len + 1
        self.gen_len = self.out_len
        self.id_to_str = {i: str(i) for i in range(10)}
        self.id_to_str[self.PLUS] = "+"
        self.id_to_str[self.EQ] = "="

    def _digits_msb(self, nums, k):
        return torch.stack([(nums // (10 ** p)) % 10 for p in range(k - 1, -1, -1)], dim=1)

    def make_batch(self, batch_size, device="cpu", generator=None):
        hi = 10 ** self.n_digits
        a = torch.randint(0, hi, (batch_size,), generator=generator)
        b = torch.randint(0, hi, (batch_size,), generator=generator)
        s = a + b
        plus = torch.full((batch_size, 1), self.PLUS, dtype=torch.long)
        inp = torch.cat([self._digits_msb(a, self.n_digits), plus,
                         self._digits_msb(b, self.n_digits)], dim=1)
        # output digits, least-significant first
        out = torch.stack([(s // (10 ** p)) % 10 for p in range(self.out_len)], dim=1)
        return _assemble(inp, out, self.sep_id, device)


# ---------------------------------------------------------------------------
# Arithmetic: like Add, but the example shows one of several operations (+ − ×) and the
# model must READ the operator and dispatch. Operations are a per-task setting (default: add).
# Answer is least-significant digit first (reversed), like Add. Subtract is kept non-negative
# (the bigger operand is shown first); multiply is the hard mode (output up to 2x as wide).
# ---------------------------------------------------------------------------
class ArithmeticTask(SeqTask):
    name = "Arithmetic"
    description = ("Apply the shown operation (+ − ×) to two n-digit numbers — the model must read "
                   "the operator and dispatch. Answer is least-significant digit first (reversed).")
    params = {
        "n_digits": (1, 3, 2),
        "ops": {"type": "multiselect", "options": ["add", "subtract", "multiply"], "default": ["add"]},
    }
    PLUS, MINUS, TIMES, EQ = 10, 11, 12, 13       # operator tokens + '=' separator
    _OP = {"add": PLUS, "subtract": MINUS, "multiply": TIMES}

    def __init__(self, n_digits=2, ops=("add",)):
        self.n_digits = n_digits
        self.ops = [o for o in ("add", "subtract", "multiply") if o in (ops or ["add"])] or ["add"]
        self.sep_id = self.EQ
        self.vocab_size = 14                       # digits 0..9, '+', '−', '×', '='
        w = n_digits + 1                           # add carries to n+1; subtract (a>=b) <= n
        if "multiply" in self.ops:
            w = max(w, 2 * n_digits)               # product is up to 2n digits
        self.out_len = w
        self.in_len = 2 * n_digits + 1             # a-digits, OP, b-digits
        self.block_size = self.in_len + 1 + self.out_len
        self.prompt_len = self.in_len + 1
        self.gen_len = self.out_len
        self.id_to_str = {i: str(i) for i in range(10)}
        self.id_to_str.update({self.PLUS: "+", self.MINUS: "−", self.TIMES: "×", self.EQ: "="})

    def _digits_msb(self, nums, k):
        return torch.stack([(nums // (10 ** p)) % 10 for p in range(k - 1, -1, -1)], dim=1)

    def make_batch(self, batch_size, device="cpu", generator=None):
        hi = 10 ** self.n_digits
        a = torch.randint(0, hi, (batch_size,), generator=generator)
        b = torch.randint(0, hi, (batch_size,), generator=generator)
        op_ids = torch.tensor([self._OP[o] for o in self.ops])
        pick = op_ids[torch.randint(0, len(self.ops), (batch_size,), generator=generator)]
        hi_ab, lo_ab = torch.maximum(a, b), torch.minimum(a, b)   # subtract: show bigger first -> >= 0
        res = torch.zeros(batch_size, dtype=torch.long)
        res = torch.where(pick == self.PLUS, a + b, res)
        res = torch.where(pick == self.MINUS, hi_ab - lo_ab, res)
        res = torch.where(pick == self.TIMES, a * b, res)
        a_show = torch.where(pick == self.MINUS, hi_ab, a)
        b_show = torch.where(pick == self.MINUS, lo_ab, b)
        inp = torch.cat([self._digits_msb(a_show, self.n_digits), pick.unsqueeze(1),
                         self._digits_msb(b_show, self.n_digits)], dim=1)
        out = torch.stack([(res // (10 ** p)) % 10 for p in range(self.out_len)], dim=1)   # LSB first
        return _assemble(inp, out, self.sep_id, device)


# ---------------------------------------------------------------------------
# Lookup: read a list of (key, value) pairs, then a query key; output its value.
# This is associative recall ("induction heads", the mechanism behind in-context
# learning). It needs a content-based TWO-hop circuit (a previous-token head +
# an induction head), which a tiny absolute-position transformer struggles to
# optimize — in practice it gets stuck well short of solving. So it is kept here
# for reference / experimentation but is NOT in the default TASKS registry; the
# 1-hop "Index" task is the reliable attention-retrieval demo instead.
# ---------------------------------------------------------------------------
class LookupTask(SeqTask):
    name = "Lookup"
    description = ("Read the (key, value) pairs, then answer with the value of the final "
                   "query key. Keys are letters, values are digits.")
    params = {"n_pairs": (3, 8, 5)}

    def __init__(self, n_pairs=5, n_keys=12, n_values=10):
        self.n_pairs = n_pairs
        self.n_keys = n_keys
        self.n_values = n_values
        self.val_off = n_keys                     # value ids start after the key ids
        self.vocab_size = n_keys + n_values       # no separator: answer follows the query
        self.in_len = 2 * n_pairs + 1             # (k v) * K, then the query key
        self.out_len = 1
        self.block_size = self.in_len + self.out_len
        self.prompt_len = self.in_len             # the query key is the last prompt token
        self.gen_len = 1
        self.id_to_str = {i: chr(ord("a") + i) for i in range(n_keys)}
        self.id_to_str.update({self.val_off + j: str(j) for j in range(n_values)})

    def make_batch(self, batch_size, device="cpu", generator=None):
        B, K = batch_size, self.n_pairs
        rows = torch.arange(B)
        keys = torch.rand(B, self.n_keys, generator=generator).argsort(dim=1)[:, :K]   # distinct keys
        values = torch.randint(0, self.n_values, (B, K), generator=generator) + self.val_off
        pairs = torch.stack([keys, values], dim=2).reshape(B, 2 * K)                    # k v k v ...
        qpos = torch.randint(0, K, (B,), generator=generator)
        query = keys[rows, qpos].unsqueeze(1)
        answer = values[rows, qpos].unsqueeze(1)
        inp = torch.cat([pairs, query], dim=1)
        return _assemble(inp, answer, None, device)   # no separator -> proper induction setup


# ---------------------------------------------------------------------------
# Index: read a list, then an index token; output the item at that position.
# This is "attention as random access" — the query maps to one position and copies
# it (one hop, like Reverse), so it trains reliably.  ->  a single bright cell.
# ---------------------------------------------------------------------------
class IndexTask(SeqTask):
    name = "Index"
    description = "Random access: read the list, then an index @i; output the i-th item."
    params = {"length": (4, 10, 6)}

    def __init__(self, length=6, n_symbols=10):
        self.length = length
        self.n_symbols = n_symbols
        self.idx_off = n_symbols                   # index tokens come after the symbols
        self.vocab_size = n_symbols + length
        self.in_len = length + 1                   # the list, then the index token
        self.out_len = 1
        self.block_size = self.in_len + self.out_len
        self.prompt_len = self.in_len
        self.gen_len = 1
        self.id_to_str = {i: str(i) for i in range(n_symbols)}
        self.id_to_str.update({self.idx_off + p: f"@{p}" for p in range(length)})

    def make_batch(self, batch_size, device="cpu", generator=None):
        B, L = batch_size, self.length
        rows = torch.arange(B)
        data = torch.randint(0, self.n_symbols, (B, L), generator=generator)
        idx = torch.randint(0, L, (B,), generator=generator)
        idx_tok = (idx + self.idx_off).unsqueeze(1)
        answer = data[rows, idx].unsqueeze(1)
        inp = torch.cat([data, idx_tok], dim=1)
        return _assemble(inp, answer, None, device)


# ---------------------------------------------------------------------------
# Registry + a helper to render one example for the UI.
# ---------------------------------------------------------------------------
TASKS = {
    "Sort": SortTask,
    "Reverse": ReverseTask,
    "Add": AddTask,
    "Arithmetic": ArithmeticTask,
    "Index": IndexTask,
}


def task_example(task, seed=0) -> str:
    """A single human-readable 'prompt  ->  output' example for the given task."""
    _, _, seq = task.make_batch(1, generator=torch.Generator().manual_seed(seed))
    return f"{task.decode(seq[0, :task.prompt_len])}   ->   {task.decode(seq[0, task.prompt_len:])}"


def example_prompt(task, seed=0) -> str:
    """Just the decoded prompt (input region) of one example — pre-fills the 'try it' box."""
    _, _, seq = task.make_batch(1, generator=torch.Generator().manual_seed(seed))
    return task.decode(seq[0, :task.prompt_len])


def encode_prompt(task, text):
    """Parse a typed prompt (in the task's display notation) into a [1, prompt_len] tensor.

    Returns (tensor, None) on success, or (None, error_message) if it can't be parsed.
    """
    str_to_id = {v: k for k, v in task.id_to_str.items()}
    ids = []
    for tok in text.split():
        if tok not in str_to_id:
            allowed = " ".join(map(str, sorted(set(task.id_to_str.values()))))
            return None, f"unknown token '{tok}'. Allowed: {allowed}"
        ids.append(str_to_id[tok])
    if len(ids) != task.prompt_len:
        return None, f"please enter exactly {task.prompt_len} tokens (got {len(ids)})."
    return torch.tensor([ids], dtype=torch.long), None
