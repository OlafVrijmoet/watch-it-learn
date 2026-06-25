# 🔬 Glassbox Transformer

> *Working title — a few name options at the bottom; pick whichever you like.*

**Assemble a tiny transformer block-by-block, train it, and scrub the entire training run — watching attention, activations, and gradients evolve step by step.** A hands-on microscope for *how* a transformer learns, built as a single Streamlit app.

<!-- TODO(step 4): hero GIF — build → train → scrub stages → gradient flow -->
<!-- ![demo](docs/demo.gif) -->

**▶ Live demo:** _coming soon (Streamlit Community Cloud)_

---

## Why

Reading about attention and backprop is one thing; *watching* them is another. I wanted to actually see a transformer learn — so I built a tool where you compose the architecture yourself, press train, then replay the whole run: the residual stream lighting up, attention sharpening, and the gradient signal flowing back and fading as it converges. Every number on screen is real and recomputed from the model, not a canned animation.

## What you can do

- **Build the model as a canvas.** Add attention / feed-forward sections, each with its own settings — number of heads (d_head derived), hidden layers & width, activation, bias, causal mask, dropout, an optional W_O hidden layer, positional encoding (learned / sinusoidal), and the output head's pooling & sampling. Global **model defaults** cascade to every section unless a section overrides them.
- **Choose how weights start.** Init scheme (normal / Xavier / Kaiming / orthogonal / zeros) + seed — then watch the effect on the very first gradients.
- **Train with live curves.** Loss and held-out accuracy draw in as it trains; **continue training** to warm-start and add more steps to the same run.
- **Scrub the whole training history.** A stage slider reconstructs the model at *any* recorded step and renders it on *your* editable probe input (with a 🟡 held-out / 🟢 in-training badge). The "Try the trained model" panel generates at the selected stage.
- **See the gradient flow.** Per-block bars (embed · each head's Q/K/V/O · per-FFN) plus an on-model overlay — ∇ badges per section and per-neuron halos — with log scale, a run-wide **fixed scale** to compare across steps, and several gradient sources (averaged / a scrubbed batch / the exact training batch / your probe).
- **Trust the metrics.** Accuracy is measured on a **held-out 20%** the model never trained on (deterministic split), with an optional train-vs-held-out overlay to see the generalization gap.
- **Play the run** as a smooth animation, or zoom / pan the model.
- **Run comparative experiments.** Hold several model **versions** in one experiment (the task is shared; each version owns its architecture, hyperparameters, init & seed), **train them all** with live loss/accuracy curves (overlaid or stacked), then open the **Compare** page: loss & accuracy overlays or small-multiples, held-out vs train curves, a **gradient-flow comparison** (total ‖∇‖ over training + per-block ‖∇‖ by version), and a status table (params · memory · steps).

## The tasks

Procedurally generated (so accuracy reflects generalization, not memorization), each framed as next-token prediction:

| Task | What it teaches |
|------|-----------------|
| **Reverse** | clean anti-diagonal attention (copy in reverse) |
| **Add** | digit-column alignment + carry (answer written least-significant-digit-first) |
| **Arithmetic** | read an operator (+ − ×) and *dispatch* — operations are a per-task setting |
| **Index** | attention as random access (one-hop retrieval) |
| **Sort** | ordering |
| **Majority / Density** | pooled classification & regression heads |

## How it works (the interesting bits)

- **Deterministic training replay.** Training records dense per-step metrics plus **log-spaced weight checkpoints** (dense early, sparse late). Any step is reconstructed bit-for-bit — snap-to-nearest checkpoint, or a full deterministic re-run — and activations are **recomputed on demand**, never stored. Storage is `checkpoints × params`; per-frame compute is O(1).
- **On-the-fly gradient introspection.** Gradients are never stored (that would be model-size × steps). Instead they're computed live: reconstruct the weights at a step, run a single backward pass, aggregate per block / per head / per neuron.
- **Honest generalization.** Train/test are split by a deterministic hash of the input, so every reported accuracy is on inputs the model never saw.
- **A from-scratch SVG renderer.** The model view is hand-built SVG matching a locked Figma design — no charting framework — which means it also rasterizes to PNG (via cairosvg) for genuine, browser-free render tests.
- **Comparative experiments, for free.** Each version is just an independent `RunConfig` + `TrainingRun`, so holding several and comparing their curves and gradients reuses the same engine — no extra machinery.
- **Tested.** 54 tests cover the engine (reconstruction is bit-exact), the model, the renderer, the app, and the experiments / compare flow.

## Quickstart

```bash
# runtime deps: streamlit, torch, matplotlib, pandas, numpy   (a requirements.txt is coming)
pip install streamlit torch matplotlib pandas numpy

streamlit run app.py
```

Then: **Build** a model (or keep the default) → **Train** → open the **Run** tab to scrub stages, feed your own input, and toggle the gradient view. To compare architectures, add **versions**, hit **▶ Train all untrained**, and switch to the **Compare** page.

Run the tests:

```bash
for t in test_replay_engine test_builder_model test_flow_svg test_flow_component test_builder_app; do
  python $t.py
done
```

## Project structure

| Module | Role |
|--------|------|
| `app.py` | entry point — the two-page nav (🛠 Builder · 📊 Compare) |
| `builder_app.py` | the **Builder** page — the build canvas, Build / Train / Run sidebar, and Versions panel |
| `compare_page.py` | the **Compare** page — multi-version curves, gradient-flow comparison, status table |
| `builder_model.py` | the configurable transformer (`BuilderModel` + per-layer configs) |
| `replay_engine.py` | training with recorded history, exact reconstruction, on-the-fly gradients |
| `flow_svg.py` | the SVG renderer — whole-model view, gradient overlay, training replay |
| `flow_component.py` | heatmap view for the pooled (classification / regression) heads |
| `tasks.py` | the task family (Reverse, Add, Arithmetic, Index, …) |
| `tiny_gpt.py` | minimal GPT primitives + the Sort task |
| `training_utils.py` | optimizer factory, LR schedule, hyperparameter heuristics |

## A bit of history

This grew out of a "teach a tiny GPT to sort" notebook and a TensorFlow-Playground-style 2D classifier experiment. Both were scaffolding for the real goal — *seeing* training happen — and were retired once the builder + replay engine could show the whole story interactively.

## Naming options

`glassbox-transformer` · `transformer-microscope` · `tiny-transformer-lab` · `watch-it-learn`

---

<!-- TODO(step 3): LICENSE (MIT), requirements.txt, .gitignore -->
<!-- TODO(step 4): GitHub repo + push, Streamlit Community Cloud deploy, GIF/screenshots in docs/ -->
