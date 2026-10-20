# regularization_labs

Regularization Lab: Character-level RNN/LSTM on Tiny Shakespeare (from scratch, NumPy only)

Build a character-level RNN **from scratch** (forward pass + BPTT backward pass, no ML frameworks) and study regularization techniques:

- **Dropout** on non-recurrent connections (inverted, fresh mask each call)
- **Recurrent dropout** on the hidden state `h(t-1)` inside the RNN/LSTM
  - **Variational** (one mask per sequence, reused at every time step) — *correct*
  - **Per-step** (fresh mask every step) — *the bad way*
- **L2 weight decay** (added to gradient) vs **decoupled weight decay** (AdamW)
- **Label smoothing** in cross-entropy loss
- **Gradient clipping** (global norm)
- **Warmup + cosine LR schedule**
- **Early stopping** (restores best weights)
- **Batch Normalization** (feature-axis, with correct backward including variance term)

Compared across optimizers:
- **SGD** with momentum
- **AdaGrad**
- **RMSProp**
- **Adam**
- **AdamW** (decoupled decay)



---
The notebook:
1. Downloads Tiny Shakespeare (~1.1 MB) and uses the first 100k chars for training, next 20k for validation (61 unique characters).
2. Trains a baseline LSTM (`E=16, H=64`) that overfits — train loss drops, val loss rises.
3. Runs an ablation grid over regularizers (dropout, recurrent dropout, label smoothing, L2) (resumable).
4. Runs an optimizer comparison grid (SGD, AdaGrad, RMSProp, Adam, AdamW) with the best regularizer combo.
5. Renders SVG loss curves, gap curves, and grouped bar charts inline.
---



## Key implementation notes

- **Variational recurrent dropout** (`per_step_masks=False`): one mask per sequence, applied identically at every time step. The backward pass re-applies the same mask. This preserves the temporal structure while regularizing.
- **Per-step recurrent dropout** (`per_step_masks=True`): fresh mask each step. The gradient checker exposes this as a worse regularizer (it destroys information each step).
- **AdamW decoupled decay** acts on `p.data` directly, independent of the adaptive learning rate. `Adam + L2` does not — the penalty is rescaled by `1/sqrt(v)` and becomes ineffective.
- **BatchNorm backward** includes the variance term: `dx = inv/N * (N*dxhat - sum(dxhat) - xhat * sum(dxhat*xhat))`. Two negative controls test forgetting the variance term and ignoring batch statistics.
- **Parameter.decay=False** marks biases and BN gamma/beta so weight penalties skip them.
- **Gradient checks** freeze dropout masks (`layer.freeze(...)`) for determinism; they test `dx`, `dW`, `db`, `dgamma`, `dbeta` as appropriate.
- **`run_grid`** caches every finished config to JSON; re-running skips completed configs automatically.

---

