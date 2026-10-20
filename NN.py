
import numpy as np


class Parameter:
    """A learnable array and its gradient. decay=False marks parameters that weight penalties must skip
    (biases, BatchNorm gamma/beta): shrinking those towards 0 is not regularisation, it just hurts."""

    def __init__(self, data, decay=True):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.decay = decay

    def zero_grad(self):
        self.grad[...] = 0.0

#  THE MAIN TREE BEHIND ALL THIS DRAMAAA(alfred stop it asap)
class Module:
    training = True

    def _children(self):
        for v in vars(self).values():
            if isinstance(v, Module):
                yield v
            elif isinstance(v, (list, tuple)):
                yield from (m for m in v if isinstance(m, Module))

    def parameters(self):
        ps = []
        for v in vars(self).values():
            if isinstance(v, Parameter):
                ps.append(v)
            elif isinstance(v, (list, tuple)):
                ps += [p for p in v if isinstance(p, Parameter)]
        for c in self._children():
            ps += c.parameters()
        return ps

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode=True):
        self.training = mode
        for c in self._children():
            c.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def __call__(self, *a, **k):
        return self.forward(*a, **k)

# WELL ACTIVATION FUNCTIONS FWD PASS AND BACKWARD(bwd) PASS
class Linear(Module):
    def __init__(self, nin, nout, bias=True, rng=None, init="he"):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt((2.0 if init == "he" else 1.0) / nin)
        self.W = Parameter(rng.standard_normal((nin, nout)) * scale)
        self.b = Parameter(np.zeros(nout), decay=False) if bias else None

    def forward(self, x):                       # x: (..., nin)
        self._x = x
        y = x @ self.W.data
        return y + self.b.data if self.b is not None else y

    def backward(self, dy):
        x2 = self._x.reshape(-1, self._x.shape[-1])
        dy2 = dy.reshape(-1, dy.shape[-1])
        self.W.grad += x2.T @ dy2
        if self.b is not None:
            self.b.grad += dy2.sum(0)
        return dy @ self.W.data.T


class ReLU(Module):
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, dy):
        return dy * self._mask


class Tanh(Module):
    def forward(self, x):
        self._y = np.tanh(x)
        return self._y

    def backward(self, dy):
        return dy * (1.0 - self._y ** 2)


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def backward(self, dy):
        for l in reversed(self.layers):
            dy = l.backward(dy)
        return dy

class Dropout(Module):
    """Inverted dropout: kept units are scaled by 1/(1-p) at train time, so eval is a plain identity.
    forward(x, mask=None) accepts an external mask: the RNN will sample ONE mask per sequence
    (variational dropout) and pass the same one at every time step.
    freeze(shape) pins a mask so repeated forward() calls are deterministic (needed for gradient checks)."""

    def __init__(self, p=0.5, rng=None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = rng or np.random.default_rng(0)
        self._fixed = None
        self._mask = None

    def sample_mask(self, shape):
        return (self.rng.random(shape) >= self.p) / (1.0 - self.p)

    def freeze(self, shape):
        self._fixed = self.sample_mask(shape)
        return self

    def unfreeze(self):
        self._fixed = None
        return self

    def forward(self, x, mask=None):
        if not self.training or self.p == 0.0:
            self._mask = None
            return x
        if mask is None:
            mask = self._fixed if self._fixed is not None else self.sample_mask(x.shape)
        self._mask = mask
        return x * mask

    def backward(self, dy):
        return dy if self._mask is None else dy * self._mask


class BatchNorm(Module):
    """Batch norm over the LAST axis (features). All leading axes form the batch: (N,C), (B,T,C), ...
    Train: normalise with batch statistics (biased variance) and update running stats (unbiased variance).
    Eval:  normalise with the running statistics."""

    def __init__(self, C, eps=1e-5, momentum=0.1):
        self.gamma = Parameter(np.ones(C), decay=False)
        self.beta = Parameter(np.zeros(C), decay=False)
        self.eps, self.momentum = eps, momentum
        self.running_mean = np.zeros(C)
        self.running_var = np.ones(C)

    def forward(self, x):
        self._shape = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        N = len(x2)
        if self.training:
            mu, var = x2.mean(0), x2.var(0)
            m = self.momentum
            self.running_mean = (1 - m) * self.running_mean + m * mu
            self.running_var = (1 - m) * self.running_var + m * var * N / max(N - 1, 1)
        else:
            mu, var = self.running_mean, self.running_var
        self._inv = 1.0 / np.sqrt(var + self.eps)
        self._xhat = (x2 - mu) * self._inv
        return (self.gamma.data * self._xhat + self.beta.data).reshape(self._shape)

    def backward(self, dy):
        dy2 = dy.reshape(-1, dy.shape[-1])
        N = len(dy2)
        self.gamma.grad += (dy2 * self._xhat).sum(0)
        self.beta.grad += dy2.sum(0)
        dxhat = dy2 * self.gamma.data
        if self.training:
            # mean AND variance both depend on x, so each x_i also reaches every other output in the batch:
            # dx = inv/N * ( N*dxhat - sum(dxhat) - xhat * sum(dxhat*xhat) )
            dx = self._inv / N * (N * dxhat - dxhat.sum(0) - self._xhat * (dxhat * self._xhat).sum(0))
        else:
            dx = dxhat * self._inv            # running stats are constants, so no cross-sample terms
        return dx.reshape(self._shape)


def cross_entropy(logits, targets, label_smoothing=0.0):
    """Stable softmax-CE with label smoothing. logits (..., V), targets int (...).
    Returns (mean loss, dlogits). dlogits = (softmax - smoothed_target) / N."""
    shape, V = logits.shape, logits.shape[-1]
    x, idx = logits.reshape(-1, V), targets.reshape(-1)
    N = len(x)
    z = x - x.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    t = np.full((N, V), label_smoothing / V)
    t[np.arange(N), idx] += 1.0 - label_smoothing
    loss = -(t * logp).sum() / N
    return loss, ((np.exp(logp) - t) / N).reshape(shape)

def _num_grad(f, arr, eps):
    """Central differences, perturbing `arr` in place."""
    num = np.zeros_like(arr)
    for i in np.ndindex(*arr.shape):
        old = arr[i]
        arr[i] = old + eps; fp = f()
        arr[i] = old - eps; fm = f()
        arr[i] = old
        num[i] = (fp - fm) / (2 * eps)
    return num


def _rel(a, n):
    """Relative error ||a-n|| / (||a||+||n||). If BOTH gradients are ~0 (e.g. the bias feeding a BatchNorm,
    whose true gradient is exactly 0) the ratio just measures float noise, so judge those on absolute error."""
    d = np.linalg.norm(a - n)
    s = np.linalg.norm(a) + np.linalg.norm(n)
    return d if s < 1e-7 else d / s


def _report(worst, tol, name):
    ok = worst < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<38} max rel err = {worst:.2e}")
    return ok


def gradcheck_layer(layer, x, tol=1e-6, eps=1e-6, name="", seed=0):
    """Checks dx and every Parameter grad of a layer."""
    R = np.random.default_rng(seed).standard_normal(layer(x).shape)
    loss = lambda: float((layer(x) * R).sum())
    layer.zero_grad(); layer(x)
    dx = layer.backward(R)
    worst = 0.0
    if np.issubdtype(x.dtype, np.floating):                 # integer inputs (token ids) have no gradient
        worst = _rel(dx, _num_grad(loss, x, eps))
    for p in layer.parameters():
        worst = max(worst, _rel(p.grad, _num_grad(loss, p.data, eps)))
    return _report(worst, tol, name)


def gradcheck_loss(fn, x, tol=1e-6, eps=1e-6, name=""):
    """fn(x) -> (scalar loss, dx)."""
    worst = _rel(fn(x)[1], _num_grad(lambda: fn(x)[0], x, eps))
    return _report(worst, tol, name)


def gradcheck_model(model, ids, targets, label_smoothing=0.0, tol=1e-6, eps=1e-6, name=""):
    """End-to-end check of a language model: every parameter's gradient vs central differences."""
    def loss():
        return cross_entropy(model(ids), targets, label_smoothing)[0]
    model.zero_grad()
    _, dlogits = cross_entropy(model(ids), targets, label_smoothing)
    model.backward(dlogits)
    worst = max(_rel(p.grad, _num_grad(loss, p.data, eps)) for p in model.parameters())
    return _report(worst, tol, name)


# =====================================================================================
# STAGE 2 -- OPTIMIZERS, LEARNING-RATE SCHEDULE, GRADIENT CLIPPING, EARLY STOPPING
# =====================================================================================
# After backward() every Parameter holds p.grad (which way is "uphill").
# An optimizer's whole job is to nudge p.data the opposite way. Each optimizer below is
# only a few lines of maths, all inside _update().

class Optimizer:
    """Base class: loops over the parameters and calls _update() on each one."""

    def __init__(self, params, lr, l2=0.0):
        self.params = list(params)
        self.lr = lr                 # the learning-rate schedule overwrites this every step
        self.l2 = l2                 # classic L2 strength: added INSIDE the gradient (coupled)
        self.t = 0                   # steps taken so far
        self.state = [{} for _ in self.params]    # each parameter's private memory (velocity, averages, ...)

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        self.t += 1
        for p, st in zip(self.params, self.state):
            g = p.grad
            if self.l2 and p.decay:                 # L2 penalty (l2/2)*w^2 has gradient l2*w
                g = g + self.l2 * p.data
            self._update(p, g, st)


class SGD(Optimizer):
    """Plain gradient descent; momentum=0.9 keeps a running 'velocity' so steps build up speed."""

    def __init__(self, params, lr=0.1, momentum=0.0, l2=0.0):
        super().__init__(params, lr, l2)
        self.momentum = momentum

    def _update(self, p, g, st):
        v = self.momentum * st.get("v", 0.0) + g
        st["v"] = v
        p.data -= self.lr * v


class AdaGrad(Optimizer):
    """Divides each step by the root of ALL past squared gradients (steps only ever shrink)."""

    def __init__(self, params, lr=0.1, eps=1e-8, l2=0.0):
        super().__init__(params, lr, l2)
        self.eps = eps

    def _update(self, p, g, st):
        s = st.get("s", 0.0) + g * g
        st["s"] = s
        p.data -= self.lr * g / (np.sqrt(s) + self.eps)


class RMSProp(Optimizer):
    """Like AdaGrad but with a moving average, so old gradients are slowly forgotten."""

    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, l2=0.0):
        super().__init__(params, lr, l2)
        self.alpha, self.eps = alpha, eps

    def _update(self, p, g, st):
        s = self.alpha * st.get("s", 0.0) + (1 - self.alpha) * g * g
        st["s"] = s
        p.data -= self.lr * g / (np.sqrt(s) + self.eps)


class Adam(Optimizer):
    """Momentum (m) + RMSProp-style scaling (v), with bias correction for the first few steps.
    NOTE: with l2>0 the penalty goes through the same division by sqrt(v) as the data gradient."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, l2=0.0):
        super().__init__(params, lr, l2)
        self.b1, self.b2, self.eps = betas[0], betas[1], eps

    def _update(self, p, g, st):
        m = self.b1 * st.get("m", 0.0) + (1 - self.b1) * g
        v = self.b2 * st.get("v", 0.0) + (1 - self.b2) * g * g
        st["m"], st["v"] = m, v
        m_hat = m / (1 - self.b1 ** self.t)         # bias correction: m and v start at 0
        v_hat = v / (1 - self.b2 ** self.t)
        p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Adam):
    """Adam with DECOUPLED weight decay: the weights shrink by a fixed fraction lr*wd every step,
    completely outside the adaptive machinery."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(params, lr, betas, eps, l2=0.0)
        self.weight_decay = weight_decay

    def _update(self, p, g, st):
        if p.decay:
            p.data *= (1.0 - self.lr * self.weight_decay)
        super()._update(p, g, st)


def make_optimizer(name, params, lr, penalty=0.0, momentum=0.9):
    """One switch for the experiments. `penalty` is a weight penalty whose meaning depends on the optimizer:
    coupled L2 for sgd/adagrad/rmsprop/adam, decoupled weight decay for adamw."""
    if name == "sgd":
        return SGD(params, lr, momentum=momentum, l2=penalty)
    if name == "adagrad":
        return AdaGrad(params, lr, l2=penalty)
    if name == "rmsprop":
        return RMSProp(params, lr, l2=penalty)
    if name == "adam":
        return Adam(params, lr, l2=penalty)
    if name == "adamw":
        return AdamW(params, lr, weight_decay=penalty)
    raise ValueError("unknown optimizer: " + name)


def warmup_cosine(step, total_steps, base_lr, warmup_steps=0, min_lr=0.0):
    """Learning rate at `step`: ramp up linearly during warmup, then glide down along a cosine."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + np.cos(np.pi * progress))


def clip_grad_norm(params, max_norm):
    """If the total gradient length exceeds max_norm, shrink ALL gradients by the same factor
    (the direction is kept). Returns the length BEFORE clipping, which is what we plot."""
    total = float(np.sqrt(sum((p.grad ** 2).sum() for p in params)))
    if max_norm is not None and total > max_norm:
        scale = max_norm / (total + 1e-12)
        for p in params:
            p.grad *= scale
    return total


class EarlyStopping:
    """Watches validation loss; remembers the best weights; says 'stop' after `patience` checks without improvement."""

    def __init__(self, patience=5):
        self.patience, self.best, self.best_step, self.bad, self.best_weights = patience, np.inf, None, 0, None

    def update(self, val_loss, step, params):
        if val_loss < self.best:
            self.best, self.best_step, self.bad = val_loss, step, 0
            self.best_weights = [p.data.copy() for p in params]
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, params):
        for p, w in zip(params, self.best_weights):
            p.data[...] = w


# =====================================================================================
# STAGE 3 -- RECURRENT LAYERS (Embedding, RNN, LSTM) AND THE CHARACTER LANGUAGE MODEL
# =====================================================================================
# A recurrent layer reads a sequence one step at a time, carrying a hidden state h along.
# Backward = "backprop through time" (BPTT): walk the steps in reverse, handing the gradient
# of h(t) back to h(t-1). The same weights are used at EVERY step, so every step ADDS to the
# weight gradients (this is why backward() always uses += ).

class Embedding(Module):
    """Turns integer character ids (B,T) into vectors (B,T,E) by looking up rows of a table."""

    def __init__(self, V, E, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = Parameter(rng.standard_normal((V, E)) * 0.1)

    def forward(self, ids):
        self._ids = ids
        return self.W.data[ids]

    def backward(self, dy):
        np.add.at(self.W.grad, self._ids, dy)    # one character can appear many times -> ADD each time
        return None                              # ids are integers: nothing to pass back


def _orthogonal(n, rng, gain=1.0):
    """A random rotation matrix (times gain). Rotations keep vector lengths, so gain~1 is a stable start."""
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return gain * q * np.sign(np.diag(r))


class _Recurrent(Module):
    """Shared plumbing for RNN and LSTM: the weights and the RECURRENT-dropout masks.

    Recurrent dropout randomly zeroes entries of h(t-1) before it re-enters the cell.
      per_step_masks=False  ->  VARIATIONAL dropout: one mask per sequence, reused at every time step. (correct)
      per_step_masks=True   ->  a fresh mask at every step: the same unit is dropped at t but kept at t+1,
                                so information stored in a unit gets destroyed over and over (the bad way).
    """

    def __init__(self, nin, nh, n_gates, p_rec, per_step_masks, rng, mask_rng=None):
        self.nin, self.nh = nin, nh
        self.p_rec, self.per_step_masks = p_rec, per_step_masks
        self.rng = mask_rng or rng                  # randomness for the dropout masks
        self._fixed = None
        self.Wx = Parameter(rng.standard_normal((nin, n_gates * nh)) / np.sqrt(nin))
        self.b = Parameter(np.zeros(n_gates * nh), decay=False)

    def sample_masks(self, B, T):
        if self.p_rec == 0.0:
            return None
        shape = (T, B, self.nh) if self.per_step_masks else (B, self.nh)
        return (self.rng.random(shape) >= self.p_rec) / (1.0 - self.p_rec)

    def freeze(self, B, T):
        """Pin the masks so repeated forward() calls match exactly (needed for gradient checks)."""
        self._fixed = self.sample_masks(B, T)
        return self

    def unfreeze(self):
        self._fixed = None
        return self

    def _get_masks(self, B, T):
        if not self.training:
            return None
        return self._fixed if self._fixed is not None else self.sample_masks(B, T)

    @staticmethod
    def _mask_at(masks, t):
        return masks[t] if masks.ndim == 3 else masks     # (T,B,H) per-step, or (B,H) shared


class RNN(_Recurrent):
    """Vanilla RNN:  h(t) = tanh( x(t) Wx + (h(t-1) * mask) Wh + b ).
    `gain` scales the recurrent matrix Wh: gain > 1 makes gradients EXPLODE going back in time."""

    def __init__(self, nin, nh, p_rec=0.0, per_step_masks=False, gain=1.0, rng=None, mask_rng=None):
        rng = rng or np.random.default_rng(0)
        super().__init__(nin, nh, 1, p_rec, per_step_masks, rng, mask_rng)
        self.Wh = Parameter(_orthogonal(nh, rng, gain))

    def forward(self, x):                                  # x: (B,T,nin) -> h: (B,T,nh)
        B, T, _ = x.shape
        H = self.nh
        masks = self._get_masks(B, T)
        XW = x @ self.Wx.data + self.b.data                # the input part of every step, in one matmul
        h = np.zeros((B, H))
        out = np.empty((B, T, H))
        HD = np.empty((B, T, H))                           # the (dropped) h(t-1) used at each step
        for t in range(T):
            hd = h if masks is None else h * self._mask_at(masks, t)
            h = np.tanh(XW[:, t] + hd @ self.Wh.data)
            HD[:, t], out[:, t] = hd, h
        self._x, self._out, self._HD, self._masks = x, out, HD, masks
        return out

    def backward(self, dout):                              # dout: gradient w.r.t. every h(t)
        B, T, H = dout.shape
        out, masks = self._out, self._masks
        dA = np.empty((B, T, H))                           # gradient w.r.t. the pre-activation at each step
        dh_next = np.zeros((B, H))                         # gradient arriving from step t+1
        for t in reversed(range(T)):
            da = (dout[:, t] + dh_next) * (1.0 - out[:, t] ** 2)     # through tanh
            dA[:, t] = da
            dh_next = da @ self.Wh.data.T                  # back into h(t-1) ...
            if masks is not None:
                dh_next = dh_next * self._mask_at(masks, t)          # ... through the same dropout mask
        self.Wh.grad += self._HD.reshape(-1, H).T @ dA.reshape(-1, H)
        self.Wx.grad += self._x.reshape(-1, self.nin).T @ dA.reshape(-1, H)
        self.b.grad += dA.sum(axis=(0, 1))
        return dA @ self.Wx.data.T


class LSTM(_Recurrent):
    """LSTM: a cell state c(t) with three gates that decide what to forget, write and show.
         i, f, o = sigmoid(...), g = tanh(...)      c(t) = f*c(t-1) + i*g      h(t) = o*tanh(c(t))
    The forget bias starts at 1 so the cell remembers by default."""

    def __init__(self, nin, nh, p_rec=0.0, per_step_masks=False, rng=None, mask_rng=None):
        rng = rng or np.random.default_rng(0)
        super().__init__(nin, nh, 4, p_rec, per_step_masks, rng, mask_rng)
        self.Wh = Parameter(np.hstack([_orthogonal(nh, rng) for _ in range(4)]))
        self.b.data[nh:2 * nh] = 1.0

    def forward(self, x):
        B, T, _ = x.shape
        H = self.nh
        masks = self._get_masks(B, T)
        XW = x @ self.Wx.data + self.b.data                # (B,T,4H)
        h, c = np.zeros((B, H)), np.zeros((B, H))
        out = np.empty((B, T, H))
        HD = np.empty((B, T, H))
        G = np.empty((B, T, 4 * H))                        # gate activations: [i | f | g | o]
        C = np.empty((B, T, H))                            # cell states
        TC = np.empty((B, T, H))                           # tanh(c)
        for t in range(T):
            hd = h if masks is None else h * self._mask_at(masks, t)
            z = XW[:, t] + hd @ self.Wh.data
            i, f = 1 / (1 + np.exp(-z[:, :H])), 1 / (1 + np.exp(-z[:, H:2 * H]))
            g, o = np.tanh(z[:, 2 * H:3 * H]), 1 / (1 + np.exp(-z[:, 3 * H:]))
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            HD[:, t], out[:, t], C[:, t], TC[:, t] = hd, h, c, tc
            G[:, t, :H], G[:, t, H:2 * H], G[:, t, 2 * H:3 * H], G[:, t, 3 * H:] = i, f, g, o
        self._x, self._HD, self._G, self._C, self._TC, self._masks = x, HD, G, C, TC, masks
        return out

    def backward(self, dout):
        B, T, H = dout.shape
        G, C, TC, masks = self._G, self._C, self._TC, self._masks
        dZ = np.empty((B, T, 4 * H))
        dh_next, dc_next = np.zeros((B, H)), np.zeros((B, H))
        for t in reversed(range(T)):
            i, f, g, o = G[:, t, :H], G[:, t, H:2 * H], G[:, t, 2 * H:3 * H], G[:, t, 3 * H:]
            c_prev = C[:, t - 1] if t > 0 else np.zeros((B, H))
            dh = dout[:, t] + dh_next
            do = dh * TC[:, t]
            dc = dh * o * (1.0 - TC[:, t] ** 2) + dc_next      # c(t) feeds h(t) AND c(t+1)
            di, dg, df = dc * g, dc * i, dc * c_prev
            dc_next = dc * f                                    # the gradient highway through time
            dZ[:, t, :H], dZ[:, t, H:2 * H] = di * i * (1 - i), df * f * (1 - f)
            dZ[:, t, 2 * H:3 * H], dZ[:, t, 3 * H:] = dg * (1 - g ** 2), do * o * (1 - o)
            dh_next = dZ[:, t] @ self.Wh.data.T
            if masks is not None:
                dh_next = dh_next * self._mask_at(masks, t)
        self.Wh.grad += self._HD.reshape(-1, H).T @ dZ.reshape(-1, 4 * H)
        self.Wx.grad += self._x.reshape(-1, self.nin).T @ dZ.reshape(-1, 4 * H)
        self.b.grad += dZ.sum(axis=(0, 1))
        return dZ @ self.Wx.data.T


class CharLM(Module):
    """Character language model:  ids -> Embedding -> Dropout -> RNN/LSTM -> Dropout -> Linear -> logits.
    p_drop : ordinary (inverted) dropout on the non-recurrent connections (fresh mask everywhere).
    p_rec  : recurrent dropout on h(t-1) inside the RNN/LSTM (see _Recurrent)."""

    def __init__(self, V, E=16, H=64, cell="lstm", p_drop=0.0, p_rec=0.0, per_step_masks=False,
                 rnn_gain=1.0, rng=None, mask_rng=None):
        rng = rng or np.random.default_rng(0)
        mask_rng = mask_rng or np.random.default_rng(1)        # dropout randomness kept apart from the init
        self.emb = Embedding(V, E, rng)
        self.drop_in = Dropout(p_drop, rng=mask_rng)
        if cell == "lstm":
            self.rnn = LSTM(E, H, p_rec, per_step_masks, rng=rng, mask_rng=mask_rng)
        else:
            self.rnn = RNN(E, H, p_rec, per_step_masks, rnn_gain, rng=rng, mask_rng=mask_rng)
        self.drop_out = Dropout(p_drop, rng=mask_rng)
        self.head = Linear(H, V, rng=rng, init="xavier")
        self.E, self.H = E, H

    def forward(self, ids):                                    # ids: (B,T) ints -> logits (B,T,V)
        x = self.drop_in(self.emb(ids))
        return self.head(self.drop_out(self.rnn(x)))

    def backward(self, dlogits):
        d = self.drop_out.backward(self.head.backward(dlogits))
        self.emb.backward(self.drop_in.backward(self.rnn.backward(d)))

    def freeze(self, B, T):
        """Pin every dropout mask (for gradient checks)."""
        self.drop_in.freeze((B, T, self.E))
        self.drop_out.freeze((B, T, self.H))
        self.rnn.freeze(B, T)
        return self

    def unfreeze(self):
        self.drop_in.unfreeze(); self.drop_out.unfreeze(); self.rnn.unfreeze()
        return self


def count_params(model):
    return int(sum(p.data.size for p in model.parameters()))


# =====================================================================================
# STAGE 4 -- TRAINING LOOP, EVALUATION, OVERFITTING MEASUREMENT
# =====================================================================================

def get_batch(ids, B, T, rng):
    """B random windows of the text. x = ids[i:i+T], y = the same window shifted by one (the next characters)."""
    i = rng.integers(0, len(ids) - T - 1, size=B)
    x = np.stack([ids[j:j + T] for j in i])
    y = np.stack([ids[j + 1:j + T + 1] for j in i])
    return x, y


def eval_windows(ids, T, max_windows=None):
    """A FIXED set of non-overlapping windows, so train/val loss are measured on the same text every time."""
    n = (len(ids) - 1) // T
    starts = np.arange(n) * T
    if max_windows is not None and max_windows < n:
        starts = starts[np.linspace(0, n - 1, max_windows).astype(int)]
    x = np.stack([ids[s:s + T] for s in starts])
    y = np.stack([ids[s + 1:s + T + 1] for s in starts])
    return x, y


def evaluate(model, x, y, B=128):
    """Average cross-entropy in nats per character, with dropout OFF and no label smoothing."""
    was_training = model.training
    model.eval()
    total = 0.0
    for s in range(0, len(x), B):
        loss, _ = cross_entropy(model(x[s:s + B]), y[s:s + B])
        total += loss * len(x[s:s + B])
    model.train(was_training)
    return total / len(x)


def default_cfg():
    """Every knob of an experiment in one place. Override only what you want to change."""
    return dict(
        cell="lstm", E=16, H=64, rnn_gain=1.0,           # model
        p_drop=0.0, p_rec=0.0, per_step_masks=False,      # dropout / recurrent dropout
        optimizer="adam", lr=5e-3, momentum=0.9,          # optimizer
        penalty=0.0,                                      # L2 (sgd/rmsprop/adam) or decoupled decay (adamw)
        label_smoothing=0.0,
        clip=1.0,                                         # max gradient norm, None = no clipping
        schedule=False, warmup=50,                        # warmup + cosine decay
        iters=600, B=64, T=32,                            # training length / batch shape
        eval_every=25, eval_windows=300,                  # how often / how much to evaluate
        early_stop_patience=None,
        seed=0, log_gnorm=False)


def train_lm(cfg, train_ids, val_ids, V, verbose=False):
    """Train one character model and return its loss curves plus summary numbers (incl. the overfitting gap)."""
    import time
    cfg = {**default_cfg(), **cfg}
    seed = cfg["seed"]
    model = CharLM(V, cfg["E"], cfg["H"], cfg["cell"], cfg["p_drop"], cfg["p_rec"], cfg["per_step_masks"],
                   cfg["rnn_gain"], rng=np.random.default_rng(seed), mask_rng=np.random.default_rng(seed + 2000))
    params = model.parameters()
    opt = make_optimizer(cfg["optimizer"], params, cfg["lr"], cfg["penalty"], cfg["momentum"])
    data_rng = np.random.default_rng(seed + 1000)          # same batches for every config with this seed
    tx, ty = eval_windows(train_ids, cfg["T"], cfg["eval_windows"])
    vx, vy = eval_windows(val_ids, cfg["T"], cfg["eval_windows"])
    stopper = EarlyStopping(cfg["early_stop_patience"]) if cfg["early_stop_patience"] else None

    hist = {"it": [], "train": [], "val": []}
    gnorms, diverged, t0 = [], False, time.time()
    with np.errstate(all="ignore"):
        for it in range(1, cfg["iters"] + 1):
            opt.lr = (warmup_cosine(it - 1, cfg["iters"], cfg["lr"], cfg["warmup"]) if cfg["schedule"] else cfg["lr"])
            x, y = get_batch(train_ids, cfg["B"], cfg["T"], data_rng)
            model.zero_grad()
            loss, dlogits = cross_entropy(model(x), y, cfg["label_smoothing"])
            model.backward(dlogits)
            g = clip_grad_norm(params, cfg["clip"])
            gnorms.append(g)
            if not (np.isfinite(loss) and np.isfinite(g)):
                diverged = True
                break
            opt.step()
            if it % cfg["eval_every"] == 0 or it == cfg["iters"]:
                tr, va = evaluate(model, tx, ty), evaluate(model, vx, vy)
                hist["it"].append(it); hist["train"].append(float(tr)); hist["val"].append(float(va))
                if verbose:
                    print(f"  it {it:>5}  train {tr:.3f}  val {va:.3f}  gap {va - tr:+.3f}  |g| {g:.2f}")
                if not np.isfinite(va):
                    diverged = True
                    break
                if stopper and stopper.update(va, it, params):
                    break
    gn = np.array(gnorms)
    out = dict(cfg=cfg, hist=hist, diverged=diverged, seconds=time.time() - t0, n_params=count_params(model),
               gnorm_max=float(np.nanmax(gn)) if len(gn) else float("nan"),
               gnorm_median=float(np.nanmedian(gn)) if len(gn) else float("nan"))
    if hist["it"]:
        v, tr = np.array(hist["val"]), np.array(hist["train"])
        k = int(np.nanargmin(v))
        out.update(final_train=float(tr[-1]), final_val=float(v[-1]), gap=float(v[-1] - tr[-1]),
                   best_val=float(v[k]), best_iter=int(hist["it"][k]), gap_at_best=float(v[k] - tr[k]))
    else:
        out.update(final_train=float("nan"), final_val=float("nan"), gap=float("nan"),
                   best_val=float("nan"), best_iter=0, gap_at_best=float("nan"))
    if cfg["log_gnorm"]:
        out["gnorms"] = [float(v) for v in gn]
    return out


def run_grid(cfgs, train_ids, val_ids, V, save_path=None):
    """Run many configs one after another. If save_path is given, results are saved after every run and a
    re-run SKIPS configs that are already finished (so you can stop and resume a long grid)."""
    import json, os, time
    results, done = [], {}
    if save_path and os.path.exists(save_path):
        results = json.load(open(save_path))
        done = {json.dumps(r["cfg"], sort_keys=True): r for r in results}
    t0 = time.time()
    for k, cfg in enumerate(cfgs, 1):
        full = {**default_cfg(), **cfg}
        key = json.dumps(full, sort_keys=True)
        if key in done:
            continue
        r = train_lm(cfg, train_ids, val_ids, V)
        results.append(r); done[key] = r
        if save_path:
            json.dump(results, open(save_path, "w"))
        print(f"[{k}/{len(cfgs)}] {r['cfg']['cell']:>4} {r['cfg']['optimizer']:>7} seed {r['cfg']['seed']}  "
              f"train {r['final_train']:.3f} val {r['final_val']:.3f} gap {r['gap']:+.3f}"
              f"{'  DIVERGED' if r['diverged'] else ''}   ({time.time() - t0:.0f}s elapsed)")
    return [done[json.dumps({**default_cfg(), **c}, sort_keys=True)] for c in cfgs]


# =====================================================================================
# STAGE 5 -- A TINY PLOTTER (NumPy only: it writes SVG text; Jupyter shows it automatically)
# =====================================================================================

_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]


class Figure:
    """An SVG image. Put it on the last line of a notebook cell to see it; fig.save('x.svg') writes the file."""

    def __init__(self, svg):
        self.svg = svg

    def _repr_svg_(self):
        return self.svg

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.svg)
        return path


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _nice_ticks(lo, hi, n=5):
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10.0 ** np.floor(np.log10(raw))
    step = min(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return np.arange(np.floor(lo / step) * step, hi + step * 0.5, step)


def _fmt(v):
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:.2g}"


def _frame(title, xlabel, ylabel, xt, yt, W, H, L, R, T, Bm, sx, sy, legend):
    """Axes, grid, tick labels, title and legend shared by all chart types."""
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L}" y="22" font-size="15" font-weight="bold" fill="#222">{_esc(title)}</text>']
    for v in yt:
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#e5e5e5"/>'
                 f'<text x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end" fill="#444">{_fmt(v)}</text>')
    for v in xt:
        s.append(f'<text x="{sx(v):.1f}" y="{H - Bm + 16}" text-anchor="middle" fill="#444">{_fmt(v)}</text>')
    s.append(f'<rect x="{L}" y="{T}" width="{W - R - L}" height="{H - Bm - T}" fill="none" stroke="#999"/>')
    s.append(f'<text x="{(L + W - R) / 2}" y="{H - 10}" text-anchor="middle" fill="#222">{_esc(xlabel)}</text>')
    s.append(f'<text transform="translate(16,{(T + H - Bm) / 2}) rotate(-90)" text-anchor="middle" fill="#222">{_esc(ylabel)}</text>')
    for k, (label, color, dashed) in enumerate(legend):
        y = T + 14 + 18 * k
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        s.append(f'<line x1="{W - R + 12}" x2="{W - R + 36}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"{dash}/>'
                 f'<text x="{W - R + 42}" y="{y + 4}" fill="#222">{_esc(label)}</text>')
    return s


def line_chart(series, title="", xlabel="", ylabel="", width=760, height=400, ylim=None):
    """series = list of dicts: {label, x, y, dashed=False, color=None, lo=None, hi=None}.
    lo/hi (optional) draw a shaded band, e.g. min..max over seeds."""
    L, R, T, Bm = 70, 200, 36, 52
    xs = np.concatenate([np.asarray(s["x"], float) for s in series])
    ys = np.concatenate([np.asarray(v, float) for s in series for v in (s["y"], s.get("lo"), s.get("hi")) if v is not None])
    ys = ys[np.isfinite(ys)]
    ylo, yhi = (ylim if ylim else (ys.min(), ys.max()))
    pad = 0.04 * (yhi - ylo + 1e-9)
    ylo, yhi = (ylo, yhi) if ylim else (ylo - pad, yhi + pad)
    xt, yt = _nice_ticks(xs.min(), xs.max()), _nice_ticks(ylo, yhi)
    xlo, xhi = xs.min(), xs.max()
    sx = lambda v: L + (v - xlo) / (xhi - xlo + 1e-12) * (width - R - L)
    sy = lambda v: height - Bm - (v - ylo) / (yhi - ylo + 1e-12) * (height - Bm - T)
    xt = xt[(xt >= xlo - 1e-9) & (xt <= xhi + 1e-9)]
    yt = yt[(yt >= ylo - 1e-9) & (yt <= yhi + 1e-9)]
    legend = []
    body = []
    for k, s in enumerate(series):
        color = s.get("color") or _COLORS[k % len(_COLORS)]
        dashed = s.get("dashed", False)
        x, y = np.asarray(s["x"], float), np.asarray(s["y"], float)
        ok = np.isfinite(y)
        if s.get("lo") is not None and s.get("hi") is not None:
            lo, hi = np.asarray(s["lo"], float), np.asarray(s["hi"], float)
            pts = [f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, hi)] + [f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x[::-1], lo[::-1])]
            body.append(f'<polygon points="{" ".join(pts)}" fill="{color}" opacity="0.15"/>')
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x[ok], y[ok]))
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        body.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>')
        legend.append((s["label"], color, dashed))
    svg = _frame(title, xlabel, ylabel, xt, yt, width, height, L, R, T, Bm, sx, sy, legend)
    return Figure("\n".join(svg + body + ["</svg>"]))


def bar_chart(categories, series, title="", ylabel="", width=760, height=400):
    """Grouped bars. series = list of dicts {label, values, err=None, color=None}; err draws +/- whiskers."""
    L, R, T, Bm = 70, 200, 36, 60
    vals = np.array([s["values"] for s in series], float)
    errs = np.array([s.get("err") if s.get("err") is not None else np.zeros(len(categories)) for s in series], float)
    lo, hi = min(0.0, np.nanmin(vals - errs)), max(0.0, np.nanmax(vals + errs))
    pad = 0.06 * (hi - lo + 1e-9)
    lo, hi = (lo - pad if lo < 0 else 0.0), hi + pad
    yt = _nice_ticks(lo, hi)
    yt = yt[(yt >= lo - 1e-9) & (yt <= hi + 1e-9)]
    sy = lambda v: height - Bm - (v - lo) / (hi - lo + 1e-12) * (height - Bm - T)
    sx = lambda v: L
    legend = [(s["label"], s.get("color") or _COLORS[k % len(_COLORS)], False) for k, s in enumerate(series)]
    svg = _frame(title, "", ylabel, [], yt, width, height, L, R, T, Bm, sx, sy, legend)
    gw = (width - R - L) / len(categories)
    bw = gw * 0.8 / len(series)
    for c, name in enumerate(categories):
        x0 = L + c * gw + gw * 0.1
        svg.append(f'<text x="{L + c * gw + gw / 2:.1f}" y="{height - Bm + 16}" text-anchor="middle" fill="#222">{_esc(name)}</text>')
        for k, s in enumerate(series):
            color = s.get("color") or _COLORS[k % len(_COLORS)]
            v, e = vals[k, c], errs[k, c]
            if not np.isfinite(v):
                continue
            x = x0 + k * bw
            svg.append(f'<rect x="{x:.1f}" y="{min(sy(v), sy(0)):.1f}" width="{bw * 0.92:.1f}" height="{abs(sy(v) - sy(0)):.1f}" fill="{color}"/>')
            if e > 0:
                cx = x + bw * 0.46
                svg.append(f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{sy(v - e):.1f}" y2="{sy(v + e):.1f}" stroke="#222"/>'
                           f'<line x1="{cx - 3:.1f}" x2="{cx + 3:.1f}" y1="{sy(v + e):.1f}" y2="{sy(v + e):.1f}" stroke="#222"/>'
                           f'<line x1="{cx - 3:.1f}" x2="{cx + 3:.1f}" y1="{sy(v - e):.1f}" y2="{sy(v - e):.1f}" stroke="#222"/>')
    svg.append(f'<line x1="{L}" x2="{width - R}" y1="{sy(0):.1f}" y2="{sy(0):.1f}" stroke="#666"/>')
    return Figure("\n".join(svg + ["</svg>"]))





# ---- small helpers for summarising many runs --------------------------------------------------

def select(results, **where):
    """Keep only the runs whose config matches, e.g. select(results, cell="lstm", optimizer="adam")."""
    return [r for r in results if all(r["cfg"].get(k) == v for k, v in where.items())]


def mean_std(runs, key):
    """Mean and standard deviation of a summary number (e.g. 'gap', 'final_val') across runs (= seeds)."""
    vals = np.array([r[key] for r in runs], dtype=float)
    return float(np.mean(vals)), float(np.std(vals))


def mean_curve(runs, key):
    """Average a loss curve over seeds. Returns (iterations, mean, lowest, highest). key: 'train', 'val' or 'gap'."""
    n = min(len(r["hist"]["it"]) for r in runs)
    its = np.array(runs[0]["hist"]["it"][:n])
    if key == "gap":
        curves = np.array([np.array(r["hist"]["val"][:n]) - np.array(r["hist"]["train"][:n]) for r in runs])
    else:
        curves = np.array([r["hist"][key][:n] for r in runs])
    return its, curves.mean(0), curves.min(0), curves.max(0)


def run_tests():
    """Gradient-check every layer. Lines marked NEGATIVE CONTROL must FAIL: they plant a known-wrong
    backward to prove the checker can catch it (including the subtle BatchNorm mistakes)."""
    rng = np.random.default_rng(0)
    results, caught = [], []

    def check(ok, name):                     # for non-gradient tests (statistics, invariances)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        results.append(ok)

    print("Layers & loss")
    results.append(gradcheck_layer(Linear(5, 4, rng=rng), rng.standard_normal((6, 5)), name="Linear (B,n)"))
    results.append(gradcheck_layer(Linear(5, 4, rng=rng), rng.standard_normal((3, 7, 5)), name="Linear (B,T,n)"))
    results.append(gradcheck_layer(Linear(5, 4, bias=False, rng=rng), rng.standard_normal((6, 5)), name="Linear no bias"))
    results.append(gradcheck_layer(ReLU(), rng.standard_normal((5, 6)), name="ReLU"))
    results.append(gradcheck_layer(Tanh(), rng.standard_normal((5, 6)), name="Tanh"))
    results.append(gradcheck_layer(Sequential(Linear(5, 7, rng=rng), Tanh(), Linear(7, 4, rng=rng), ReLU(), Linear(4, 3, rng=rng)),
                                   rng.standard_normal((6, 5)), name="Sequential MLP (all params)"))
    t, t3 = rng.integers(0, 4, 8), rng.integers(0, 4, (3, 5))
    results.append(gradcheck_loss(lambda z: cross_entropy(z, t), rng.standard_normal((8, 4)), name="softmax-CE"))
    results.append(gradcheck_loss(lambda z: cross_entropy(z, t, 0.1), rng.standard_normal((8, 4)), name="softmax-CE + label smoothing"))
    results.append(gradcheck_loss(lambda z: cross_entropy(z, t3, 0.1), rng.standard_normal((3, 5, 4)), name="softmax-CE (B,T,V) logits"))

    # backward must accumulate: two passes without zero_grad double the gradient (BPTT relies on this)
    l = Linear(3, 2, rng=rng); x = rng.standard_normal((4, 3)); dy = rng.standard_normal((4, 2))
    l.zero_grad(); l(x); l.backward(dy); g1 = l.W.grad.copy(); l(x); l.backward(dy)
    check(np.allclose(l.W.grad, 2 * g1), "backward accumulates (+=)")

    print("\nDropout")
    results.append(gradcheck_layer(Dropout(0.3, rng=rng).freeze((6, 5)), rng.standard_normal((6, 5)), name="Dropout (frozen mask) dx"))
    d = Dropout(0.3, rng=rng); big = np.ones(200_000); y = d(big)
    check(abs((y == 0).mean() - 0.3) < 0.01, f"drops ~p of units (measured {(y == 0).mean():.3f}, p=0.3)")
    check(abs(y.mean() - 1.0) < 0.01, f"inverted scaling keeps E[y]=E[x] (measured {y.mean():.3f})")
    xx = rng.standard_normal((4, 6)); d.train(); y1 = d(xx); g = d.backward(np.ones_like(xx))
    check(np.allclose(g, y1 / xx), "backward reuses the forward mask")
    d.eval(); check(np.array_equal(d(xx), xx) and np.array_equal(d.backward(xx), xx), "eval mode is the identity")
    m = d.sample_mask((4, 6)); d.train()
    check(np.array_equal(d(xx, mask=m), xx * m) and np.array_equal(d(xx * 2, mask=m), xx * 2 * m),
          "external mask reused across calls (variational-dropout hook)")

    print("\nBatchNorm")
    results.append(gradcheck_layer(BatchNorm(5), rng.standard_normal((8, 5)) * 3 + 2, name="BatchNorm train (N,C): dx, dgamma, dbeta"))
    bn = BatchNorm(4); bn.gamma.data = rng.uniform(0.5, 2, 4); bn.beta.data = rng.standard_normal(4)
    results.append(gradcheck_layer(bn, rng.standard_normal((3, 6, 4)) * 2 - 1, name="BatchNorm train (B,T,C), random gamma/beta"))
    bn = BatchNorm(4); bn.gamma.data = rng.uniform(0.5, 2, 4); bn.beta.data = rng.standard_normal(4)
    for _ in range(30): bn(rng.standard_normal((32, 4)) * 2 + 1)         # warm up running stats
    bn.eval()
    results.append(gradcheck_layer(bn, rng.standard_normal((6, 4)), name="BatchNorm eval (running stats)"))

    bn = BatchNorm(3); xb = rng.standard_normal((512, 3)) * 5 + 3; yb = bn(xb)
    check(np.abs(yb.mean(0)).max() < 1e-9 and np.abs(yb.var(0) - 1).max() < 1e-3,
          f"train output has mean 0, var 1 per feature (var err {np.abs(yb.var(0) - 1).max():.1e})")
    bn = BatchNorm(3)
    for _ in range(200): bn(rng.standard_normal((256, 3)) * 5 + 3)
    check(np.abs(bn.running_mean - 3).max() < 0.5 and np.abs(bn.running_var - 25).max() < 3,
          f"running stats converge to the data (mean~{bn.running_mean.mean():.2f} vs 3, var~{bn.running_var.mean():.1f} vs 25)")
    xs = rng.standard_normal((64, 6)); W0 = rng.standard_normal((6, 5))
    o1 = BatchNorm(5)(xs @ W0); o2 = BatchNorm(5)(xs @ (7.0 * W0))
    check(np.abs(o1 - o2).max() < 1e-3,
          f"scale invariance: BN(x @ 7W) == BN(x @ W) (max diff {np.abs(o1 - o2).max():.1e})   <- why weight decay acts through the effective LR")

    print("\nIntegration")
    net = Sequential(Linear(5, 7, rng=rng), BatchNorm(7), Tanh(), Dropout(0.25, rng=rng).freeze((6, 7)), Linear(7, 3, rng=rng))
    results.append(gradcheck_layer(net, rng.standard_normal((6, 5)), name="Linear-BN-Tanh-Dropout-Linear (all params)"))
    # BN subtracts the batch mean, so a bias on the layer BEFORE it cancels out: its true gradient is exactly 0
    net.zero_grad(); net(rng.standard_normal((6, 5))); net.backward(rng.standard_normal((6, 3)))
    check(np.abs(net.layers[0].b.grad).max() < 1e-12, f"bias before BN gets zero gradient (max |db| = {np.abs(net.layers[0].b.grad).max():.1e}): redundant, drop it")

    print("\nOptimizers, schedule, clipping, early stopping")
    g0, p0 = np.array([0.5, 0.25]), np.array([1.0, -2.0])

    def one_step(make, steps=1, g=(0.5, 0.25), start=(1.0, -2.0), decay=True):
        p = Parameter(np.array(start)); p.decay = decay
        opt = make([p])
        for _ in range(steps):
            p.grad = np.array(g, dtype=float); opt.step()
        return p.data

    check(np.allclose(one_step(lambda ps: SGD(ps, lr=0.1)), p0 - 0.1 * g0), "SGD: p - lr*g")
    check(np.allclose(one_step(lambda ps: SGD(ps, lr=0.1, momentum=0.9), steps=2), p0 - 0.1 * 2.9 * g0), "SGD momentum: v = 0.9*v + g")
    check(np.allclose(one_step(lambda ps: SGD(ps, lr=0.1, l2=0.2)), p0 - 0.1 * (g0 + 0.2 * p0)), "SGD + L2: gradient gets + l2*w")
    check(np.allclose(one_step(lambda ps: AdaGrad(ps, lr=0.1, eps=0)), p0 - 0.1 * np.sign(g0)), "AdaGrad: first step = lr*sign(g)")
    check(np.allclose(one_step(lambda ps: RMSProp(ps, lr=0.01, alpha=0.9, eps=0)), p0 - 0.01 / np.sqrt(0.1) * np.sign(g0)),
          "RMSProp: first step = lr/sqrt(1-alpha)*sign(g)")
    check(np.allclose(one_step(lambda ps: Adam(ps, lr=0.1, eps=0)), p0 - 0.1 * np.sign(g0)), "Adam: first step = lr*sign(g) (bias correction)")
    check(np.allclose(one_step(lambda ps: AdamW(ps, lr=0.1, eps=0, weight_decay=0.5)), p0 * (1 - 0.05) - 0.1 * np.sign(g0)),
          "AdamW: shrink p by (1 - lr*wd), then the Adam step")

    zero = (0.0, 0.0)
    shrink = lambda make: 1.0 - one_step(make, g=zero, start=(1.0, 1.0))[0]       # how much a weight shrinks in ONE step
    a_small, a_big = shrink(lambda ps: Adam(ps, lr=0.01, l2=0.001)), shrink(lambda ps: Adam(ps, lr=0.01, l2=1.0))
    w_small, w_big = shrink(lambda ps: AdamW(ps, lr=0.01, weight_decay=0.001)), shrink(lambda ps: AdamW(ps, lr=0.01, weight_decay=1.0))
    check(abs(a_small - a_big) < 1e-3 and abs(a_small - 0.01) < 1e-3,
          f"Adam+L2: shrink per step is ~lr whatever l2 is ({a_small:.4f} vs {a_big:.4f}): Adam rescales the penalty away")
    check(abs(w_big / w_small - 1000) < 1, f"AdamW: shrink per step scales with weight_decay ({w_small:.0e} vs {w_big:.0e})")
    check(np.allclose(one_step(lambda ps: AdamW(ps, lr=0.1, weight_decay=0.5), g=zero, decay=False), p0) and
          np.allclose(one_step(lambda ps: SGD(ps, lr=0.1, l2=0.5), g=zero, decay=False), p0),
          "parameters marked decay=False (biases, BN gamma/beta) are never penalised")

    A = np.diag([1.0, 10.0])                                  # a bowl 10x steeper in one direction
    start_loss = 0.5 * np.array([3.0, 3.0]) @ A @ np.array([3.0, 3.0])

    def minimise(make, steps=300):
        p = Parameter(np.array([3.0, 3.0])); opt = make([p])
        for _ in range(steps):
            p.grad = A @ p.data; opt.step()
        return 0.5 * p.data @ A @ p.data

    for name, make in [("SGD+momentum", lambda ps: SGD(ps, 0.02, 0.9)), ("AdaGrad", lambda ps: AdaGrad(ps, 1.0)),
                       ("RMSProp", lambda ps: RMSProp(ps, 0.02)), ("Adam", lambda ps: Adam(ps, 0.05)),
                       ("AdamW", lambda ps: AdamW(ps, 0.05, weight_decay=0.0))]:
        final = minimise(make)
        check(final < 0.01 * start_loss, f"{name} minimises a bowl (loss {start_loss:.0f} -> {final:.1e})")

    lrs = [warmup_cosine(s, 100, 1.0, warmup_steps=10) for s in range(100)]
    check(abs(lrs[0] - 0.1) < 1e-9 and abs(lrs[10] - 1.0) < 1e-9 and lrs[-1] < 0.01 and all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:])),
          "warmup_cosine: ramps up, peaks at the base lr, then decays smoothly")
    ps = [Parameter(np.zeros(3)), Parameter(np.zeros(4))]
    ps[0].grad, ps[1].grad = np.array([3.0, 0, 0]), np.array([0, 4.0, 0, 0])
    before = clip_grad_norm(ps, 1.0)
    after = float(np.sqrt(sum((p.grad ** 2).sum() for p in ps)))
    ps[0].grad, ps[1].grad = np.array([0.3, 0, 0]), np.zeros(4)
    clip_grad_norm(ps, 1.0)
    check(abs(before - 5) < 1e-12 and abs(after - 1) < 1e-9 and abs(ps[0].grad[0] - 0.3) < 1e-12,
          "clip_grad_norm: reports the old norm, rescales to max_norm, leaves small gradients alone")
    p = Parameter(np.array([0.0])); es = EarlyStopping(patience=2); stops = []
    for step, v in enumerate([3.0, 2.0, 2.5, 2.6, 2.7]):
        p.data[:] = step; stops.append(es.update(v, step, [p]))
    es.restore([p])
    check(stops == [False, False, False, True, True] and es.best_step == 1 and p.data[0] == 1,
          "EarlyStopping: stops after 2 bad checks and restores the best weights")

    print("\nRecurrent layers (backprop through time)")
    ids = rng.integers(0, 7, (3, 5))
    results.append(gradcheck_layer(Embedding(7, 4, rng=rng), ids, name="Embedding (params only; ids are integers)"))
    X = rng.standard_normal((3, 6, 4))
    results.append(gradcheck_layer(RNN(4, 5, rng=rng), X.copy(), name="RNN over 6 steps"))
    results.append(gradcheck_layer(RNN(4, 5, gain=1.5, rng=rng), X.copy(), name="RNN, gain 1.5 (strong recurrence)"))
    results.append(gradcheck_layer(RNN(4, 5, p_rec=0.3, rng=rng).freeze(3, 6), X.copy(), name="RNN + variational recurrent dropout"))
    results.append(gradcheck_layer(RNN(4, 5, p_rec=0.3, per_step_masks=True, rng=rng).freeze(3, 6), X.copy(), name="RNN + per-step recurrent dropout"))
    results.append(gradcheck_layer(LSTM(4, 5, rng=rng), X.copy(), name="LSTM over 6 steps"))
    results.append(gradcheck_layer(LSTM(4, 5, p_rec=0.3, rng=rng).freeze(3, 6), X.copy(), name="LSTM + variational recurrent dropout"))
    results.append(gradcheck_layer(LSTM(4, 5, p_rec=0.3, per_step_masks=True, rng=rng).freeze(3, 6), X.copy(), name="LSTM + per-step recurrent dropout"))
    results.append(gradcheck_layer(LSTM(4, 5, rng=rng), rng.standard_normal((2, 15, 4)), name="LSTM over 15 steps (long BPTT)"))
    for per_step in (False, True):
        lay = LSTM(4, 8, p_rec=0.5, per_step_masks=per_step, rng=rng)
        lay(rng.standard_normal((2, 10, 4)))
        dropped = (lay._HD == 0)[:, 1:]                      # which units of h(t-1) were zeroed at steps 1..9
        same = all((dropped[:, k] == dropped[:, 0]).all() for k in range(dropped.shape[1]))
        check(same == (not per_step), "variational masks: the SAME units are dropped at every time step" if not per_step
              else "per-step masks: a DIFFERENT set of units is dropped at each step")
    l_drop, l_none = LSTM(4, 5, p_rec=0.4, rng=np.random.default_rng(3)), LSTM(4, 5, rng=np.random.default_rng(3))
    check(np.allclose(l_drop.eval()(X), l_none(X)), "recurrent dropout is switched off in eval mode")

    print("\nCharacter language model (end to end, every regulariser on)")
    ids_in, ids_tg = rng.integers(0, 7, (3, 8)), rng.integers(0, 7, (3, 8))
    for cell in ("rnn", "lstm"):
        lm = CharLM(7, E=4, H=5, cell=cell, p_drop=0.2, p_rec=0.3, rng=np.random.default_rng(1),
                    mask_rng=np.random.default_rng(2)).freeze(3, 8)
        results.append(gradcheck_model(lm, ids_in, ids_tg, label_smoothing=0.1, name=f"CharLM {cell.upper()}: dropout + recurrent dropout + smoothing"))

    print("\nNegative controls (each must FAIL)")
    class BuggyLinear(Linear):
        def backward(self, dy):
            dx = super().backward(dy)
            self.b.grad -= dy.reshape(-1, dy.shape[-1]).sum(0) * 0.5     # BUG: corrupts db
            return dx

    class BuggyBN(BatchNorm):
        def __init__(self, C, bug):
            super().__init__(C); self.bug = bug

        def backward(self, dy):
            dy2 = dy.reshape(-1, dy.shape[-1]); N = len(dy2)
            self.gamma.grad += (dy2 * self._xhat).sum(0); self.beta.grad += dy2.sum(0)
            dxhat = dy2 * self.gamma.data
            if self.bug == "ignore_batch_stats":      # treats mean/var as constants: the classic mistake
                dx = dxhat * self._inv
            else:                                     # "forgot_variance_term": subtle, drops the xhat*sum(dxhat*xhat) piece
                dx = self._inv / N * (N * dxhat - dxhat.sum(0))
            return dx.reshape(self._shape)

    xn = rng.standard_normal((8, 5)) * 3 + 2
    caught.append(not gradcheck_layer(BuggyLinear(5, 4, rng=rng), rng.standard_normal((6, 5)), name="buggy Linear db"))
    caught.append(not gradcheck_layer(BuggyBN(5, "ignore_batch_stats"), xn.copy(), name="buggy BN: ignores batch statistics"))
    caught.append(not gradcheck_layer(BuggyBN(5, "forgot_variance_term"), xn.copy(), name="buggy BN: forgot the variance term"))


    class BuggyRNN(RNN):
        def backward(self, dout):                            # forgets that the dropout mask also blocks the GRADIENT
            B, T, H = dout.shape; out = self._out
            dA = np.empty((B, T, H)); dh_next = np.zeros((B, H))
            for t in reversed(range(T)):
                da = (dout[:, t] + dh_next) * (1.0 - out[:, t] ** 2)
                dA[:, t] = da
                dh_next = da @ self.Wh.data.T                # BUG: mask not applied on the way back
            self.Wh.grad += self._HD.reshape(-1, H).T @ dA.reshape(-1, H)
            self.Wx.grad += self._x.reshape(-1, self.nin).T @ dA.reshape(-1, H)
            self.b.grad += dA.sum(axis=(0, 1))
            return dA @ self.Wx.data.T

    class BuggyLSTM(LSTM):
        def backward(self, dout):                            # drops the cell-state carry between time steps
            B, T, H = dout.shape; G, C, TC = self._G, self._C, self._TC
            dZ = np.empty((B, T, 4 * H)); dh_next = np.zeros((B, H))
            for t in reversed(range(T)):
                i, f, g, o = G[:, t, :H], G[:, t, H:2 * H], G[:, t, 2 * H:3 * H], G[:, t, 3 * H:]
                c_prev = C[:, t - 1] if t > 0 else np.zeros((B, H))
                dh = dout[:, t] + dh_next
                do = dh * TC[:, t]
                dc = dh * o * (1.0 - TC[:, t] ** 2)          # BUG: no "+ dc_next", the highway through time is cut
                di, dg, df = dc * g, dc * i, dc * c_prev
                dZ[:, t, :H], dZ[:, t, H:2 * H] = di * i * (1 - i), df * f * (1 - f)
                dZ[:, t, 2 * H:3 * H], dZ[:, t, 3 * H:] = dg * (1 - g ** 2), do * o * (1 - o)
                dh_next = dZ[:, t] @ self.Wh.data.T
            self.Wh.grad += self._HD.reshape(-1, H).T @ dZ.reshape(-1, 4 * H)
            self.Wx.grad += self._x.reshape(-1, self.nin).T @ dZ.reshape(-1, 4 * H)
            self.b.grad += dZ.sum(axis=(0, 1))
            return dZ @ self.Wx.data.T

    caught.append(not gradcheck_layer(BuggyRNN(4, 5, p_rec=0.3, rng=rng).freeze(3, 6), X.copy(), name="buggy RNN: mask missing in backward"))
    caught.append(not gradcheck_layer(BuggyLSTM(4, 5, rng=rng), X.copy(), name="buggy LSTM: cell-state carry dropped"))

    print(f"\n{sum(results)}/{len(results)} checks passed | planted bugs caught: {sum(caught)}/{len(caught)}")
    return all(results) and all(caught)


if __name__ == "__main__":
    run_tests()