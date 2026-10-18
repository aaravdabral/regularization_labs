
import numpy as np


class Parameter:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)

    def zero_grad(self):
        self.grad[...] = 0.0


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

class Linear(Module):
    def __init__(self, nin, nout, bias=True, rng=None, init="he"):
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt((2.0 if init == "he" else 1.0) / nin)
        self.W = Parameter(rng.standard_normal((nin, nout)) * scale)
        self.b = Parameter(np.zeros(nout)) if bias else None

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
        self.gamma = Parameter(np.ones(C))
        self.beta = Parameter(np.zeros(C))
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
    worst = _rel(layer.backward(R), _num_grad(loss, x, eps))
    for p in layer.parameters():
        worst = max(worst, _rel(p.grad, _num_grad(loss, p.data, eps)))
    return _report(worst, tol, name)


def gradcheck_loss(fn, x, tol=1e-6, eps=1e-6, name=""):
    """fn(x) -> (scalar loss, dx)."""
    worst = _rel(fn(x)[1], _num_grad(lambda: fn(x)[0], x, eps))
    return _report(worst, tol, name)


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

    print(f"\n{sum(results)}/{len(results)} checks passed | planted bugs caught: {sum(caught)}/{len(caught)}")
    return all(results) and all(caught)


if __name__ == "__main__":
    run_tests()