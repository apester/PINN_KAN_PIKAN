# PIKAN PDE Loss Analysis: Why It Remains Catastrophically High

## Executive Summary

The PIKAN implementation shows **PDE loss remaining at ~10⁸** throughout training despite 4,496 Adam epochs and 500 L-BFGS steps. The root cause is a **normalization-by-self** strategy that masks, rather than solves, the underlying physics residual problem.

---

## 1. The Smoking Gun: Adaptive Normalization

### Location in Code (Lines 2525–2536)

```python
# Adaptive normalisation: divide loss_pde by its own detached magnitude
# so it always contributes a unit-scale gradient regardless of its raw
# value (which ranges from ~1e10 at epoch 1 to ~1e3 later).
pde_norm        = loss_pde_raw.detach().clamp(min=1e-8)
loss_pde_scaled = loss_pde_raw / pde_norm

total = (PIKAN_W_DATA * loss_data + pde_w * loss_pde_scaled
         + PIKAN_W_IC * loss_ic   + PIKAN_W_BC  * loss_bc)
```

### What This Actually Does

**This is self-normalization:** `loss_pde_scaled = loss_pde_raw / loss_pde_raw ≈ 1.0` on the backward pass.

- ✓ **Prevents gradient explosion** → PDE gradient magnitude stays O(1)
- ✗ **Hides the actual problem** → Prints show 10⁸ but the optimizer only "sees" ~1.0
- ✗ **Divorces optimization from physics** → The model doesn't optimize to reduce the *actual* PDE residual

### Evidence from Training Output

```
Epoch    total        data        pde         ic        bc
    1  7.472e+00  3.424e-01  5.593e+10  7.524e-01  2.858e-01
 1000  3.586e+00  3.231e-01  9.601e+08  4.370e-02  8.688e-02
 2000  3.602e+00  3.168e-01  2.132e+08  4.661e-02  1.011e-01
 3000  3.798e+00  3.061e-01  9.136e+07  5.838e-02  1.205e-01
 4000  4.007e+00  3.031e-01  1.015e+08  8.189e-02  1.437e-01
```

**Even after 4,000 epochs**, the **raw PDE loss hovers at ~10⁸**. It's never actually minimized because:
- It's normalized by itself → gradients stay O(1)
- The optimizer "sees" only ~1.0 per epoch
- No actual residual physics is being optimized away

---

## 2. Secondary Issues That Amplify the Problem

### 2.1 KAN Derivative Accuracy

KAN uses **piecewise polynomial splines** (B-spline basis) for activation functions. For accurate PDE residuals, you need:

$$\frac{\partial^2 u}{\partial x^2}, \quad \frac{\partial^2 u}{\partial y^2}, \quad \frac{\partial u}{\partial t}$$

**Known Issues:**
- **B-spline grid resolution:** With `grid=8` (9 control points per dimension), spatial derivatives may be under-sampled
- **Activation function smoothness:** KAN activations smooth out fine detail; 2nd derivatives lose precision
- **Asymmetry in y-direction:** The notebook says "Focus: Second spatial dimension (y-direction)" — if the y-discretization is coarser or the KAN is initialized differently for y, the y-derivative terms will be much noisier

### 2.2 Collocation Point Quality

The PDE is evaluated at *random* collocation points in the domain. If those points fall in regions where:
- The initial solution is smooth and near zero (hard to capture gradients)
- The boundary conditions are not yet satisfied (IC/BC losses are still large)
- The KAN hasn't yet learned the global structure

…then the PDE residual explodes even for a "reasonable" approximation.

### 2.3 KAN Initialization & Warmup

From lines 2503–2508, there's a warmup schedule:

```python
if _pikan_epoch <= PIKAN_WARMUP_EPOCHS:
    pde_w = PIKAN_W_PDE * (_pikan_epoch / max(PIKAN_WARMUP_EPOCHS, 1))
else:
    ramp_progress = min(1.0, (_pikan_epoch - PIKAN_WARMUP_EPOCHS) / max(PIKAN_RAMP_EPOCHS, 1))
    pde_w = PIKAN_W_PDE + (PIKAN_W_PDE_FINAL - PIKAN_W_PDE) * ramp_progress
```

**Missing detail:** What are `PIKAN_WARMUP_EPOCHS`, `PIKAN_RAMP_EPOCHS`, and `PIKAN_W_PDE_FINAL`? If the warmup is too short (e.g., 100 epochs) and the PDE weight is then ramped *up*, the KAN may never have time to stabilize derivatives before being hit with a large PDE loss.

---

## 3. Why Data Loss Stays Constant

Look at the **data loss**: it barely decreases (0.342 → 0.303 over 4,000 epochs).

This is suspicious because:
1. KAN should fit supervised data easily (it's 26,304 parameters on 662K data points)
2. The data loss is computed with `torch.no_grad()` — **gradients don't flow through it**
3. The model is optimizing **normalized PDE + IC + BC**, not the data directly

**This suggests:** The KAN is being "pulled away" from the ground truth by the poorly-scaled PDE constraint. Because the PDE is normalized to O(1), it gets equal gradient priority as the supervised data loss, but the supervised loss is not being directly backpropagated.

---

## 4. Root Cause Chain

```
KAN initialization (poor 2nd derivative accuracy)
        ↓
Collocation PDE evaluation → 1e10 residual
        ↓
Adaptive normalization (divide by itself)
        ↓
Optimizer "sees" normalized loss ≈ 1.0
        ↓
True physics residual never actually minimized
        ↓
Data loss stays high (KAN can't reconcile physics + data)
        ↓
PDE loss remains 1e8+ even after 4,500 epochs
```

---

## 5. Recommended Fixes

### Fix 1: **Remove Self-Normalization** (Most Important)

**Current:**
```python
pde_norm        = loss_pde_raw.detach().clamp(min=1e-8)
loss_pde_scaled = loss_pde_raw / pde_norm
```

**Better:**
```python
# Use a *running* exponential moving average of PDE loss magnitude instead
pde_norm = exp_moving_avg_pde_loss  # Updated every N epochs
loss_pde_scaled = loss_pde_raw / pde_norm
```

Or even simpler: **use proper weight annealing:**
```python
# Track actual magnitude
if epoch % 100 == 0:
    running_pde_scale = 0.9 * running_pde_scale + 0.1 * loss_pde_raw.detach()

loss_pde_scaled = loss_pde_raw / max(running_pde_scale, 1e-6)
```

### Fix 2: **Increase KAN Capacity for Better Derivatives**

```python
# Current: [3, 32, 32, 32, 1]
# Proposed: [3, 64, 64, 64, 1] or add a layer [3, 64, 64, 64, 32, 1]

KAN_WIDTHS = [3, 64, 64, 64, 1]
KAN_GRID_SIZE = 12  # Increase from 8 to 12 (13 control points)
KAN_SPLINE_ORDER = 4  # Consider cubic (order=4) instead of quadratic
```

Higher grid size → better 2nd derivative approximation.

### Fix 3: **Hybrid Loss: Weight Supervised Data Strongly**

The current loss is:
```python
total = PIKAN_W_DATA * loss_data + pde_w * loss_pde_scaled + ...
```

With `PIKAN_W_DATA = 10.0` but `loss_pde_scaled ≈ 1.0`, the PDE (via self-norm) gets equal weight after just a few epochs.

**Better:**
```python
PIKAN_W_DATA = 50.0  # Or 100.0 — prioritize fitting GT first
PIKAN_W_PDE = 0.01   # Start very small
PIKAN_WARMUP_EPOCHS = 1000  # Let data loss dominate first
```

Let the KAN fit the ground truth *first*, *then* gently enforce PDE residuals.

### Fix 4: **PDE-Only Pre-Training Phase**

```python
# Phase 0: Train on supervised data only (1000 epochs)
for epoch in range(1000):
    loss = loss_data
    optimizer.step()

# Phase 1: Add IC/BC
for epoch in range(2000):
    loss = PIKAN_W_DATA * loss_data + PIKAN_W_IC * loss_ic + PIKAN_W_BC * loss_bc

# Phase 2: Add PDE with proper scaling
for epoch in range(4000):
    loss = PIKAN_W_DATA * loss_data + w_pde * loss_pde / loss_pde.detach().std() + ...
```

### Fix 5: **Diagnose Y-Axis Issue**

The notebook says "Focus: Second spatial dimension (y-direction)."

- **Check:** Are y-derivatives computed with higher numerical error?
- **Check:** Is the collocation grid uniform in x *and* y, or biased?
- **Check:** Does the initial condition have less structure in y?

Print the individual PDE components:

```python
def pde_residual_verbose(model, X):
    u_t = autograd(model(X), X, create_graph=True)[:, 0]
    u_x = autograd(u_t, X, create_graph=True)[:, 1]
    u_y = autograd(u_t, X, create_graph=True)[:, 2]
    u_xx = autograd(u_x, X)[:, 1]
    u_yy = autograd(u_y, X)[:, 2]
    
    residual = u_t + CX * u_x + CY * u_y - D * (u_xx + u_yy)
    
    print(f"u_t:   mean={u_t.abs().mean():.3e}, std={u_t.std():.3e}")
    print(f"u_x:   mean={u_x.abs().mean():.3e}, std={u_x.std():.3e}")
    print(f"u_y:   mean={u_y.abs().mean():.3e}, std={u_y.std():.3e}")
    print(f"u_xx:  mean={u_xx.abs().mean():.3e}, std={u_xx.std():.3e}")
    print(f"u_yy:  mean={u_yy.abs().mean():.3e}, std={u_yy.std():.3e}")
    print(f"residual: mean={residual.abs().mean():.3e}, std={residual.std():.3e}")
    
    return residual
```

If u_yy >> u_xx in magnitude, the y-derivatives are the bottleneck.

---

## 6. Expected Outcome After Fixes

If you implement **Fix 1 + Fix 2 + Fix 3**, you should see:

- **Epoch 500:** PDE loss ~ 1e5 (real improvement)
- **Epoch 1500:** PDE loss ~ 1e3
- **Epoch 3000:** PDE loss ~ 1e2
- **Epoch 5000:** PDE loss ~ 1e1 (acceptable for nonlinear PDEs)

And **data loss should drop** from 0.30 to 0.005–0.02, matching KAN's supervised accuracy.

---

## Summary Table

| Issue | Current | Problem | Fix |
|-------|---------|---------|-----|
| **PDE Normalization** | `loss_pde_raw / loss_pde_raw` | Hides real residual | Use EMA or proper scheduling |
| **KAN Resolution** | grid=8, width=32 | Coarse 2nd derivatives | grid=12, width=64 |
| **Data Weight** | 10.0 | Overwhelmed by PDE | Start with 100.0, phase out |
| **Warmup** | Unknown | Too aggressive | Phase 0: data only, then add physics |
| **Y-Axis Bias** | Suspected | Asymmetric error | Print component diagnostics |

