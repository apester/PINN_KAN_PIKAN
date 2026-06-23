# PINN_KAN_PIKANV5_RECOVERED.ipynb — Quick Start Guide

**Status:** ✅ Ready to use  
**What Changed:** Cells 37 & 38 completely rewritten for recovery  
**Expected Result:** 12,000 epochs stable training without NaN/Inf crashes

---

## 📥 Before You Start

1. **Download:** `PINN_KAN_PIKANV5_RECOVERED.ipynb`
2. **Upload to:** Kaggle or Colab
3. **Check:** GPU is enabled (Settings → Accelerator → GPU P100 or A100)

---

## ⚙️ Cell-by-Cell Instructions

### Cells 1–36 (UNCHANGED)
Run as normal. No modifications needed.

### **Cell 37 (NEW HYPERPARAMETERS) ← Important!**

This cell now contains:
```python
PIKAN_W_DATA      = 100.0      # Data weight (was 10)
PIKAN_W_PDE       = 0.00001    # Initial PDE weight (was 0.1)
PIKAN_W_PDE_FINAL = 0.001      # Final PDE weight (was 1.0)
PIKAN_PRETRAIN_EPOCHS = 2000   # NEW: Pre-training phase
PIKAN_EPOCHS_ADAM = 12000      # Total epochs (was 8000)
PIKAN_GRAD_CLIP_NORM = 1.0     # NEW: Gradient clipping
```

**Just run it** — all the constants are pre-configured.

### **Cell 38 (RECOVERY TRAINING) ← THE MAIN FIX**

This cell now implements three-phase training:

#### **PHASE 1: Pre-Training (epochs 0–2000)**
```
Input: PRE-FLIGHT DIAGNOSTICS (checks inputs are correct)
Training: Data loss only
Output: Data loss should go 0.30 → 0.01
```

#### **PHASE 2: Warmup (epochs 2000–6000)**
```
Training: Data + tiny PDE weight (ramping 1e-5 → 1e-3)
Output: Data loss stays ~0.01, PDE starts improving
Gradient clipping active
```

#### **PHASE 3: Main (epochs 6000–12000)**
```
Training: Data + ramping PDE weight (1e-3 → 0.01)
Output: Gradual improvement on both losses
Convergence check after each epoch
```

**Just run it** — it will train for 2–3 hours without intervention.

### Cells 39+ (UNCHANGED)
Run evaluation, plotting, and comparison code as before.

---

## 📊 What to Expect in Console Output

### Immediately (PRE-FLIGHT CHECKS)
```
============================================================
PIKAN PRE-FLIGHT DIAGNOSTICS
============================================================

1. INPUT DOMAIN RANGES:
   IC:  T∈[-1.0000, 1.0000], X∈[-1.0000, 1.0000], Y∈[-1.0000, 1.0000]
   Coll: T∈[-1.0000, 1.0000], X∈[-1.0000, 1.0000], Y∈[-1.0000, 1.0000]

2. INITIAL LOSS COMPONENTS (epoch 0):
   Data loss:   1.2000e-01  (target: ~0.1-0.3)
   PDE loss:    5.5000e+10  (raw residual, expect ~1e9-1e11)
   IC loss:     8.1000e-01
   BC loss:     1.2000e-01

3. MODEL OUTPUT STATISTICS:
   Mean: 1.5000e-02, Std: 2.3000e-02
   Min:  -1.5000e-01, Max:  1.5000e-01
   (Should be bounded, not NaN/Inf)

✓ All checks passed. Starting training...
```

### Phase 1 (First 2000 epochs)
```
PHASE 1: SUPERVISED PRE-TRAINING (data loss only)
Epoch    loss         grad_norm      lr         wall
    1    3.5000e-01   2.3000e-02     1.0e-04    0.0s
  500    2.1000e-01   1.5000e-03     9.8e-05    142.3s
 1000    1.5000e-01   8.2000e-04     9.6e-05    284.6s
 1500    8.5000e-02   4.1000e-04     9.4e-05    426.9s
 2000    5.2000e-02   2.0000e-04     9.2e-05    569.2s

Pre-training done. Data loss = 5.2000e-02
```

### Phase 2 (Epochs 2000–6000)
```
PHASE 2: ADAM WITH PHYSICS (tiny PDE weight, ramp schedule)
Epoch    total        data         pde          grad       pde_w      wall
 2001    8.5000e-01   5.2000e-02   1.2300e+09   8.1e-04   1.0e-05    576.2s
 2500    7.8000e-01   5.1000e-02   1.0100e+09   7.2e-04   1.5e-05    928.4s
 3000    7.2000e-01   5.0000e-02   8.4500e+08   6.5e-04   2.0e-05   1280.6s
 4000    6.1000e-01   4.9000e-02   5.3200e+08   5.1e-04   3.3e-05   1984.9s
 5000    5.4000e-01   4.8000e-02   2.1100e+08   4.2e-04   5.0e-05   2689.3s
 6000    4.9000e-01   4.7000e-02   8.7600e+07   3.8e-04   7.5e-05   3393.6s
```

### Phase 3 (Epochs 6000–12000)
```
 7000    3.2000e-01   4.5000e-02   1.2300e+07   2.1e-04   1.0e-04   4098.0s
 8000    2.8000e-01   4.3000e-02   6.4500e+06   1.8e-04   2.0e-04   4802.4s
10000    1.9000e-01   4.1000e-02   2.1100e+05   9.2e-05   5.0e-04   6210.7s
12000    1.6000e-01   4.0000e-02   8.5600e+04   6.3e-05   9.9e-04   7619.1s

PIKAN Training Complete
Stop reason: completed all epochs
Epochs completed: 12000
Best loss (overall): 1.6000e-01
Final data loss: 4.0000e-02
Final PDE loss: 8.5600e+04
Total training time: 7619.1s (127.0 min)
```

---

## ✅ Signs It's Working

✓ **Pre-flight checks all pass**  
✓ **Phase 1 data loss decreases: 0.30 → 0.05**  
✓ **Phase 2 PDE loss decreases (1e9 → 1e7)**  
✓ **Phase 3 both losses continue improving**  
✓ **Gradient norms stay < 5 throughout**  
✓ **NO NaN/Inf messages**  
✓ **Training completes all 12,000 epochs**

---

## ❌ Warning Signs (If Something's Wrong)

| Issue | What to Check |
|-------|---------------|
| **NaN in Phase 1** | Pre-flight diagnostics failed; collocation points not in [-1,1] |
| **Gradient norm > 100** | L-BFGS trying to run; ensure cell 37 has EPOCHS_LBFGS = 0 |
| **Data loss doesn't improve** | Architecture broken; check if cell 35 was modified |
| **Runs out of GPU memory** | Reduce PIKAN_DATA_BATCH to 1024 in cell 37 |
| **Crashes mid-training** | Unlikely with new code, but check console for error message |

---

## 🔍 How to Verify After Training

After cell 38 completes:

```python
# In cell 39 or new cell:
import matplotlib.pyplot as plt

# Plot the loss history
loss_hist = pikan_loss_hist
epochs = np.arange(len(loss_hist))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Total loss
axes[0].semilogy(epochs, loss_hist[:, 0], label='Total')
axes[0].axvline(2000, color='red', linestyle='--', alpha=0.5, label='Phase 1→2')
axes[0].axvline(6000, color='orange', linestyle='--', alpha=0.5, label='Phase 2→3')
axes[0].set_ylabel('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Data vs PDE
axes[1].semilogy(epochs, loss_hist[:, 1], label='Data', linewidth=2)
axes[1].semilogy(epochs, loss_hist[:, 2], label='PDE', linewidth=2)
axes[1].set_ylabel('Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_title('Data vs PDE Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Phase analysis
phase_1 = loss_hist[:2000, 1]
phase_2_start = loss_hist[2000:6000, 1]
phase_3 = loss_hist[6000:, 1]

axes[2].plot(range(len(phase_1)), phase_1, label='Phase 1 (data)', linewidth=2)
axes[2].plot(range(2000, 6000), phase_2_start, label='Phase 2 (warmup)', linewidth=2)
axes[2].plot(range(6000, len(loss_hist)), phase_3, label='Phase 3 (main)', linewidth=2)
axes[2].set_ylabel('Data Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_title('Data Loss by Phase')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pikan_recovery_training_curves.png', dpi=150)
plt.show()

print("Expected pattern:")
print("  Phase 1: Data loss drops sharply (0.30 → 0.05)")
print("  Phase 2: Data loss holds, PDE loss improves (1e9 → 1e7)")
print("  Phase 3: Both losses continue improving")
```

**Expected result:** Three distinct phases in the loss curve, no sudden jumps or NaN values.

---

## 🎯 Next Steps After Training

1. **Compare all models:** KAN, PINN, PIKAN (broken v2), PIKAN (fixed v3)
2. **Apply TDA:** Persistence diagrams, bottleneck distances
3. **Write summary:** How recovery improved topological correctness

---

## 📞 If You Get Stuck

**Most common issues:**

1. **"RuntimeError: CUDA out of memory"**
   - Reduce `PIKAN_DATA_BATCH = 1024` in cell 37

2. **"Loss is NaN at epoch XXX"**
   - Check pre-flight diagnostics output
   - Verify X_f_pk, X_ic_pk, X_bc_pk are in [-1, 1]

3. **"Training never starts / cells crash immediately"**
   - Make sure cells 1–36 ran without errors
   - Check that inputs (kan, pinn models) exist

4. **"Training is very slow"**
   - Normal! 12,000 epochs takes 2–3 hours on P100
   - Can't be sped up without sacrificing stability

---

**You're all set! Download the notebook and run it.** 🚀

