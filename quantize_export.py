"""
Step 2: quantize the trained MLP to int8 and export everything the RTL needs.

This script IS the hardware spec. Whatever integer arithmetic happens here is
exactly what the Verilog must reproduce, bit for bit. If simulation disagrees
with the golden vectors this script emits, the RTL is wrong -- not this.

Pure numpy; run it as many times as you like while tuning.

Inputs:   mlp_fp32.npz     (from train_mlp.py)
Outputs:  rtl/w1.mem  w2.mem  b1.mem  b2.mem
          rtl/w1_t.mem      (layer-1 weights, pixel-major, 256-bit words)
          rtl/test_images.mem
          rtl/params.vh
          rtl/golden.txt
          quant_report.txt

Run:  python quantize_export.py
"""

import os
import numpy as np

# ---------------------------------------------------------------- config
OUT_DIR = "rtl"
N_GOLDEN = 64        # images dumped as golden vectors for the testbench
N_HW = 100           # images exported for the on-board accuracy run
ROUND_SHIFT = True   # round-half-up on requantize (costs one adder in RTL)

os.makedirs(OUT_DIR, exist_ok=True)
d = np.load("mlp_fp32.npz")
W1, b1, W2, b2 = d["W1"], d["b1"], d["W2"], d["b2"]
X_test, y_test, X_calib = d["X_test"], d["y_test"], d["X_calib"]


# =======================================================================
# QUANTIZATION SCHEME  (symmetric, per-tensor, power-of-two requantize)
# =======================================================================
# Per-tensor rather than per-channel: per-channel would mean 32 different
# scale factors and 32 different shifts in the RTL. Per-tensor costs a
# little accuracy and saves a lot of hardware. For MNIST there is plenty
# of headroom to spend.
#
#   x_q  = round(x * 127)                     int8,  x in [0,1]
#   W1_q = round(W1 / s_w1)                   int8,  s_w1 = max|W1| / 127
#   b1_q = round(b1 / (s_x * s_w1))           int32, lives in acc domain
#
#   acc1 = W1_q @ x_q + b1_q                  int32
#   acc1 = max(acc1, 0)                       ReLU -- acc1 now non-negative,
#                                             so the shift below is a plain
#                                             logical shift in hardware
#   h_q  = clamp(acc1 >> SHIFT1, 0, 127)      int8
#
#   W2_q = round(W2 / s_w2)                   int8
#   b2_q = round(b2 / (s_h * s_w2))           int32
#   acc2 = W2_q @ h_q + b2_q                  int32
#   pred = argmax(acc2)                       no requantize needed --
#                                             argmax is scale-invariant
# =======================================================================

INT8_MAX = 127


def quantize_weights(W):
    """Symmetric int8. Returns (int8 array, scale)."""
    scale = np.abs(W).max() / INT8_MAX
    Wq = np.clip(np.round(W / scale), -INT8_MAX, INT8_MAX).astype(np.int32)
    return Wq, scale


s_x = 1.0 / INT8_MAX
W1q, s_w1 = quantize_weights(W1)
W2q, s_w2 = quantize_weights(W2)

acc1_scale = s_x * s_w1
b1q = np.round(b1 / acc1_scale).astype(np.int32)


def quantize_input(X):
    """Pixels in [0,1] -> int8 in [0,127].

    Note this is ~ (pixel_byte >> 1), so the host can ship raw MNIST bytes
    over UART and the board can shift, rather than doing a real multiply.
    """
    return np.clip(np.round(X * INT8_MAX), 0, INT8_MAX).astype(np.int32)


# ------------------------------------------------- pick the requantize shift
# Run calibration images through layer 1 and choose the smallest shift that
# keeps the post-ReLU accumulator inside int8 range.
Xc_q = quantize_input(X_calib)
acc1_calib = Xc_q @ W1q.T + b1q
acc1_calib = np.maximum(acc1_calib, 0)
max_acc1 = int(acc1_calib.max())

SHIFT1 = max(0, int(np.ceil(np.log2(max(max_acc1, 1) / INT8_MAX))))
h_scale = acc1_scale * (2 ** SHIFT1)
b2q = np.round(b2 / (h_scale * s_w2)).astype(np.int32)


# =======================================================================
# GOLDEN MODEL -- the exact integer pipeline the RTL must match
# =======================================================================
def requantize(acc):
    """int32 post-ReLU accumulator -> int8 activation."""
    if SHIFT1 == 0:
        shifted = acc
    elif ROUND_SHIFT:
        shifted = (acc + (1 << (SHIFT1 - 1))) >> SHIFT1
    else:
        shifted = acc >> SHIFT1
    return np.clip(shifted, 0, INT8_MAX)


def int_forward(Xq):
    """Returns (acc1, h_q, acc2, pred) -- all integer, all int32-safe."""
    acc1 = Xq @ W1q.T + b1q
    acc1 = np.maximum(acc1, 0)
    hq = requantize(acc1)
    acc2 = hq @ W2q.T + b2q
    return acc1, hq, acc2, acc2.argmax(1)


def float_forward(X):
    h = np.maximum(X @ W1.T + b1, 0)
    return (h @ W2.T + b2).argmax(1)


Xq_test = quantize_input(X_test)
_, _, acc2_all, pred_q = int_forward(Xq_test)
pred_f = float_forward(X_test)

acc_float = (pred_f == y_test).mean()
acc_quant = (pred_q == y_test).mean()
agreement = (pred_q == pred_f).mean()

# Overflow headroom check. If this ever approaches 2^31 the accumulator
# width is wrong and the RTL will silently wrap.
peak_acc = max(int(np.abs(acc2_all).max()), max_acc1)


# =======================================================================
# EXPORT
# =======================================================================
def write_mem(path, values, bits):
    """Two's-complement hex, one value per line -- $readmemh compatible."""
    mask = (1 << bits) - 1
    digits = bits // 4
    with open(path, "w") as f:
        for v in np.asarray(values).flatten():
            f.write(f"{int(v) & mask:0{digits}x}\n")


p = lambda name: os.path.join(OUT_DIR, name)

# Weight memories. These are OUT-MAJOR: all 784 weights for neuron 0, then
# all 784 for neuron 1, and so on. tb_mac.v reads w1.mem in this order.
write_mem(p("w1.mem"), W1q, 8)     # 32 * 784 = 25088 entries
write_mem(p("w2.mem"), W2q, 8)     # 10 *  32 =   320 entries
write_mem(p("b1.mem"), b1q, 32)
write_mem(p("b2.mem"), b2q, 32)

# Layer-1 weights for the parallel datapath (layer1.v): PIXEL-MAJOR and
# packed. One 256-bit word per pixel holds that pixel's weight for all 32
# hidden neurons, so a single memory read per clock feeds all 32 MACs.
# Neuron n occupies bits [8n+7:8n] -- neuron 0 is the LOW byte, which is the
# RIGHTMOST two hex digits on each line.
with open(p("w1_t.mem"), "w") as f:
    n_hidden, n_in = W1q.shape
    for i in range(n_in):
        word = 0
        for n in range(n_hidden):
            word |= (int(W1q[n, i]) & 0xFF) << (8 * n)
        f.write(f"{word:0{n_hidden * 2}x}\n")   # 784 lines x 64 hex digits

# Test images for the on-board run.
write_mem(p("test_images.mem"), Xq_test[:N_HW], 8)
with open(p("test_labels.txt"), "w") as f:
    for y in y_test[:N_HW]:
        f.write(f"{int(y)}\n")

# Verilog parameter header -- single source of truth for the RTL.
with open(p("params.vh"), "w") as f:
    f.write("// Generated by quantize_export.py -- do not edit by hand.\n")
    f.write(f"`define N_IN      784\n")
    f.write(f"`define N_HIDDEN  {W1q.shape[0]}\n")
    f.write(f"`define N_OUT     {W2q.shape[0]}\n")
    f.write(f"`define ACC_W     32\n")
    f.write(f"`define SHIFT1    {SHIFT1}\n")
    f.write(f"`define ROUND_SHIFT {1 if ROUND_SHIFT else 0}\n")

# Golden vectors: full intermediate state per image, so the testbench can
# localize a mismatch to a specific layer instead of just "output wrong".
acc1_g, hq_g, acc2_g, pred_g = int_forward(Xq_test[:N_GOLDEN])
with open(p("golden.txt"), "w") as f:
    f.write(f"# image_idx label pred | acc1[{W1q.shape[0]}] | h_q[{W1q.shape[0]}] | acc2[{W2q.shape[0]}]\n")
    for i in range(N_GOLDEN):
        f.write(f"img {i} label {int(y_test[i])} pred {int(pred_g[i])}\n")
        f.write("acc1 " + " ".join(str(int(v)) for v in acc1_g[i]) + "\n")
        f.write("hq   " + " ".join(str(int(v)) for v in hq_g[i]) + "\n")
        f.write("acc2 " + " ".join(str(int(v)) for v in acc2_g[i]) + "\n")

report = f"""int8 quantization report
========================
float32 test accuracy : {acc_float * 100:.2f}%
int8    test accuracy : {acc_quant * 100:.2f}%
accuracy delta        : {(acc_quant - acc_float) * 100:+.2f} pp
prediction agreement  : {agreement * 100:.2f}%  (int8 vs float32, same input)

scales
  s_x   (input)   : {s_x:.8f}
  s_w1  (layer 1) : {s_w1:.8f}
  s_w2  (layer 2) : {s_w2:.8f}
  s_h   (hidden)  : {h_scale:.8f}

requantization
  SHIFT1          : {SHIFT1}   (rounding: {'on' if ROUND_SHIFT else 'off'})
  max post-ReLU acc1 on calib : {max_acc1}

accumulator headroom
  peak |acc| observed : {peak_acc}
  int32 limit         : {2**31 - 1}
  headroom            : {(2**31 - 1) / max(peak_acc, 1):.0f}x

memory footprint (int8 weights)
  w1 : {W1q.size} B
  w2 : {W2q.size} B
  total weights : {(W1q.size + W2q.size) / 1024:.1f} KiB
"""

with open("quant_report.txt", "w") as f:
    f.write(report)
print(report)
print(f"wrote {OUT_DIR}/ -- w1.mem w1_t.mem w2.mem b1.mem b2.mem test_images.mem params.vh golden.txt")
