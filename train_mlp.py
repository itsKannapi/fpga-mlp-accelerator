"""
Step 1 of the FPGA accelerator pipeline: train the float32 reference MLP.

Architecture: 784 -> 32 (ReLU) -> 10
Deliberately small so the int8 weights fit comfortably in Basys 3 BRAM.

Output: mlp_fp32.npz  (weights + the MNIST test set, so the quantization
        step downstream needs nothing but numpy)

Run:  python train_mlp.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------------------------------------------------- config
HIDDEN = 32
EPOCHS = 8
BATCH = 128
LR = 1e-3
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- model
class MLP(nn.Module):
    """No bias-free tricks, no dropout, no batchnorm.

    Every layer here has a direct hardware counterpart:
      fc1  -> MAC array pass 1
      relu -> comparator against zero
      fc2  -> MAC array pass 2
    Anything fancier (batchnorm, softmax) would add RTL we do not need,
    since argmax over the final accumulator is scale-invariant.
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        return self.fc2(F.relu(self.fc1(x)))


# ---------------------------------------------------------------- data
# ToTensor() alone puts pixels in [0, 1]. We deliberately skip mean/std
# normalization: it would push inputs negative and force a signed input
# path in RTL. With plain [0, 1], int8 input quantization collapses to
# roughly "pixel >> 1", which is free in hardware.
tf = transforms.ToTensor()

train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("./data", train=False, download=True, transform=tf)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=512)


# ---------------------------------------------------------------- train
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / total


model = MLP()
opt = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for xb, yb in train_dl:
        opt.zero_grad()
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * yb.numel()
    acc = evaluate(model, test_dl)
    print(f"epoch {epoch:2d}  loss {running / len(train_ds):.4f}  test acc {acc * 100:.2f}%")

float_acc = evaluate(model, test_dl)
print(f"\nfloat32 baseline test accuracy: {float_acc * 100:.2f}%")


# ---------------------------------------------------------------- export
# Flatten the whole test set once so the quantization script is pure numpy.
X_test = test_ds.data.numpy().astype(np.float32).reshape(-1, 784) / 255.0
y_test = test_ds.targets.numpy().astype(np.int64)

# Calibration slice comes from TRAIN, not test. The requantization shift is a
# model parameter; picking it off test images would quietly leak test data into
# the design and inflate the accuracy number we put on the resume.
X_calib = train_ds.data.numpy().astype(np.float32).reshape(-1, 784)[:2000] / 255.0

np.savez(
    "mlp_fp32.npz",
    W1=model.fc1.weight.detach().numpy(),   # (32, 784)  out-major
    b1=model.fc1.bias.detach().numpy(),     # (32,)
    W2=model.fc2.weight.detach().numpy(),   # (10, 32)
    b2=model.fc2.bias.detach().numpy(),     # (10,)
    X_test=X_test,
    y_test=y_test,
    X_calib=X_calib,
    float_acc=np.float32(float_acc),
)
print("wrote mlp_fp32.npz")
