# FPGA int8 MLP Accelerator

A hardware neural-network inference engine for handwritten-digit recognition (MNIST), written in Verilog for the **Digilent Basys 3** (AMD Artix-7). A 784 → 32 → 10 network is trained in PyTorch, quantized to 8-bit integers, and run on dedicated parallel multiply-accumulate hardware. The hardware matches the Python model **bit for bit**.

> **Status:** in progress. Layer 1 (32 parallel MACs) is built and verified; layer 2, argmax, and the UART link to a PC are next. See [`docs/progress.pdf`](docs/progress.pdf).

## Results so far

| | |
|---|---|
| float32 test accuracy | **95.86%** |
| int8 test accuracy | **95.86%** (no loss from quantization; 99.74% of predictions identical) |
| Layer-1 latency | **786 cycles = 7.86 µs** at 100 MHz |
| Layer-1 resources | 32 of 90 DSP slices, 15 RAMB18, ~1.7k LUTs (Yosys estimate) |
| Weight storage | 24.8 KiB, all on-chip (no external memory) |
| Verification | 6,525 automated checks across 2 testbenches, all passing in Vivado xsim |

## How it works

```mermaid
flowchart LR
    A["Image buffer<br/>784 B BRAM<br/>(from UART)"]:::plan -->|"1 pixel / clock"| B["<b>Layer 1</b><br/>32 MACs in parallel<br/>ReLU + requantize"]
    W["Weight ROM<br/>784 × 256 bit"] --> B
    B -->|"32 × int8"| C["Layer 2<br/>10 MACs"]:::plan
    C -->|"10 scores"| D["argmax"]:::plan
    D --> E(["digit"])
    classDef plan stroke-dasharray: 5 5
```
*Solid: built and verified. Dashed: planned.*

Each clock, **one pixel is broadcast to all 32 MACs** while a single 256-bit memory read gives every MAC its own weight for that pixel. All 32 hidden neurons finish together after 784 cycles. Accumulators are 32-bit, then ReLU, round-half-up shift, and clamp to int8, exactly as specified in [`quantize_export.py`](quantize_export.py).

## Verification

Python exports **golden vectors**: the exact integer value of every intermediate signal for 64 real test images. Each testbench is self-checking and compares the RTL against them.

| Testbench | Checks | What it covers |
|---|---|---|
| [`sim/tb_mac.v`](sim/tb_mac.v) | 2,266 | directed edge cases, 2,000 random dot products, golden neurons |
| [`sim/tb_layer1.v`](sim/tb_layer1.v) | 4,259 | all 32 neurons × 64 golden images, synthetic edge-case images, latency and handshake |

**The testbenches are themselves tested.** We plant realistic bugs in the RTL and confirm each one is caught. This found that real MNIST images hide three bug classes: border pixels are always black, so skipped pixels go unnoticed, and no neuron ever saturates. Synthetic test images close those gaps, and all 10 planted bugs in layer 1 are now caught, including a pipeline-timing bug that passes *every* real image. Full tables are in [`docs/design.pdf`](docs/design.pdf).

## Repository layout

```
src/        synthesizable Verilog (mac.v, layer1.v)
sim/        self-checking testbenches
rtl/        generated weights, golden vectors, params.vh (canonical, committed)
scripts/    create_project.tcl (rebuild Vivado project), run_sims.sh (Icarus regression)
docs/       design spec, progress log, setup guide (LaTeX + PDF)
*.py        train_mlp.py (training), quantize_export.py (integer spec + export)
```

## Quick start

**Vivado 2026.1** (Windows/Linux). In the Tcl console on the start screen:
```tcl
source C:/path/to/fpga-mlp-accelerator/scripts/create_project.tcl
```
Then **Run Simulation** and look for `ALL TESTS PASSED`.

**No Vivado** (macOS, or a quick check before pushing), with [Icarus Verilog](https://steveicarus.github.io/iverilog/):
```bash
./scripts/run_sims.sh
```
```
tb_layer1    ALL TESTS PASSED  (4259 checks)
tb_mac       ALL TESTS PASSED  (2266 checks)
```

Full instructions, including the free Vivado license that 2026.1 requires: [`docs/setup.pdf`](docs/setup.pdf).

## Documentation

- [**Design specification**](docs/design.pdf): architecture, arithmetic contract, timing, verification
- [**Progress log**](docs/progress.pdf): milestones, dated log, open issues
- [**Setup guide**](docs/setup.pdf): toolchain, license, running tests, team workflow

## Team

- Joey ([@itsKannapi](https://github.com/itsKannapi))
- Teammate 2 (TBD)
- Teammate 3 (TBD)
