# Microcontroller benchmark package (STM32F746, Cortex-M7 @ 216 MHz)

This package holds everything needed to benchmark one small closed-loop DBS controller network on a
microcontroller, plus an optional second network (the spiking student) for comparison. All headers are
self-contained C (no framework). `test_main.c` checks every header against the expected outputs on a host.

## Build and run the host check

```
gcc -O2 -o test_main test_main.c -lm && ./test_main
```

Expected output (all 100 test windows match for every pipeline):

```
MLP float32: actions match 100/100, max |dy| 3.05e-05
MLP int8   : actions match 100/100, max |dy| 7.63e-06
MLP int16  : actions match 100/100, max |dy| 2.59e-04
SNN float32: actions match 100/100, max |dQ| 1.22e-03, spike-count mismatches 0
```

The last digits of the `max |dy|` / `max |dQ|` values can differ with the compiler and flags. Each line compares a
pipeline with its own expected values (`MLP_ACT_F32`, `MLP_ACT_INT8`, `MLP_ACT_INT16`, `SNN_ACT`), not with float32.

## Primary network: MLP controller
80 -> 128 -> 128 -> 9, ReLU after the two hidden layers, 27,776 multiply-accumulates per inference.
Input: 80 values, the mean firing of each recorded channel over a 100 ms window (0..1).
Output: 9 values; the decision is the argmax of each group of three (frequency / pulse width / amplitude).
One inference every 100 ms.

| file | contents |
|---|---|
| `mlp_weights_f32.h` | float32 weights/biases (`MLP_W1[128][80]`, `MLP_b1[128]`, `MLP_W2[128][128]`, `MLP_b2`, `MLP_W3[9][128]`, `MLP_b3`) |
| `mlp_weights_int8.h` | int8 weights (`MLP8_*q`), int32 biases, float requantisation multipliers `MLP8_M1..M3`, input scale `MLP8_S_X` |
| `mlp_weights_int16.h` | same scheme at int16 (`MLP16_*`), int64 biases/accumulators |
| `mlp_f32.tflite`, `mlp_int8.tflite` | TensorFlow Lite models of the same network (float32 114 kB; full-int8 37 kB with int8 input/output: input scale 7.0588e-4, zero point -128; output scale 0.35111, zero point -128). For TFLite Micro + CMSIS-NN. Checked with the TFLite interpreter: float32 reproduces 100/100 float32 decisions, int8 34/100 (see "Precision" below) |
| `mlp_f32.onnx` | the float32 network as ONNX (opset 13), for any other toolchain |
| `mlp_test_vectors.h` | 100 windows: `MLP_X` (float input), `MLP_Xq8` / `MLP_Xq16` (quantised input), expected outputs `MLP_Y_F32` / `MLP_Y_INT8` / `MLP_Y_INT16`, expected decisions `MLP_ACT_*` |

Integer arithmetic (both widths, per layer): `acc = bias + sum_i x[i] * W[j][i]` (int32 for int8, int64 for int16);
hidden layers `q = clip(round_half_away_from_zero(max(acc, 0) * M), intN)`; output `y = acc * M3` as float.
`test_main.c` has a reference implementation of all three pipelines. The requantisation uses a float multiply;
a fixed-point multiplier + shift (CMSIS-NN style) is also acceptable for benchmarking, although decisions may then
differ on a few windows.

**Precision.** The int8 pipeline reproduces the float32 *decisions* on only 32 of 100 windows (this controller's
decision margins are ~0.1 on outputs of ~78, below 8-bit resolution); int16 reproduces 98/100. So int8 gives the
lower bound on energy/latency for this layer sizing, and int16 is the faithful fixed-point controller. Measuring
all three precisions is recommended, float32 and int8 at least.

## Notes for the STM32F746 Discovery (1 MB flash, 320 kB RAM)
- Include one network at a time. Flash footprint of the constant data: MLP float32 weights ~112 kB, int8 ~28 kB,
  int16 ~56 kB, MLP test vectors ~55 kB; SNN weights ~78 kB, SNN test vectors ~165 kB (inputs stored as bytes).
- The inference functions in `test_main.c` (`mlp_f32`, `mlp_i8`, `mlp_i16`, `snn_f32`) are plain C and can be
  compiled for the MCU as they are; they are the reference, not an optimised implementation. For CMSIS-NN,
  `arm_fully_connected_s8` takes the int8 weights, an int32 bias and a per-layer fixed-point multiplier + shift:
  the float multipliers `MLP8_M1..M3` convert as `M = multiplier * 2^shift` with multiplier in Q31 (the same scheme
  the TFLite int8 model already carries internally, so TFLite Micro with the CMSIS-NN kernels is the shortest path).
- The int8 TFLite input is the mean rate quantised as `xq = round(x / 7.0588e-4) - 128`; `MLP_X` in the test
  vectors is the float input. Decisions of int8 and float32 differ on most windows because of the small margins
  (see "Precision"), so for int8 compare against the TFLite interpreter's own output rather than the float32 decisions.

## Optional second network: the spiking student (same task)
16 -> 128 -> 128 -> 9 leaky integrate-and-fire neurons, feedforward, no biases, 100 time steps per 100 ms window.
Input per step: 16 values = mean of 5 neighbouring channels of the 1 ms spike bins (multiples of 0.2).
Update per step and layer (weights `W[out][in]`, `SNN_ALPHA` = 0.875, `SNN_BETA` = 0.75):
`isyn[j] = (isyn[j] + sum_i W[j][i] x[i]) * BETA; vmem[j] = ALPHA * vmem[j] + isyn[j]`;
hidden layers emit `n = floor(vmem)` spikes when `vmem >= 1` and subtract `n`; the output layer does not spike and
`Q[j] += vmem[j]` over the 100 steps; decision = argmax per group of three. State is zero at the start of every window.
Hidden spikes are sparse (on average 778 per window across the 256 hidden neurons x 100 steps, i.e. ~3%), so an
event-driven implementation (skip zero inputs, as in `test_main.c`) does ~0.26 M synaptic accumulates per window
versus 1.96 M for a dense one.

| file | contents |
|---|---|
| `snn_weights_f32.h` | `SNN_W0[128][16]`, `SNN_W2[128][128]`, `SNN_W4[9][128]`, `SNN_ALPHA`, `SNN_BETA` |
| `snn_test_vectors.h` | 100 windows: `SNN_X16c[100][100][16]` (uint8 sums of 5 channels, 0..5; the input is 0.2 x count), expected `SNN_Q`, decisions `SNN_ACT`, hidden spike counts `SNN_SPIKES` |

A fixed-point version is not provided; float32 on the M7 FPU is sufficient for this measurement.

## Measurement protocol
For each network/precision: (1) inference latency at 216 MHz (cycles or µs per window, average over the 100 windows);
(2) core current/power during inference; and (3) average power at a duty cycle of one inference per 100 ms with the
core in the lowest sleep state between inferences, if the board setup allows it ((3) can otherwise be estimated from
(1) and (2)). Record the clock, voltage, compiler and flags, and whether CMSIS-NN/DSP kernels were used.

## Other files
`mlp_reference.npz`, `snn_reference.npz` (numpy arrays of the same weights and test data), `package_info.json`
(source models, fixed-point vs float32 agreement, operation counts, test-window indices), `tflite_info.json`.
Generated by `scripts/export_rev_mcu_package.py` in the CL-DBS-RL repository.
