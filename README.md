# Neuromorphic Energy-Aware Learning for Adaptive Deep Brain Stimulation

Code accompanying the paper *"Neuromorphic Energy-Aware Learning for Adaptive Deep
Brain Stimulation"*.

A spiking reinforcement-learning agent learns to modulate deep brain stimulation
(DBS) parameters in a simulated basal-ganglia–thalamocortical (CBGT) circuit,
suppressing pathological beta-band oscillations while minimizing delivered charge.
The trained agent is distilled into a sparse spiking student network deployable on
the [Xylo](https://www.synsense.ai/) neuromorphic processor.

## Repository layout

```
src/
  simulation/   CBGT network simulation (simulate_network_optimized.py) + DBS wrapper
  environment/  Gymnasium DBS environment (gym_pd.py) and replay buffers
  models/       Spiking Q-network (rockpool_dqsn.py) and ANN/RNN baselines
  utils/        Action selection, optimization, and I/O helpers
scripts/        Training, experiment, and hardware-export entry points
models/         The two deployed checkpoints (teacher + distilled student)
```

## Installation

```bash
conda env create -f environment.yml
conda activate cl-dbs-rl
# or: pip install -r requirements.txt
```

Core dependencies: PyTorch, [Rockpool](https://rockpool.ai/) (spiking networks /
Xylo), Gymnasium, NumPy/SciPy, Numba.

## Reproducing the main results

**1. Train the RL teacher** (spiking DQN on the 16-channel, Xylo-compatible network):

```bash
python -m scripts.train_rl_rockpool_16ch --curriculum --num-episodes 500
```

**2. Distill a sparse student** from the trained teacher:

```bash
python -m scripts.train_distill_rockpool_16ch \
    --teacher-checkpoint models/final_rockpool_16ch_curriculum/teacher.pth
```

The trained checkpoints used in the paper are included:

| Role | Path |
| --- | --- |
| RL teacher (spiking) | `models/final_rockpool_16ch_curriculum/teacher.pth` |
| Distilled student (ρ=0.015, λ=1500) | `models/distilled_rockpool_16ch_new/student_s0.015_t4.0_w1500.pth` |
| ANN baseline | `models/ann_baseline_curriculum/ann_baseline.pth` |
| RNN baseline | `models/rnn_baseline_curriculum/rnn_baseline.pth` |

## Experiments and baselines

| Script | Purpose |
| --- | --- |
| `run_exp1_therapeutic_efficacy.py` | Beta-suppression efficacy of teacher / student vs. baselines |
| `run_exp2b_fine_grained.py` | Fine-grained parameter-control analysis |
| `run_on_off_experiment.py` | Closed-loop on/off charge-reduction comparison |
| `run_real_to_silent_experiment.py` | Robustness to silenced neural input (supplementary) |
| `train_{ann,rnn}_baseline.py` | Non-spiking and recurrent RL baselines |

Neuromorphic deployment and benchmarking live in the Xylo export/package scripts
(`export_rl_to_xylo_v2.py`, `generate_dbs_xylo_package.py`), the on-chip benchmark
(`run_xylo_rl_benchmark.py`, `validate_on_hardware.py`), and the Jetson baselines
(`benchmark_jetson_ann*.py`).

## Citation

Citation details to be added on publication.

## License

MIT — see [LICENSE](LICENSE).
