# Neuromorphic Energy-Aware Learning for Adaptive Deep Brain Stimulation

Code, checkpoints and data accompanying the paper *"Neuromorphic Energy-Aware Learning for
Adaptive Deep Brain Stimulation"*.

A deep spiking Q-network learns to set deep brain stimulation (DBS) parameters in a biophysical
model of the rat cortico-basal ganglia-thalamic (CBGT) circuit. The controller observes the spikes
of 80 simulated neurons in each 100 ms window and adjusts stimulation frequency, pulse width and
amplitude in fixed steps. Its reward is energy-aware. Above a threshold on GPi alpha-beta power,
the reward penalizes the excess power. Below the threshold, it rewards the stimulation current
saved. The trained network (the teacher) is distilled into a sparse student that runs on the
SynSense XyloAudio 3 neuromorphic chip. The repository also contains the non-spiking controllers
(MLP, GRU and LSTM) trained with the same reward, dual-threshold and integral controllers, and a
benchmark package for an STM32F746 microcontroller (Arm Cortex-M7). In the paper, the distilled
controller held alpha-beta power in the healthy range on ten simulated patients held out from
training, using 10% of the charge of continuous stimulation.

This is the second release of the repository, published with the revised manuscript. Compared with
the initial-submission release, it adds:

- a revised training pipeline: a non-spiking Q-value readout, a reward computed on the 1 s running
  mean of the biomarker with a threshold calibrated on the training patients, and a curriculum with
  a disease-state switch and a random starting setting in every episode;
- non-spiking baselines trained with the same reward, curriculum and training budget as the
  spiking network;
- an evaluation protocol on ten held-out simulated patients, with alternating healthy and
  parkinsonian blocks and with sustained parkinsonian activity;
- a hardware benchmark of the deployed student on the XyloAudio 3 development kit, compared with
  the Cortex-M7 microcontroller.

The initial-submission code is still included. The section
[Initial-submission code](#initial-submission-code) describes what it reproduces.

## Installation

```bash
conda env create -f environment.yml      # or: ./install.sh environment.yml
conda activate cl-dbs-rl
# or: pip install -r requirements.txt
```

The paper's experiments used Python 3.9.18, PyTorch 2.3.1, Rockpool 3.0.1, Gymnasium 0.29.1,
NumPy 1.26.2, SciPy 1.11.4, Numba 0.59.0 and Matplotlib 3.8.2. Some steps need further packages:

- **PyYAML** for the experiment runner `scripts/run_rev.py`.
- **xylosim** (0.1.3; the bit-accurate XyloSim simulator, used through Rockpool) for every XyloSim step.
- **samna** (0.48.0). Because Rockpool's XyloAudio 3 module imports samna, XyloSim steps need it
  installed even without a board. The evaluation controllers `published` and `xylosim:` also need
  it, since they load `scripts/quant_fidelity_harness.py` and `scripts/quant_fidelity_lib.py`. The
  XyloAudio 3 development kit itself is needed only for the power and latency measurements.
- **TensorFlow** (2.20, CPU) and **onnx** for the TFLite and ONNX exports of the microcontroller
  package. Install them in a separate environment.
- **wandb**. The training scripts log to Weights & Biases. Set `WANDB_MODE=offline` to train
  without an account.

Training uses a CUDA GPU when one is available. The simulator runs in a single CPU thread per
job. One spiking-network training run took 6 to 18 hours.

## Repository layout

```
src/
  simulation/    CBGT network simulator (Numba) and the stepping wrapper used by all experiments
  environment/   Gymnasium environments: gym_pd.py (initial submission), gym_pd_rev.py and
                 gym_pd_round3.py (revised reward and curriculum), replay memory
  models/        spiking Q-network (rockpool_dqsn.py), revised MLP/GRU/LSTM baselines
                 (rev_baselines.py), initial-submission ANN and RNN baselines
  utils/         Q-value readout, action selection and optimization step of the revised pipeline
                 (rev_rl_utils.py), helpers of the initial submission
scripts/         entry points; files with "rev" in the name and rev_manifest.yaml are the revised
                 pipeline, the others are the initial-submission code
models/
  revision/      revised checkpoints: teachers, distilled students (the deployed student's weights,
                 every student's configuration, the Xylo configurations), non-spiking baselines
  final_rockpool_16ch_curriculum/, distilled_rockpool_16ch_new/,
  ann_baseline_curriculum/, rnn_baseline_curriculum/
                 initial-submission checkpoints
data/revision/
  records/       per-window closed-loop records, one .tar.gz per evaluation directory
  source_data/   Source Data: one CSV per figure panel and supplementary table
  mcu_package/   microcontroller benchmark package (C headers, test vectors, TFLite and ONNX models)
  mcu_results/   microcontroller and XyloAudio 3 measurement records
figures/revision/  figures and tables written by the plotting scripts (created on first run)
```

## Reproducing the results

### The experiment manifest

Every experiment of the revised pipeline is an entry in `scripts/rev_manifest.yaml`. An entry
holds a command template, a parameter grid and an output folder. `scripts/run_rev.py` expands an
entry into one job per grid point and runs the jobs in parallel:

```bash
python scripts/run_rev.py list                     # all entries and their status
python scripts/run_rev.py show final_cycling       # print the expanded commands
python scripts/run_rev.py run final_cycling [--parallel N] [--only-missing] [--match RE] [--exclude RE] [--dry-run]
```

Each job writes a log to `<outputs>/logs/`, and each invocation appends a record of its commands
and exit codes to `<outputs>/RUNS.jsonl`. The runner limits NumPy and Numba to one thread per job
and sets `WANDB_MODE=offline`. Entries with a `gpus` field spread their jobs over GPUs 0 and 1.
Before each job, the runner activates the conda environment named by `CLDBS_CONDA_ENV` (default
`cl-dbs-rl`). Set `CLDBS_PRELUDE` to replace the whole shell prefix if you do not use conda.

The manifest also keeps entries from the development of the pipeline. The steps below list the
entries behind the paper, in pipeline order.

Run names and flags:

- `RT4` is the teacher design described in the Methods: the revised reward on the 1 s running mean
  of the biomarker, with a disease-state switch and a random starting setting in every episode.
  `RT4_p202` is the teacher trained on patient 202.
- `R4` (no state switch or random start) and `RTS4` (RT4 with the stimulation setting as an extra
  input) are development arms. They are kept in the manifest.
- `--env round3` selects the revised training environment (`src/environment/gym_pd_round3.py`).
- `--pipeline corrected` selects the revised Q-value readout and action decoder
  (`src/utils/rev_rl_utils.py`): the output membrane potential is summed over the window, and the
  network state is reset before each window.
- `published` controllers are the initial-submission checkpoints.

Simulated patients are fixed random seeds of the network:

- training patients: 101, 202 and 303;
- evaluation patients: 42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192 and 9999;
- teacher selection used 2048, 3141, 5678, 8192 and 9999;
- the perturbation, block-length, simple-controller, hardened-teacher, distillation-sweep and
  quantization experiments used 42, 123, 456, 789 and 1024.

The training scripts refuse to overwrite an existing `model.pth`. To retrain a shipped checkpoint,
move it aside or change `--run-name` or `--out-root`.

### 1. Train the teacher

The deployed teacher was trained on patient 202:

```bash
python scripts/train_rev_teacher.py --pipeline corrected --env round3 --tau-beta 290 --beta-smooth 10 \
    --energy-norm max --out-threshold 1e6 --reward-scale 3000 --q-scale auto \
    --transitions --random-start --max-steps 100 \
    --patient-seed 202 --run-name RT4_p202 --out-root models/revision/teachers_round4
```

The flags map onto the Methods as follows:

| Flag | Setting |
| --- | --- |
| `--tau-beta 290 --beta-smooth 10` | reward threshold of 290 µV² on the 1 s running mean (10 windows) of the biomarker |
| `--energy-norm max` | energy term normalized by the RMS current at the maximum setting |
| `--out-threshold 1e6 --q-scale auto` | non-spiking output layer; the Q-value scale is set once before training |
| `--reward-scale 3000` | rewards divided by 3,000 before the temporal-difference loss |
| `--transitions --random-start --max-steps 100` | 100-decision episodes with one state switch and a random starting setting |

The agent seed defaults to the patient seed. All three training patients:

```bash
python scripts/run_rev.py run teachers_round4 --match 'run-name RT4_p'   # RT4 on 101, 202, 303
```

The `--match` filter keeps the RT4 arm. The entry also holds the two development arms (R4, RTS4).

### 2. Select the teacher

```bash
python scripts/run_rev.py run eval_round4 --match 'RT4_'   # screening: cycling and sustained protocols
python scripts/run_rev.py run select_round4                # cycling on the five selection patients
python scripts/analyze_rev_select.py --dir data/revision/select_round4
```

`analyze_rev_select.py` picks the RT4 teacher with the highest median charge-rate ratio
(parkinsonian over healthy blocks) among those whose parkinsonian-block power is within 10% of the
healthy unstimulated power. The shipped `eval_round4` archive also holds the reference conditions
(no stimulation, cDBS, constant and oracle stimulation), which were run by the development entries
`eval_round3` and `headroom_oracle`.

The agent-seed replicates and the energy ablation follow the same design:

```bash
python scripts/run_rev.py run teachers_round4b    # agent seeds 2 and 3, and the energy ablation (--delta 0)
python scripts/run_rev.py run eval_round4b        # cycling on the 10 patients for these runs
python scripts/analyze_rev_select.py --dir data/revision/eval_round4b --prefix RT4
```

### 3. Distill the student

The teacher's recorded closed-loop windows (100 episodes of 100 decisions on its training patient,
exploration rate 0.1) are shipped as `data/revision/distill/RT4_p202/windows.npz`. To record them
again:

```bash
python scripts/train_rev_distill.py collect --teacher models/revision/teachers_round4/RT4_p202 \
    --episodes 100 --out data/revision/distill/RT4_p202
```

Then train the students on the recorded windows. The deployed student is Xylo-native: its linear
layers have no biases, and its time constants match the chip's bit-shift decays.

```bash
python scripts/train_rev_distill.py train --teacher models/revision/teachers_round4/RT4_p202 \
    --windows data/revision/distill/RT4_p202/windows.npz --rho 0.03 --lam 500 --temperature 4 \
    --mse-weight 1 --epochs 60 --no-linear-bias --xylo-taus \
    --run-name xn_s0.03_w500 --out-root models/revision/students_round4
```

The full sweep is split over four entries:

```bash
python scripts/run_rev.py run students_round4_xn  # Xylo-native students (includes the deployed one)
python scripts/run_rev.py run students_round4     # standard students, rho x lambda grid
python scripts/run_rev.py run students_round4_t   # distillation temperature and the gap term
python scripts/run_rev.py run students_round4_nb  # students without linear biases
python scripts/analyze_rev_students.py            # agreement, hidden firing rate, closed-loop behavior
python scripts/analyze_rev_distill_margin.py      # agreement on windows where the teacher's choice is clear
```

Each student's `config.json` stores its held-out agreement with the teacher. The deployed student
has the highest held-out agreement among the Xylo-native students. `analyze_rev_distill_margin.py`
loads every student's `model.pth`, so it needs the students re-distilled first (see
[Checkpoints](#checkpoints)).

### 4. Map to the XyloAudio 3 and check on XyloSim

```bash
python scripts/run_rev_xylo_check.py --run models/revision/students_round4/xn_s0.03_w500 \
    --windows data/revision/distill/RT4_p202/windows.npz --n 300
```

This maps and quantizes the student, compares its decisions on XyloSim with the float32 network,
and writes `xylo_check.json` and `xylo/hw_config.json` into the student's folder. It overwrites the
shipped copies of both files. XyloSim needs samna installed but no board.

Closed-loop evaluation on XyloSim uses the controller `xylosim:<student folder>`:

```bash
python scripts/run_rev.py run eval_students_round4   # all students in float32, cycling
python scripts/run_rev.py run eval_xylosim_round4    # Xylo-native students on XyloSim, cycling
```

### 5. Closed-loop evaluation

All closed-loop runs go through `scripts/run_rev_eval.py`. One call runs one controller on one
patient under one protocol and writes one JSON record:

```bash
python scripts/run_rev_eval.py --controller model:models/revision/teachers_round4/RT4_p202 \
    --protocol cycling --seed 42 --out data/revision/final_cycling
```

| `--protocol` | Paper name | Schedule |
| --- | --- | --- |
| `cycling` | cycling protocol | five alternating 60 s blocks, starting healthy (3,000 decisions of 100 ms) |
| `acute` | sustained-activity protocol | 40 s of continuous parkinsonian activity |

| `--controller` | Controller |
| --- | --- |
| `model:<run dir>` | revised teacher or student (float32) |
| `xylosim:<run dir>` | student on bit-accurate XyloSim (needs `xylo/hw_config.json`) |
| `baseline:<run dir>` | revised MLP, GRU or LSTM baseline |
| `baseline_int8:<run dir>`, `baseline_int16:<run dir>` | MLP baseline through the microcontroller package's fixed-point pipeline |
| `published`, `published_ann`, `published_gru` | initial-submission teacher, ANN and GRU |
| `unstim`, `cdbs` | no stimulation; continuous DBS at 130 Hz, 0.3 ms, 300 µA |
| `constant:F,PW,A` | fixed stimulation (Hz, ms, µA) |
| `oracle:F,PW,A` | true-state switching: the setting during parkinsonian blocks only |
| `dual_threshold:LO,HI[,F,PW,A]` | on above HI, off below LO (default on-setting: cDBS) |
| `integral:TARGET,GAIN,F,PW` | amplitude changed in proportion to the biomarker's deviation from TARGET |

Learned controllers read their window length from their `config.json`. The other controllers need
`--window-ms 100`. Further options:

- `--ctrl-smooth 10`: the threshold and integral controllers observe the 1 s running mean of the
  biomarker.
- `--pd-scale S`, `--obs-drop P`, `--obs-insert R`, `--obs-delay K`: disease severity, recorded
  spikes missed with probability P, false spikes added at R Hz per neuron, observation delayed by
  K windows.
- `--block-s`, `--blocks`, `--block-jitter`: block length, block count and randomized block
  lengths in the cycling protocol.

The final evaluation entries:

```bash
python scripts/run_rev.py run final_cycling       # every condition, cycling, 10 patients
python scripts/run_rev.py run final_acute         # every condition, sustained protocol, 10 patients
python scripts/run_rev_acute_spectral.py          # sustained protocol with spectra (deployed student, unstimulated, constant 212 µA)
python scripts/run_rev.py run eval_mlp_int        # MLP in int8 and int16, cycling
python scripts/analyze_rev_final.py               # summary.json for final_cycling and final_acute
```

`final_cycling` runs only the conditions that earlier entries had not produced. The remaining
records are symbolic links into `eval_round4`, `select_round4`, `eval_round4b`,
`eval_baselines_round4`, `eval_students_round4` and `eval_xylosim_round4`, all of which are
shipped as archives. To rerun from scratch, run those entries first. Any single record can also
be regenerated with one `run_rev_eval.py` call.

### 6. Non-spiking baselines

```bash
python scripts/train_rev_baseline.py --arch mlp --input mean80 --env round3 --tau-beta 290 --beta-smooth 10 \
    --energy-norm max --reward-scale 3000 --transitions --random-start --max-steps 100 \
    --patient-seed 202 --run-name mlp_mean80_p202 --out-root models/revision/baselines_round4
```

`--arch` is `mlp`, `gru` or `lstm`. `--input mean80` averages the 80 channels over the window,
`mean16` averages the 16 pooled channels (the spiking network's layer sizes), and `raw80` passes the
full spike sequence to the recurrent networks. The entries:

```bash
python scripts/run_rev.py run baselines_round4        # MLP (mean80, mean16), GRU, LSTM x 3 training patients
python scripts/run_rev.py run baselines_round4_sweep  # learning rate x hidden width, MLP and GRU, patient 101
python scripts/run_rev.py run eval_baselines_round4
python scripts/run_rev.py run eval_baselines_sweep
python scripts/analyze_rev_baselines.py
```

### 7. Threshold and integral controllers

```bash
python scripts/run_rev.py run simple_sweep         # on the 100 ms biomarker
python scripts/run_rev.py run simple_sweep_smooth  # on the 1 s running mean
python scripts/run_rev.py run simple_sweep_smooth2 # lower dual-threshold bands on the 1 s running mean
```

The Supplementary Information lists the settings carried into the final tables.

### 8. Robustness and hardened teachers

```bash
python scripts/run_rev.py run final_robustness   # severity, spike loss, added spikes, observation delay
python scripts/run_rev.py run final_severity     # mild disease states
python scripts/run_rev.py run final_blocks       # 30 s, 120 s and randomized block lengths
python scripts/analyze_rev_robustness.py all
python scripts/run_rev.py run teachers_round5    # hardened teachers (restricted observation, added spikes)
python scripts/run_rev.py run teachers_round5b
python scripts/run_rev.py run eval_round5
python scripts/run_rev.py run eval_round5b
python scripts/analyze_rev_select.py --dir data/revision/eval_round5 --prefix RT4
```

### 9. XyloAudio 3 measurements (needs the development kit)

```bash
python scripts/run_rev_xylo_hdk.py --run models/revision/students_round4/xn_s0.03_w500 \
    --windows data/revision/mcu_package/snn_reference.npz --clocks 6.25 12.5 25 50 --n 100
python scripts/run_rev_xylo_hdk_timing.py --run models/revision/students_round4/xn_s0.03_w500 \
    --clocks 6.25 25 50 --n 20
```

The first script loads `xylo/hw_config.json` on the chip and records power on the core, IO and
analog rails, the latency per window, and the chip's output potentials for comparison with XyloSim
and float32. The second disables state monitoring to time the chip's processing alone. Both write
time-stamped JSON files to the student's `xylo/` folder. `run_rev_xylo_hdk.py --sim-only` runs
without the board.

### 10. Microcontroller package

```bash
python scripts/export_rev_mcu_package.py --out data/revision/mcu_package
<python with tensorflow and onnx> scripts/export_rev_mcu_tflite.py --pkg data/revision/mcu_package \
    --windows data/revision/distill/RT4_p202/windows.npz
cd data/revision/mcu_package && gcc -O2 -o test_main test_main.c -lm && ./test_main
```

The package holds the 80-channel MLP baseline (trained on patient 202) in float32, int8 and int16,
and the deployed student in float32, as self-contained C headers with 100 test windows and their
expected outputs. `test_main.c` checks every header on a host computer. The package README
describes the arithmetic and the files. The STM32F746 measurement record, with build settings and
timing, is `data/revision/mcu_results/stm32f746_2026-10-22.md`.

### 11. Model behavior and the field-potential surrogate

```bash
python scripts/run_rev_activity.py          # unstimulated activity of patients 42, 123, 456 (10 s each); shipped as records/activity.tar.gz
python scripts/run_rev.py run obs_collect   # observation data for the field-potential surrogate
python scripts/analyze_rev_lfp_surrogate.py
```

### 12. Figures and tables

```bash
python scripts/make_rev_fig_model_behaviour.py   # Fig. 2
python scripts/make_rev_main_figures.py          # Figs. 3, 4, 5 and 7
python scripts/make_rev_hardware_figure.py       # Fig. 6 and the Table 1 numbers
python scripts/plot_rev_acute_spectral.py        # supplementary spectral figure
python scripts/write_rev_supp_tables.py          # supplementary tables (LaTeX)
python scripts/write_rev_source_data.py          # Source Data CSV files
python scripts/write_rev_numbers.py              # every reported number with its source file
```

Figures and tables, Fig. 2 included, are written to `figures/revision/`. Diagnostic figures of
the analysis scripts go to `figures/revision/diagnostics/`. The environment variables
`CLDBS_FIG_DIR`, `CLDBS_TABLE_DIR` and `CLDBS_NUMBERS_PATH` override the figure folder, the table
folder and the path of the numbers file. The plotting scripts read the `summary.json` files of the
analysis steps. These files are included in the record archives, and the figures are built from
the shipped copies without rerunning any simulation. Some analysis scripts need every run folder
of their step (for example all student checkpoints) to rebuild a `summary.json`.

Some scripts were used only during development and need data that is not shipped:
`analyze_rev_eval_round2.py`, `analyze_rev_headroom.py`, `analyze_rev_reward_candidates.py`,
`analyze_rev_observability.py` (also needs scikit-learn), `run_rev_calibrate_tau.py`,
`run_rev_duty_cycle.py`, `plot_rev_front.py`, `plot_rev_training.py`, `make_rev_table1.py`,
`make_rev_fig2_drawio.py` and `preview_drawio.py`.

## Data and checkpoints

### Checkpoints

Each revised teacher and baseline folder holds `model.pth` and `config.json`. The `config.json`
records the exact command line that trained the model.

| Role | Path |
| --- | --- |
| RT4 teachers (patients 101, 202, 303; RT4_p202 is deployed) | `models/revision/teachers_round4/RT4_p{101,202,303}/` |
| RT4 agent-seed replicates | `models/revision/teachers_round4/RT4_p{101,202,303}_s{2,3}/` |
| Energy ablation (`--delta 0`) | `models/revision/teachers_round4/RT4d0_p{101,202,303}/` |
| Hardened teachers (patient 202) | `models/revision/teachers_round5/{RT4bg,RT4n,RT4bgn,RT4bgnd}_p202/`, `RT4bgn_p202_s2/` |
| Deployed student (Xylo-native, rho = 0.03, lambda = 500) | `models/revision/students_round4/xn_s0.03_w500/` (`model.pth`, `config.json`) |
| Configuration of every distilled student | `models/revision/students_round4/*/config.json` |
| XyloSim checks and XyloAudio 3 configurations of the Xylo-native students | `xylo_check.json`, `xylo/hw_config.json` in `models/revision/students_round4/xn_*/` |
| Development-kit measurements of the deployed student | `xylo/hdk_*.json` in the deployed student's folder |
| Non-spiking baselines, default hyperparameters | `models/revision/baselines_round4/{mlp_mean80,mlp_mean16,gru_raw80,lstm_raw80}_p{101,202,303}/` |
| Initial-submission teacher | `models/final_rockpool_16ch_curriculum/teacher.pth` |
| Initial-submission student (rho = 0.015, lambda = 1500) | `models/distilled_rockpool_16ch_new/student_s0.015_t4.0_w1500.pth` |
| Initial-submission ANN baseline | `models/ann_baseline_curriculum/ann_baseline.pth` |
| Initial-submission RNN baseline | `models/rnn_baseline_curriculum/rnn_baseline.pth` |

Only the deployed student's `model.pth` is shipped. The other students must be re-distilled from
`data/revision/distill/RT4_p202/windows.npz`; the `argv` field of each shipped `config.json` holds
the exact command line. The checkpoints of the baseline hyperparameter sweep and of the development
arms are not shipped either; their entries in the manifest retrain them.

### Closed-loop records

`data/revision/records/` holds one archive per evaluation directory (about 70 MB compressed):
`final_cycling`, `final_acute`, `final_acute_spectral`, `final_robustness`, `final_blocks`,
`simple_sweep`, `eval_round4`, `select_round4`, `eval_round4b`, `eval_baselines_round4`,
`eval_round5`, `eval_students_round4`, `eval_xylosim_round4` and `activity` (the unstimulated
activity behind Fig. 2). Unpack them in place:

```bash
for f in data/revision/records/*.tar.gz; do tar -xzf "$f" -C data/revision; done
```

Each archive unpacks into `data/revision/<name>/`, where the analysis and plotting scripts expect
it. Because `final_cycling` and `eval_round5` contain symbolic links to records in the other
directories, unpack all archives together.

Each record is one JSON file named `<protocol>_<controller>_w100[_<perturbation>]_seed<patient>.json`.
It holds one value per 100 ms decision for:

- `lfp`: GPi alpha-beta power (the biomarker, µV²);
- `freq`, `pw`, `amp`: stimulation frequency (Hz), pulse width (ms) and amplitude (µA);
- `pd_state`: disease state (0 healthy, 1 parkinsonian).

The file also stores the run's settings and its total charge. The `final_acute_spectral` records
are NumPy archives that add the 80-channel spike matrix and the GPi power spectrum of every window.

### Source Data

`data/revision/source_data/` holds one CSV per figure panel and supplementary table, written by
`scripts/write_rev_source_data.py`. Its `README.md` maps each file to its figure panel or table.

### Hardware data

`data/revision/mcu_package/` is the microcontroller benchmark package (see its `README.md`).
`data/revision/mcu_results/` holds the STM32F746 and XyloAudio 3 measurement records and
`table1_numbers.json`, the energy-per-window values behind Fig. 6 and Table 1.

## Initial-submission code

The repository as it stood at the initial submission is kept unchanged on the `v1-submission`
branch; `main` holds the revised version. Within `main`, the scripts without "rev" in their names,
and the checkpoints listed above as initial-submission, are the code of the initial submission. They implement the alternative training configuration that
the Supplementary Information compares with the revised pipeline (section "Design of the Training
Pipeline"). They also hold the edge-GPU comparison of the Supplementary Information
(`benchmark_jetson_ann*.py`). The revised evaluation harness runs the initial-submission checkpoints
as reference conditions (`published`, `published_ann`, `published_gru`).

Train the initial-submission teacher and distill its student:

```bash
python -m scripts.train_rl_rockpool_16ch --curriculum --num-episodes 500
python -m scripts.train_distill_rockpool_16ch \
    --teacher-checkpoint models/final_rockpool_16ch_curriculum/teacher.pth
```

| Script | Purpose |
| --- | --- |
| `run_exp1_therapeutic_efficacy.py` | Beta suppression of teacher and student against baselines |
| `run_exp2b_fine_grained.py` | Fine-grained parameter-control analysis |
| `run_on_off_experiment.py` | Closed-loop on/off charge comparison |
| `run_real_to_silent_experiment.py` | Response to silenced neural input |
| `train_{ann,rnn}_baseline.py` | Feedforward and recurrent RL baselines |
| `export_rl_to_xylo_v2.py`, `generate_dbs_xylo_package.py` | Xylo export and deployment package |
| `run_xylo_rl_benchmark.py`, `validate_on_hardware.py` | On-chip benchmark |
| `benchmark_jetson_ann*.py` | Jetson Orin Nano baselines |

## Citation

Citation details to be added on publication.

## License

MIT. See [LICENSE](LICENSE).
