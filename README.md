# Robotic World Model Extension for Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.2.0--rc.14-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

## Overview

This repository extends [**Isaac Lab**](https://github.com/isaac-sim/IsaacLab) with environments and training pipelines for
- [**Robotic World Model (RWM)**](https://sites.google.com/view/roboticworldmodel/home),
- [**Uncertainty-Aware Robotic World Model (RWM-U)**](https://sites.google.com/view/uncertainty-aware-rwm),

and related model-based reinforcement learning methods.

It enables:
- joint training of policies and neural dynamics models in Isaac Lab (online),
- training of policies with learned neural network dynamics without any simulator (offline),
- evaluation of model-based vs. model-free policies,
- visualization of autoregressive imagination rollouts from learned dynamics,
- visualization of trained policies in Isaac Lab.


<table>
  <tr>
  <td valign="top" width="50%">

![Robotic World Model](rwm.png)

**Paper**: [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)  
**Project Page**: [https://sites.google.com/view/roboticworldmodel](https://sites.google.com/view/roboticworldmodel)

  </td>
  <td valign="top" width="50%">

![Uncertainty-Aware Robotic World Model](rwm-u.png)

**Paper**: [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)  
**Project Page**: [https://sites.google.com/view/uncertainty-aware-rwm](https://sites.google.com/view/uncertainty-aware-rwm)

  </td>
  </tr>
</table>

**Authors**: [Chenhao Li](https://breadli428.github.io/), [Andreas Krause](https://las.inf.ethz.ch/krausea), [Marco Hutter](https://rsl.ethz.ch/the-lab/people/person-detail.MTIxOTEx.TGlzdC8yNDQxLC0xNDI1MTk1NzM1.html)  
**Affiliation**: [ETH AI Center](https://ai.ethz.ch/), [Learning & Adaptive Systems Group](https://las.inf.ethz.ch/) and [Robotic Systems Lab](https://rsl.ethz.ch/), [ETH Zurich](https://ethz.ch/en.html)


---


## Installation

This repository supports **one runtime**:

| Isaac Lab / Isaac Sim | Python | `rsl_rl` |
|---|---|---|
| 3.0.0 / 6.2 (Kit `6.2.0-rc.14`) | >= 3.12 | the RWM fork, **vendored** in `source/mbrl/mbrl/rl/rsl_rl_fork` |

The runtime was validated against this machine's installs: Isaac Lab tag `v3.0.0` and an
`isaacsim` package `6.2.0.0` whose Kit build is `6.2.0-rc.14+release.52805.02189a84.gl` (identical
to the standalone `C:\isaacsim`). The final upstream release tags are Isaac Sim `v6.2.0` and Isaac
Lab `v3.0.0`; the Kit build used here is the `rc.14` build, not the final `v6.2.0` release.

The vendored fork keeps the RWM runner/algorithm/policy ABI (`MBPOOnPolicyRunner`, `MBPOPPO`,
`ActorCritic`, `SystemDynamicsEnsemble`) so existing checkpoints keep their tensor meaning: policy
and world-model weights produced by an Isaac Lab 2.2.1 runtime also load here, while optimizer state
is only restored on request and is not claimed to be portable across torch builds. Nothing is
installed over Isaac Lab's own `rsl_rl`.
See [docs/migration/lab3_migration.md](docs/migration/lab3_migration.md) for the full contract,
the verification log and the list of unverified items.

1. **Install Isaac Lab** (not needed for offline policy training)

Follow the official [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the Conda installation as it simplifies calling Python scripts from the terminal.

2. **Install model-based RSL RL**

This step is **not** required: the fork is vendored into the repository (BSD-3-Clause, commit
`18eebcdd7145284c8d5eed5d8ed1a4b96c649693`, see `source/mbrl/mbrl/rl/rsl_rl_fork/NOTICE.md`).
Project changes to the vendored code live in `scripts/setup/rsl_fork_patches/`; regenerate with
`scripts/setup/vendor_rsl_fork.py --force` and verify a checkout with `--check` (it refuses to
overwrite unpatched local edits silently).

3. **Clone this repository** (outside your Isaac Lab directory)

```bash
git clone git@github.com:leggedrobotics/robotic_world_model.git
```

4. **Install the extension** using the Python environment where Isaac Lab is installed

```bash
python -m pip install -e source/mbrl
```

On a shared/pre-existing Isaac Lab 3.x install you can skip the install: the entry points locate
`source/mbrl` from the checkout and prepend it to `sys.path`, so the host virtual environment stays
untouched.

5. **Verify the installation** (not needed for offline policy training)

```bash
# any python from a venv that has Isaac Lab 3.x installed
python ./scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 \
  --physics isaacsim_physx --headless --device cuda:0 --num_envs 4 --max_iterations 1 --seed 42
```

The run reports `Total timesteps: 96` (24 steps per env times 4 envs) after one iteration and writes
`logs/rsl_rl/anymal_d_flat/<timestamp>/model_0.pt` with finite PPO losses.

There is also a pre-flight check that prints the interpreter, package versions, the resolved task
configs and (with `--launch`) a one-step environment smoke:

```bash
python scripts/environments/check_lab3_runtime.py --launch --physics isaacsim_physx
```

---

## World Model Pretraining & Evaluation

Robotic World Model is a model-based reinforcement learning algorithm that learns a dynamics model and a policy concurrently.

### Configure model inputs/outputs

You can configure the model inputs and outputs under `ObservationsCfg_PRETRAIN` in [`AnymalDFlatEnvCfg_PRETRAIN`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py).

Available components:
- `SystemStateCfg`: state input and output head
- `SystemActionCfg`: action input
- `SystemExtensionCfg`: continuous privileged output head (e.g. rewards etc.)
- `SystemContactCfg`: binary privileged output head (e.g. contacts)
- `SystemTerminationCfg`: binary privileged output head (e.g. terminations)

And you can configure the model architecture and training hyperparameters under `RslRlSystemDynamicsCfg` and `RslRlMbrlPpoAlgorithmCfg` in [`AnymalDFlatPPOPretrainRunnerCfg`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py) .

Available options:
- `ensemble_size`: ensemble size for uncertainty estimation
- `history_horizon`: stacked history horizon
- `architecture_config`: architecture configuration
- `system_dynamics_forecast_horizon`: autoregressive prediction steps

### Run dynamics model pretraining:

```bash
# add --smoke for an acceptance-sized run
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Pretrain-v0 \
  --physics isaacsim_physx --headless --device cuda:0 --num_envs 4 --max_iterations 1 --smoke
```

It trains a PPO policy from scratch, while the induced experience during training is used to train the dynamics model.
`--smoke` collects at least one `history_horizon + forecast_horizon` window (32 + 8 transitions) and
runs a small number of world-model mini-batches, so a short run still performs a real model update.

### Visualize autoregressive predictions

```bash
python scripts/reinforcement_learning/rsl_rl/visualize.py \
  --task Template-Isaac-Velocity-Flat-Anymal-D-Visualize-v0 \
  --physics isaacsim_physx --viz kit --num_envs 4 \
  --checkpoint <checkpoint_path> \
  --system_dynamics_load_path <dynamics_model_path> \
  --max_steps 200 --video --video_length 200
```

It visualizes the learned dynamics model by rolling out the model autoregressively in imagination, conditioned on the actions from the learned policy.
The `dynamics_model_path` should point to the pretrained dynamics model checkpoint (e.g. `model_<iteration>.pt`) inside the saved run directory.
`--num_envs` is the **total** number of instances and must be even: even indices are the real
instances, odd indices are their imagination partners. Recordings are written with the Isaac Lab 3.x
`video_recorders` API (`visualizer:kit`) next to the checkpoint (`videos/visualize/clip_*.mp4`).

---

## Model-Based Policy Training & Evaluation

Once a dynamics model is pretrained, you can train a model-based policy purely from **imagined rollouts** generated by the learned dynamics.

There are two options:
- **Option 1: Train policy in imagination *online***, where additional environment interactions are continually collected using the latest policy to update the dynamics model (as implemented with RWM and MBPO-PPO in [Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics](https://arxiv.org/abs/2501.10100)).
- **Option 2: Train policy in imagination *offline*** where no additional environment interactions are collected and the policy has to rely on the static dynamics model (as implemented with RWM-U and MOPO-PPO in [Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots](https://arxiv.org/abs/2504.16680)).

### Option 1: Train policy in imagination *online*

The online data collection relies on interactions with the environment and thus brings up the simulator.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Finetune-v0 --physics isaacsim_physx --headless --checkpoint <checkpoint_path> --system_dynamics_load_path <dynamics_model_path>
```

You can either start the policy from pretrained checkpoints or from scratch by simply omitting the `--checkpoint` argument (the world model is still restored from `--system_dynamics_load_path`).
Both stages are loaded separately and strictly: a shape/key mismatch is reported with the offending
tensor instead of silently producing an untrained run. `--smoke` sets the warmup to zero so a single
iteration still contains a non-empty imagination rollout and a PPO update.

### Option 2: Train policy in imagination *offline*

The offline policy training does not request any new data and thus relies solely on the static dynamics model.
Align the model architecture and specify the model load path under `ModelArchitectureConfig` in [`AnymalDFlatConfig`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).

Additionally, the offline imagination needs to branch off from some initial states. Specify the data path under `DataConfig` in [`AnymalDFlatConfig`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).
The offline path is pure PyTorch (no Isaac Lab/Kit import) and no longer waits for a data producer:
a missing file, a wrong column count or too few rows fails immediately.

```bash
python scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat --device cpu --max_iterations 500 --logger tensorboard
python scripts/reinforcement_learning/model_based/train.py --task anymal_d_flat --smoke --device cpu --max_iterations 1
```

### Play the learned model-based policy

You can play the learned policies with the original Isaac Lab task registry. The official flat
ANYmal-D task is `Isaac-Velocity-Flat-AnymalD`; play mode overrides are applied through the
`--play_mode` handling inside the script.

The checkpoint is selected like before: `--use_pretrained_checkpoint` (published checkpoint of the
task), else `--checkpoint <file>`, else the newest checkpoint of `--load_run` under
`logs/rsl_rl/<experiment_name>` (the default run directory). `--max_steps` bounds the 3.x loop.

```bash
# bounded GUI run with the pre-migration policy ABI
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-AnymalD --physics isaacsim_physx --viz kit --num_envs 4 \
  --max_steps 200 --checkpoint <checkpoint_path>

# same, but let the script find the newest checkpoint of a run
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-AnymalD --physics isaacsim_physx --headless --max_steps 20 \
  --load_run 2026-11-23_19-29-43
```

When the task registers Isaac Lab 3.x's actor/critic runner config (the official task does), play
uses the project's `ActorCritic` configuration - pre-migration checkpoints cannot be loaded into the
host ABI - and prints a warning saying so.

---

## Isaac Lab 3.x / Isaac Sim 6.2 runtime

The migration keeps the data contract (observation groups, shapes, joint/body order, quaternion
convention and contact semantics) documented in
[docs/migration/schema_manifest.json](docs/migration/schema_manifest.json) and verified in
[docs/migration/lab3_migration.md](docs/migration/lab3_migration.md).

What changed for the new runtime:

- Tasks are registered the same way; the env configs import the velocity task modules through
  `mbrl.compat` from the `isaaclab_tasks.core` layout.
- Kit is started by `add_launcher_args` + `launch_simulation` once the config (and therefore the
  backend) is resolved. `--physics isaacsim_physx` is the default; Newton/OVPhysX remain opt-in.
- The environment is wrapped by a project adapter (`mbrl.rl.rsl_rl.MbrlVecEnvWrapper`) instead of
  Isaac Lab 3.x's wrapper, because the fork's `VecEnv` ABI is older. Action clipping,
  `dones = terminated | truncated` and `extras["time_outs"]` are preserved.
- Articulation/contact data is read through `mbrl.compat.to_tensor` (Warp `ProxyArray` in 3.x) and
  the RWM contact labels use `net_normal_forces_w_history` explicitly, because
  `net_forces_w_history` became the *total* force in 6.2.
- The world-model runner configuration is defined by the project (`RslRlRwmRunnerCfg`) instead of
  inheriting the host's actor/critic runner config.
- Checkpoint loading separates a **resume** (same run: policy + world model + iteration counter)
  from a **warm start** (weights from another run, counter reset); each optimizer follows the
  checkpoint that supplied its weights and is only restored with `--resume_optimizer`.
- The system-dynamics replay buffer uses the environment's real `dones` as episode boundaries and
  reads the ring buffer in insertion order, so training/evaluation windows are contiguous,
  reset-free and never zero-padded. `events.add_base_mass` keeps the pre-migration uniform additive
  range instead of 3.x's log-uniform scaling.
- Video recording uses `env_cfg.video_recorders` (`VideoRecorderCfg(source="visualizer:kit")`).
- The visualize environment mirrors the 3.x `ManagerBasedRLEnv.step` lifecycle and only adds the
  imagination write, the paired reset and the history synchronization.

Testing (no simulator required):

```bash
python -m unittest discover -s tests/migration -p "test_*.py" -v
```

---

## Code Structure

We provide a reference pipeline that enables RWM and RWM-U on ANYmal D.

Key files:

**Runtime**

- Runtime helpers (`ProxyArray` access, contact semantics, config resolution, launcher)
  [`compat.py`](source/mbrl/mbrl/compat.py).
- Vendored RWM `rsl_rl` fork (BSD-3-Clause, pinned commit, patch log)
  [`rsl_rl_fork/`](source/mbrl/mbrl/rl/rsl_rl_fork).
- `VecEnv` adapter + staged checkpoint loader
  [`vecenv_wrapper.py`](source/mbrl/mbrl/rl/rsl_rl/vecenv_wrapper.py),
  [`runner.py`](source/mbrl/mbrl/rl/rsl_rl/runner.py).
- Isaac Lab 3.x launch/run helpers
  [`lab3_runtime.py`](scripts/reinforcement_learning/rsl_rl/lab3_runtime.py).

**Online**

- Environment configurations + dynamics model setup
  [`flat_env_cfg.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py).
- Algorithm configuration + training parameters
  [`rsl_rl_ppo_cfg.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py).
- Imagination rollout logic (constructs policy observations & rewards from model outputs)
  [`anymal_d_manager_based_mbrl_env`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_mbrl_env.py).
- Visualization environment + rollout reset
  [`anymal_d_manager_based_visualize_env.py`](source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/anymal_d_manager_based_visualize_env.py).

**Offline**

- Environment configurations + Imagination rollout logic (constructs policy observations & rewards from model outputs)
  [`anymal_d_flat.py`](scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py).
- Algorithm configuration + training parameters
  [`anymal_d_flat_cfg.py`](scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py).
- Pretrained RWM-U checkpoint
  [`pretrain_rnn_ens.pt`](assets/models/pretrain_rnn_ens.pt).
- Initial states for imagination rollout
  [`state_action_data_0.csv`](assets/data/state_action_data_0.csv).


---

## Citation
If you find this repository useful for your research, please consider citing:

```text
@article{li2025robotic,
  title={Robotic world model: A neural network simulator for robust policy optimization in robotics},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2501.10100},
  year={2025}
}
@article{li2025uncertainty,
  title={Uncertainty-Aware Robotic World Model Makes Offline Model-Based Reinforcement Learning Work on Real Robots},
  author={Li, Chenhao and Krause, Andreas and Hutter, Marco},
  journal={arXiv preprint arXiv:2504.16680},
  year={2025}
}
```
