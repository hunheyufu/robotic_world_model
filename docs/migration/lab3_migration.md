# Isaac Lab 3.x / Isaac Sim 6.2 runtime

Status: **the only supported runtime.** Isaac Lab 3.x, Isaac Sim 6.2 and Python >= 3.12. Every
command in the verification matrix was run from the repository root; the raw logs are in
`logs/migration/`. The legacy Isaac Lab 2.x entry points, launcher, Hydra decorator and `VecEnv`
wrapper have been removed; existing checkpoints remain loadable (see §3 and §7).

## 1. Contract that had to survive

The migration keeps the *data semantics* of the pre-migration checkpoints, not just the shapes.
The machine-checked contract lives in [`schema_manifest.json`](schema_manifest.json):

| Item | Value |
|---|---|
| Policy observation | `policy=[N,48]` = lin vel 3 / ang vel 3 / projected gravity 3 / command 3 / joint pos 12 / joint vel 12 / last action 12 |
| Actions | `[N,12]`, joint position targets, scale `0.5`, default-position offset |
| World-model state | `system_state=[N,45]`, last 12 columns are the applied torque |
| Contact labels | `system_contact=[N,8]` (4 thighs then 4 feet), **normal** force `max_history(norm(force)) > 1 N` |
| Termination label | `system_termination=[N,1]`, base-body contact (not a time-out) |
| No extension head | `system_extension` is absent on ANYmal-D |
| Windows | `history_horizon=32`, forecast `8`, ensemble 1 (online) / 5 (offline) |
| Timing | `sim dt 0.005`, decimation 4, `step_dt 0.02` |
| Quaternions | never re-interpreted: the 3.x tensors are `xyzw` and are only moved between simulator and models |

## 2. Isolation (the host environment is never modified)

`C:\IsaacLab\.venv` (Python 3.12.15) and `C:\isaacsim` are used **read-only**:

* The Isaac Lab checkout is the **official `v3.0.0` tag** (`C:\IsaacLab\VERSION` = `3.0.0`). The
  installed distribution metadata in that venv reads `isaaclab 28.0.0`, `isaaclab_tasks 25.2.0`,
  `isaaclab_rl 1.5.1`; those are packaging versions, not the release tag. Rest of the stack:
  `torch 2.11.0+cu130`, `tensordict 0.14.2`, `rsl-rl-lib 5.5.1`.
* The Kit build that venv actually runs is `6.2.0-rc.14+release.52805.02189a84.gl` (`isaacsim`
  Python package `6.2.0.0`, see `.venv/Lib/site-packages/isaacsim/VERSION`). The standalone install
  `C:\isaacsim` reports the *same* Kit build. Neither is the final **Isaac Sim v6.2.0** release tag:
  in this document "6.2" always means that `rc.14` Kit build unless a release tag is named.
* no package is installed into, upgraded in or removed from that virtual environment;
* the entry points prepend `source/mbrl` to `sys.path` (see the `_REPO_SRC` block at the top of
  `scripts/reinforcement_learning/rsl_rl/{train,play,visualize}.py`), so a plain checkout is enough;
* the RWM `rsl_rl` fork is vendored as `mbrl.rl.rsl_rl_fork` (BSD-3-Clause, pinned commit
  `18eebcdd`). It keeps its `LICENSE` and carries a generated `NOTICE.md` that lists the applied
  patch series; Isaac Lab's own `rsl_rl` 5.5.1 is untouched and still used by `isaaclab_rl`;
* no second runtime path exists in the checkout. The entry points have a single (3.x)
  implementation, and `mbrl.compat` contains no generation detection.

Re-vendoring is reproducible **and checked**:

```powershell
# regenerate (refuses an existing tree without --force; warns which files it overwrites)
python scripts/setup/vendor_rsl_fork.py --source <site-packages>/rsl_rl \
    --license-file <LICENSE> --dest source/mbrl/mbrl/rl/rsl_rl_fork --force
# verify that the checked-in tree is exactly upstream + import rewrite + patch series
python scripts/setup/vendor_rsl_fork.py --source <site-packages>/rsl_rl \
    --license-file <LICENSE> --check
```

The import rewrite is mechanical; every project change lives in
`scripts/setup/rsl_fork_patches/*.patch` and is applied by the tool (and listed in `NOTICE.md`).
A hand edit or an outdated patch series is reported by `--check` instead of being overwritten, and
a failing patch or a non-compiling module aborts before the destination is replaced.
`tests/migration/test_vendor_manifest.py` covers those failure modes.

## 3. Behaviour changes required by Isaac Lab 3.x

| Area | Change |
|---|---|
| Launcher | `add_launcher_args` + `with launch_simulation(cfg, args)`; the backend is resolved from the config and reported. Default `--physics isaacsim_physx`; Newton/OVPhysX stay opt-in. `--help` and `list_envs.py` do not start Kit. |
| Pre-launch imports | Any USD/Kit import before `SimulationApp` breaks it. `mbrl/__init__.py` does not import tasks/UI eagerly, `mbrl.mbrl.envs` resolves the env classes lazily, the task walk blacklists `envs`, and `isaaclab.sensors` is type-only in the velocity mdp. `mbrl.compat` imports the `isaaclab_tasks.core` modules directly. |
| VecEnv | `mbrl.rl.rsl_rl.MbrlVecEnvWrapper` adapts the 3.x gym env to the fork's `VecEnv`: TensorDict observations, `clip_actions`, `dones = terminated \| truncated`, `extras["time_outs"]` for infinite-horizon tasks. It is the only adapter; the host wrapper (which returns the concatenated policy tensor) is not used by the fork. |
| Runner config | `RslRlRwmRunnerCfg` (own `policy`/`obs_groups`/`algorithm` layout) replaces the host runner config, which uses `actor`/`critic` models. `play.py` uses the agent config registered by the task; only when the task registers a host 3.x actor/critic config (e.g. `Isaac-Velocity-Flat-AnymalD`) does it fall back to the project's `ActorCritic` runner config, and it says so with a warning. |
| Checkpoints | `mbrl.rl.rsl_rl.runner.load_checkpoint` loads policy / world model / optimizer as explicit stages with strict shape+key validation. A **resume** (`--resume True`) restores policy, world model and the checkpoint's `iter` from the same file; a **warm start** (`--checkpoint`, `--system_dynamics_load_path`) takes each weight from the run it was named with and keeps the iteration counter at zero. Each optimizer follows the checkpoint that supplied its weights and is only restored with `--resume_optimizer`; a missing optimizer entry raises instead of being skipped. The **tensor content** of Isaac Lab 2.2.1 checkpoints (same state/action columns, same normalizer constants) remains loadable; optimizer state is not promised across torch builds. |
| Play checkpoint | `play.py` keeps the pre-migration selection order: `--use_pretrained_checkpoint` (published checkpoint of the task) > `--checkpoint <file>` > newest checkpoint of `--load_run` under `logs/rsl_rl/<experiment_name>` (default run directory, `.*`). |
| Replay buffer | The vendored `ReplayBuffer` stores resets from the environment's real `dones` (`terminated \| truncated`), not from the `system_contact`/`system_termination` learning labels, and reads the ring buffer back in insertion order. Windows never cross a reset and are never zero-padded: evaluation and imagination raise when no valid window exists, training skips that world-model update (with a warning). The five learning inputs keep their order and meaning (`state, action, extension, contact, termination`). |
| Base mass | Isaac Lab 3.x's velocity template switched `events.add_base_mass` from a uniform additive range to a multiplicative log-uniform scale. `flat_env_cfg` restores the pre-migration parameters explicitly: `randomize_rigid_body_mass`, `operation="add"`, `mass_distribution_params=(-5.0, 5.0)`, `distribution="uniform"`, body `base` (the 3.x API accepts exactly this; inertia is recomputed in both releases). |
| State access | `mbrl.compat.to_tensor` applies `.torch` at the Warp `ProxyArray` boundary; values that outlive a step are cloned. |
| Contact semantics | `net_normal_forces_w_history` is used explicitly for the RWM contact labels; 6.2 moved the *total* force to the old name. |
| env ids | Event/command terms accept `slice` (3.x full reset) as well as tensors. |
| Visualize | `ManagerBasedVisualizeEnv.step` mirrors the 3.x `ManagerBasedRLEnv.step` lifecycle (recorder, physics-owned decimation, `is_rendering`, `num_rerenders_on_reset`, UI reset) and adds only the imagination write, paired reset and history sync. |
| Video | `env_cfg.video_recorders` + `VideoRecorderCfg(source="visualizer:kit")`; the old gym `rgb_array` path no longer exists. |
| Offline | Pure PyTorch: vendored fork imports, no `wandb` requirement (`--logger tensorboard\|wandb\|none`), fail-fast CSV validation instead of waiting forever. |

## 4. Verification matrix (executed)

Interpreter: `C:\IsaacLab\.venv\Scripts\python.exe` (Python 3.12.15).
All commands were run from `D:\Train\robotic_world_model`.

Rows 1-4 and 6-11 were executed again after the replay/checkpoint/smoke/play fixes; the run
directories are from that second pass. Rows 5 and 9 (visualize, offline policy training) are from
the first pass and were not affected by those changes. The convergence re-run after removing the
legacy branches is recorded in §4.1.

| # | Path | Command | Result |
|---|---|---|---|
| 1 | Init (PPO update) | `train.py --task Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 --physics isaacsim_physx --headless --device cuda:0 --num_envs 4 --max_iterations 1 --seed 42` | `Total timesteps: 96`, finite losses, `model_0.pt` in `logs/rsl_rl/anymal_d_flat/2026-11-24_09-05-00_init` |
| 2 | Pretrain (world model update, smoke) | same entry + `--task ...-Pretrain-v0 --smoke` | `Total timesteps: 256` (= 4 envs x 64 steps, i.e. the smoke rollout really took effect), `params/agent.yaml` shows `algorithm.system_dynamics_{num_mini_batches: 2, mini_batch_size: 64, replay_buffer_size: 512, num_eval_trajectories: 2, len_eval_trajectory: 40}`; losses `state 258.16 / bound 0.9995 / contact 0.686 / termination 0.584` (finite), evaluation on a real 40-step window: `autoregressive_error 0.9246` and all noised variants finite; run `2026-11-24_09-10-00_pretrain_pretrain` |
| 3 | Finetune, world model only (warm start) | `--task ...-Finetune-v0 --smoke --system_dynamics_load_path <2>/model_1.pt` | `Loaded system dynamics ... (34 tensors)`, `Total timesteps: 160` (4 x 40), 64 valid imagination envs, PPO losses finite, evaluation on real windows (`autoregressive_error 0.9138`); run `2026-11-24_09-20-00_finetune_finetune` |
| 4 | Finetune, policy + world model (warm start) | same + `--checkpoint <1>/model_0.pt` | `Warm start ... (iteration counter restarts at 0)`, `Loaded policy ... (17 tensors)` + `Loaded system dynamics ... (34 tensors)`, `Total timesteps: 160`; run `2026-11-24_09-40-00_finetune_policydyn_finetune` |
| 5 | Visualize | `visualize.py --task ...-Visualize-v0 --physics isaacsim_physx --viz kit --num_envs 4 --max_steps 60 --video --video_length 60 --checkpoint <1> --system_dynamics_load_path <2>` | 60 steps with the GUI, `VideoRecorder` wrote 60 frames; `logs/rsl_rl/anymal_d_flat/2026-11-23_20-46-12/videos/visualize/clip_0000.mp4` decodes as h264 1280x720 @50 fps, 1.2 s |
| 6 | Play, explicit checkpoint (old run) | `play.py --task Isaac-Velocity-Flat-AnymalD --physics isaacsim_physx --headless --num_envs 2 --max_steps 20 --checkpoint logs/rsl_rl/anymal_d_flat/2026-11-23_19-29-43/model_0.pt` | `Loaded policy ... (17 tensors)`, `Reached step 20` |
| 7 | Play, only `--load_run` (old run) | same without `--checkpoint`, `--load_run 2026-11-23_19-29-43` | same checkpoint resolved from the default run root, `Reached step 20` |
| 8 | Play, project task config | `play.py --task Template-Isaac-Velocity-Flat-Anymal-D-Init-v0 ... --load_run 2026-11-23_19-29-43` | no ABI fallback warning (task runner config used), `Reached step 20` |
| 9 | Offline | `model_based/train.py --task anymal_d_flat --smoke --device cpu --max_iterations 1 --logger tensorboard --log_root logs/migration/offline` | 66-column CSV accepted, 5-member world model restored, `Total timesteps: 512`, finite PPO losses, `policy_0.pt` exported |
| 10 | Play of the offline export | `play.py --task Isaac-Velocity-Flat-AnymalD --physics isaacsim_physx --headless --num_envs 2 --max_steps 20 --checkpoint <9>/policy_0.pt` | policy loaded (17 tensors), `Reached step 20` |
| 11 | Pretrain resume | `--task ...-Pretrain-v0 --smoke --resume True --load_run 2026-11-24_09-10-00_pretrain_pretrain --max_iterations 1` | `Resuming the run ... (policy + world model + iteration counter)`, policy (17) **and** world model (34) loaded from the same `model_1.pt`, `Total timesteps: 256`, finite losses, real-window evaluation (`autoregressive_error 0.899`); run `2026-11-24_09-30-00_pretrain_resume_pretrain` |
| 12 | Visualize pairing (forced resets) | `logs/migration/12_pairing_probe.py ... env.episode_length_s=0.4` (60 steps, 20-step episodes) | 3 paired resets, history sync only on real ids `[0, 2]`, imagination writes only on odd ids `[1, 3]` for all 60 steps, zero violations |
| 13 | Vendoring check | `python scripts/setup/vendor_rsl_fork.py --source <rsl_rl> --license-file <LICENSE> --check` | `CHECK OK: ... equals upstream + rewrite + 1 patches` |

Correction of the earlier report: the first acceptance pass of row 2 claimed "smoke scaling", but
the overrides were written to the *runner* config while the fork reads
`train_config["algorithm"]`; that pass therefore ran the full world-model batch sizes (20 x 5000,
replay buffer 1000, 100 x 400 evaluation trajectories). The values above are the ones actually
consumed now, visible in `params/agent.yaml`, and the evaluation window is verified to contain real
transitions (no zero padding).

Config parity: the new `params/agent.yaml` differs from the pre-migration baseline
(`2026-11-23_19-29-43`) only by additional optional keys (`init_at_random_ep_len`, normalization
flags, empty world-model defaults); every PPO value is identical.

Unit tests (no simulator, standard library runner):

```powershell
C:\IsaacLab\.venv\Scripts\python.exe -m unittest discover -s tests\migration -p "test_*.py" -v
```

89 tests pass on Python 3.12: runtime helpers (`ProxyArray` access, `slice` env ids, normal-force
selection and the refusal to fall back to the total force), quaternion yaw/sign invariants, VectorEnv
adapter (clipping, done semantics, time-outs), staged checkpoint loading (resume vs warm start,
per-stage optimizer sources, shape/key rejection, partial loads), training/play entry-point
decisions, replay-buffer chronology/reset/no-padding invariants, smoke-override placement, the
restored base-mass event, vendoring safety (license/patches/`--check`/failure atomicity),
runner-config contract, offline window selection and the schema manifest.

The one-off pairing probe used for row 12 lives in `logs/migration/12_pairing_probe.py` (it launches
Kit, so it is kept out of the unit-test discovery pattern).

### 4.1 Convergence re-run (single-runtime code)

The legacy branches were then removed (`IS_LAB3`, `AppLauncher`, `hydra_task_config`, the old
`RslRlTensorDictVecEnvWrapper`, the duplicated 2.x `step` of the visualize env, and the
`--distributed`/`rsl-rl-lib` version check). The following was executed again on the converged tree.
All new artifacts were written under `%TEMP%\rwm_verify`; the existing `logs/`, checkpoints and
assets were only read.

| Check | Command | Result |
|---|---|---|
| Unit tests | `python -m unittest discover -s tests\migration -p "test_*.py"` | `Ran 89 tests ... OK` |
| List envs (no Kit) | `python scripts/environments/list_envs.py` | 4 `Template-...` tasks printed; no Kit startup |
| List envs missing deps | same with a Python that has no `gymnasium` | exit 1, `[ERROR] list_envs needs gymnasium and prettytable (...)` |
| Preflight | `python scripts/environments/check_lab3_runtime.py` | all four task configs resolved, `layout lab3`, `rsl-rl-lib 5.5.1` |
| Preflight launch | `... check_lab3_runtime.py --launch --physics isaacsim_physx` | `obs group policy (1, 48)`, `step ok`, `PhysxCfg`, `num_actions=12`, `launch OK` |
| CLI | `train.py --help`, `play.py --help`, `visualize.py --help` | exit 0 (argparse only; no Kit) |
| Init | `train.py ...-Init-v0 --num_envs 4 --max_iterations 1 --seed 42 --log_root %TEMP%\rwm_verify\init` | `Total timesteps: 96`, `model_0.pt` |
| Pretrain smoke | `train.py ...-Pretrain-v0 --smoke --log_root %TEMP%\rwm_verify\pretrain` | `Total timesteps: 256`; tensorboard: `state_loss 258.16`, `bound_loss 0.9995`, `contact_loss 0.6857`, `termination_loss 0.5842`, `autoregressive_error 0.9246` (all finite) |
| Finetune warm start | `train.py ...-Finetune-v0 --smoke --checkpoint <init>/model_0.pt --system_dynamics_load_path <pretrain>/model_1.pt` | policy 17 and world model 34 tensors loaded, `Total timesteps: 160`, `num_valid_imagination_envs=64`, world-model losses and `autoregressive_error 0.8904` finite |
| Visualize (GUI) | `visualize.py ... --viz kit --num_envs 4 --max_steps 20 --checkpoint logs/.../2026-11-23_19-29-43/model_0.pt --system_dynamics_load_path logs/.../2026-11-24_09-10-00_pretrain_pretrain/model_1.pt` | 20 steps, old policy (17 tensors) and world model (34 tensors) loaded, paired reset code exercised |
| Play (GUI) | `play.py --task Isaac-Velocity-Flat-AnymalD --viz kit --num_envs 4 --max_steps 20 --checkpoint <old model_0.pt>` | 20 steps, `Loaded policy ... (17 tensors)`, bounded loop stopped |
| Offline | `model_based/train.py --task anymal_d_flat --smoke --device cpu --max_iterations 1 --logger tensorboard --log_root %TEMP%\rwm_verify\offline` | 66-column CSV loaded, `Total timesteps: 512`, `policy_0.pt` exported |
| Build metadata | `setuptools.build_meta.prepare_metadata_for_build_wheel` into a temp dir | `Requires-Python: >=3.12`, classifier `Programming Language :: Python :: 3.12`, `Isaac Sim :: 6.2.0`, both TOML files parse with `tomllib` |

Resume (row 11) was not re-executed after the convergence; its resume/warm-start decision logic is
covered by `tests/migration/test_training_entrypoints.py` and `test_checkpoint_loader.py`, and the
warm-start path was executed for real in the finetune row above.

## 5. Known deviations and defects

* **Imagination `dof_acc_l2` (pre-existing).** `anymal_d_manager_based_mbrl_env.py` computes the
  joint acceleration from `policy[:, 12:24]`, which holds joint *positions*; the velocity slice is
  `[24:36]`. Kept unchanged on purpose so the migration comparison stays clean (see
  `known_defects` in the manifest). Fixing it needs its own numerical test.
* **The first pretrain iterations do not update the world model.** A training window needs
  `history_horizon + forecast horizon` (40) contiguous transitions and must not cross a reset. With
  the default 24 steps per iteration the buffer holds only 24 transitions after iteration 0, so the
  update is skipped (a `[WARN]` names the requested length) until a valid window exists. Earlier
  builds filled the buffer with zero padding; that fabricated windows and also sampled across the
  ring-buffer wrap point, which is why it was removed.
* **Evaluation is skipped while no valid trajectory exists.** `System Dynamics/autoregressive_error`
  is only written when a real, reset-free trajectory of `system_dynamics_len_eval_trajectory` is in
  the buffer. When it is written it was finite in every run of the matrix (0.899-0.928); the earlier
  "may be `inf`" note described the zero-padded evaluation and no longer applies.
* **`sequence_loss`/`kl_loss` log 0.0** for the ANYmal-D configuration (no extension head / fixed
  prior); this matches the model's own computation, not a loader problem.
* **No legacy entry point remains.** `train.py`, `play.py` and `visualize.py` each have one
  implementation (the 3.x one). Old checkpoints are still supported by the loader, not by a second
  runtime path.

## 6. Not verified (no claim made)

* No step-by-step numerical equality with the pre-migration Sim 4.5 dynamics, and no bit-identical
  policy outputs across simulators. The data contract and the tensor-level loadability of the old
  policy were verified (rows 6-8, 10 load Lab 2.2.1 checkpoints and drive them for 20 steps);
  optimizer state is deliberately *not* claimed to be portable.
* Newton (`newton_mjwarp`) and OVPhysX backends were not exercised; only `isaacsim_physx`.
* Distributed/multi-GPU training, `DistillationRunner`, and the recurrent policy variants were not
  exercised.
* Long training runs (thousands of iterations, `imagination.num_envs=8192`) were not run; the
  acceptance runs use the smoke scaling described above.
* No hardware/real-robot validation is implied by any of the above.
* The GUI runs opened Kit on this workstation; the MP4 decoding check is the machine-verifiable
  evidence for the viewport, not a human visual review.

### Environment observations (pre-existing, not introduced here)

`uv pip check --python C:\IsaacLab\.venv\Scripts\python.exe` reports six dependency
incompatibilities inside the host environment (`numba`/`numpy`, `isaacsim-core`/`fsspec`/`pywin32`,
`isaacsim-kernel`/`numpy`/`urllib3`, missing `onnxruntime-gpu`). They were present before this work,
the host environment was not modified, and none of them affected the runs above. `python -m pip`
is not available in that uv-managed venv, so the check was run with `uv pip check` (log:
`logs/migration/13_uv_pip_check.log`).

## 7. Single-runtime notes

Nothing in `C:\IsaacLab` or `C:\isaacsim` is modified, so a checkout can be removed without
uninstalling anything. There is no runtime switch: `mbrl.compat` has no generation detection and the
entry points import the `isaaclab_tasks.core` layout unconditionally. A missing/wrong Isaac Lab
install fails at import time with the normal Python error instead of silently selecting a second
code path. Old checkpoints (policy, world model, optimizer) are handled by
`mbrl.rl.rsl_rl.runner.load_checkpoint`, which validates shape and keys and reports the offending
tensor.
