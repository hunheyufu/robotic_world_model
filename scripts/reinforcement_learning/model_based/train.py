import os
import sys

# Run from a checkout without installing the extension into the host environment.
_REPO_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "source", "mbrl"))
if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)

from configs import BaseConfig, AnymalDFlatConfig
from envs import BaseEnv, AnymalDFlatEnv
from policy_training import PolicyTraining, RunLogger

# The offline path is pure PyTorch: it uses the vendored RWM fork and never imports Isaac Lab/Kit.
from mbrl.rl.rsl_rl_fork.modules import ActorCritic, SystemDynamicsEnsemble
from mbrl.rl.rsl_rl_fork.algorithms import PPO
from mbrl.rl.rsl_rl_fork.utils import resolve_obs_groups

import torch
from torch.utils.data import Dataset
import argparse
from datetime import datetime
import time
import pandas as pd


def select_valid_windows(num_steps: int, history_horizon: int, forecast_horizon: int, termination_data: torch.Tensor) -> list[int]:
    """Return the start indices of windows that do not cross an episode boundary.

    A window covers ``history_horizon + forecast_horizon`` steps.  It is rejected when a
    termination flag falls inside it (excluding the final step, which starts the next episode), so
    the RNN is never trained across two episodes.

    Args:
        num_steps: Number of transition steps in the loaded trajectory.
        history_horizon: Number of past steps in a window.
        forecast_horizon: Number of predicted steps in a window.
        termination_data: Termination flags shaped like the loaded state data; every non-zero entry
            marks an episode boundary.

    Returns:
        Ascending list of valid window start indices.
    """
    window = history_horizon + forecast_horizon
    reset_indices = termination_data.flatten().nonzero(as_tuple=False).squeeze(-1)
    valid_indices: list[int] = []
    for i in range(num_steps - window + 1):
        inside = reset_indices[(reset_indices >= i) & (reset_indices < i + window - 1)]
        if len(inside) == 0:
            valid_indices.append(i)
    return valid_indices


class ModelBasedExperiment:
      

    def __init__(self, environment, device):
        self.env_cls = self.resolve_environment_cls(environment)
        self.device = device
        self.data_file_idx = 0
        
    
    def resolve_environment_cls(self, environment):
        if environment == "anymal_d_flat":
            return AnymalDFlatEnv
        else:
            raise ValueError(f"Unknown environment: {environment}")

    
    def _load_data(self, dataset_root, dataset_folder, file_data_size=10000, batch_data_size=50000):
        batch_state_data = []
        batch_action_data = []
        batch_extension_data = []
        batch_contact_data = []
        batch_termination_data = []
        batch_total_num_data = 0
        while True:
            file = f"state_action_data_{self.data_file_idx}.csv"
            path = os.path.join(dataset_root, dataset_folder, file)
            if not os.path.isfile(path):
                # The offline path no longer waits for a producer: a missing file is a hard error.
                raise FileNotFoundError(
                    f"[Motion Loader] Dataset file not found: {os.path.abspath(path)}. "
                    "The offline run expects the shipped CSV; pass --dataset_root/--dataset_folder "
                    "if your data lives elsewhere."
                )
            data = pd.read_csv(path, header=None)
            expected_columns = self.state_dim + self.action_dim + self.extension_dim + self.contact_dim + self.termination_dim
            if data.shape[1] != expected_columns:
                raise ValueError(
                    f"[Motion Loader] {file} has {data.shape[1]} columns, expected {expected_columns} "
                    f"(state {self.state_dim} + action {self.action_dim} + extension {self.extension_dim} "
                    f"+ contact {self.contact_dim} + termination {self.termination_dim})."
                )
            if len(data) < file_data_size:
                raise ValueError(
                    f"[Motion Loader] {file} has {len(data)} rows, fewer than the required {file_data_size}."
                )
            state_data = torch.tensor(data.iloc[:, :self.state_dim].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_data = torch.tensor(data.iloc[:, self.state_dim:self.state_dim + self.action_dim].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            extension_data = torch.tensor(data.iloc[:, self.state_dim + self.action_dim:self.state_dim + self.action_dim + self.extension_dim].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            contact_data = torch.tensor(data.iloc[:, self.state_dim + self.action_dim + self.extension_dim:self.state_dim + self.action_dim + self.extension_dim + self.contact_dim].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            termination_data = torch.tensor(data.iloc[:, self.state_dim + self.action_dim + self.extension_dim + self.contact_dim:].values, dtype=torch.float32, device=self.device).unsqueeze(0)
            batch_state_data.append(state_data)
            batch_action_data.append(action_data)
            batch_extension_data.append(extension_data)
            batch_contact_data.append(contact_data)
            batch_termination_data.append(termination_data)
            batch_total_num_data += len(data)
            num_trajs, num_steps, state_dim = state_data.shape
            print(f"[Motion Loader] Loaded {num_trajs} {dataset_folder} trajectories of {num_steps} steps from {file}.")
            print(f"[Motion Loader] Total number of data: {batch_total_num_data} / {batch_data_size}")
            self.data_file_idx += 1
            if batch_total_num_data >= batch_data_size:
                break
        batch_state_data = torch.cat(batch_state_data, dim=1)
        batch_action_data = torch.cat(batch_action_data, dim=1)
        batch_extension_data = torch.cat(batch_extension_data, dim=1)
        batch_contact_data = torch.cat(batch_contact_data, dim=1)
        batch_termination_data = torch.cat(batch_termination_data, dim=1)
        action_dim, extension_dim, contact_dim, termination_dim = action_data.shape[-1], extension_data.shape[-1], contact_data.shape[-1], termination_data.shape[-1]
        print(f"[Motion Loader] State dim: {state_dim} | Action dim: {action_dim} | Extension dim: {extension_dim} | Contact dim: {contact_dim} | Termination dim: {termination_dim}")
        return batch_state_data, batch_action_data, batch_extension_data, batch_contact_data, batch_termination_data
    

    def _build_eval_traj_config(self, eval_state_data, eval_action_data, eval_extension_data, eval_contact_data, eval_termination_data, num_eval_trajectories, num_visualizations, len_eval_trajectory, state_idx_dict):
        num_eval_trajs, num_eval_steps, _ = eval_state_data.shape
        ids = torch.randint(0, num_eval_trajs, (num_eval_trajectories,))
        len_eval_trajectory = min(len_eval_trajectory, num_eval_steps)
        start_steps = torch.randint(0, num_eval_steps - len_eval_trajectory + 1, (num_eval_trajectories,))
        start_steps_expanded = start_steps[:, None] + torch.arange(len_eval_trajectory)
        traj_data = [
            eval_state_data[ids[:, None], start_steps_expanded],
            eval_action_data[ids[:, None], start_steps_expanded],
            eval_extension_data[ids[:, None], start_steps_expanded],
            eval_contact_data[ids[:, None], start_steps_expanded],
            eval_termination_data[ids[:, None], start_steps_expanded],
            ]
        self.eval_traj_config = {
            "num_trajs": num_eval_trajectories,
            "num_visualizations": num_visualizations,
            "len_traj": len_eval_trajectory,
            "traj_data": traj_data,
            "state_idx_dict": state_idx_dict,
        }


    def prepare_environment(self, num_envs, max_episode_length, step_dt, reward_term_weights, uncertainty_penalty_weight, observation_noise, command_resample_interval_range, event_interval_range):
        self.env: BaseEnv = self.env_cls(num_envs, max_episode_length, step_dt, reward_term_weights, self.device, uncertainty_penalty_weight, observation_noise, command_resample_interval_range, event_interval_range)


    def prepare_data(self, dataset_root, dataset_folder, file_data_size, batch_data_size, state_data_mean=None, state_data_std=None, action_data_mean=None, action_data_std=None, init_data_ratio=0.0, num_eval_trajectories=100, num_visualizations=2, len_eval_trajectory=400, state_idx_dict=None):
        state_data, action_data, extension_data, contact_data, termination_data = self._load_data(dataset_root, dataset_folder=dataset_folder, file_data_size=file_data_size, batch_data_size=batch_data_size)
        self._build_eval_traj_config(state_data, action_data, extension_data, contact_data, termination_data, num_eval_trajectories, num_visualizations, len_eval_trajectory, state_idx_dict)
        class SystemDynamicsDataset(Dataset):
            def __init__(
                self,
                history_horizon,
                forecast_horizon,
                state_data,
                action_data,
                extension_data,
                contact_data,
                termination_data,
                state_data_mean=None,
                state_data_std=None,
                action_data_mean=None,
                action_data_std=None,
                ):
                self.history_horizon = history_horizon
                self.forecast_horizon = forecast_horizon
                
                self.state_data_mean = torch.tensor(state_data_mean, device=state_data.device) if state_data_mean is not None else state_data.mean(dim=(0, 1))
                self.state_data_std = torch.tensor(state_data_std, device=state_data.device) if state_data_std is not None else state_data.std(dim=(0, 1)) + 1e-6
                self.action_data_mean = torch.tensor(action_data_mean, device=action_data.device) if action_data_mean is not None else action_data.mean(dim=(0, 1))
                self.action_data_std = torch.tensor(action_data_std, device=action_data.device) if action_data_std is not None else action_data.std(dim=(0, 1)) + 1e-6
                state_data, action_data = self.normalize(state_data, action_data)
                
                reset_indices = termination_data.flatten().nonzero(as_tuple=False).squeeze(-1)
                print(f"[Motion Loader] Termination steps in the loaded data: {len(reset_indices)}")
                valid_indices = select_valid_windows(
                    state_data.shape[1], history_horizon, forecast_horizon, termination_data
                )
                # (num_groups, history_horizon + forecast_horizon, dim)
                self.state_data = torch.cat([state_data[:, i:i + history_horizon + forecast_horizon, :] for i in valid_indices], dim=0)
                self.action_data = torch.cat([action_data[:, i:i + history_horizon + forecast_horizon, :] for i in valid_indices], dim=0)
                self.extension_data = torch.cat([extension_data[:, i:i + history_horizon + forecast_horizon, :] for i in valid_indices], dim=0)
                self.contact_data = torch.cat([contact_data[:, i:i + history_horizon + forecast_horizon, :] for i in valid_indices], dim=0)
                self.termination_data = torch.cat([termination_data[:, i:i + history_horizon + forecast_horizon, :] for i in valid_indices], dim=0)

            def __len__(self):
                return len(self.state_data)

            def __getitem__(self, idx):
                return self.state_data[idx], self.action_data[idx], self.extension_data[idx], self.contact_data[idx], self.termination_data[idx]
            
            def normalize(self, state_data=None, action_data=None):
                state_data = (state_data - self.state_data_mean) / self.state_data_std if state_data is not None else None
                action_data = (action_data - self.action_data_mean) / self.action_data_std if action_data is not None else None
                return state_data, action_data
            
            def denormalize(self, state_data, action_data):
                state_data = state_data * self.state_data_std + self.state_data_mean if state_data is not None else None
                action_data = action_data * self.action_data_std + self.action_data_mean if action_data is not None else None
                return state_data, action_data
            
            def sample_batch(self, batch_size, normalized=True):
                idx = torch.randint(0, len(self.state_data), (batch_size,))
                if normalized:
                    return self.state_data[idx, :self.history_horizon], self.action_data[idx, :self.history_horizon]
                else:
                    return self.denormalize(self.state_data[idx, :self.history_horizon], self.action_data[idx, :self.history_horizon])
            
        self.dataset = SystemDynamicsDataset(
            self.history_horizon,
            self.forecast_horizon,
            state_data,
            action_data,
            extension_data,
            contact_data,
            termination_data,
            state_data_mean=state_data_mean,
            state_data_std=state_data_std,
            action_data_mean=action_data_mean,
            action_data_std=action_data_std
            )
        # init env dataset
        self.env.set_dataset(self.dataset)
        if self.env.init_dataset is None:
            self.env.set_init_dataset(self.dataset, init_data_ratio)


    def prepare_model(self, history_horizon, forecast_horizon, extension_dim, contact_dim, termination_dim, ensemble_size, architecture_config, freeze_auxiliary=False, resume_path=None):
        self.history_horizon = history_horizon
        self.forecast_horizon = forecast_horizon
        self.state_dim = self.env.state_dim
        self.action_dim = self.env.action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.system_dynamics = SystemDynamicsEnsemble(
            self.state_dim,
            self.action_dim,
            self.extension_dim,
            self.contact_dim,
            self.termination_dim,
            self.device,
            ensemble_size=ensemble_size,
            history_horizon=self.history_horizon,
            architecture_config=architecture_config,
            freeze_auxiliary=freeze_auxiliary,
            )
        self.model_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Model] Loading model from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.system_dynamics.load_state_dict(loaded_dict["system_dynamics_state_dict"])
            self.model_learning_iteration = loaded_dict["iter"]
        # init env system dynamics
        self.env.set_system_dynamics(self.system_dynamics)
            

    def prepare_policy(self, observation_dim, obs_groups, action_dim, actor_hidden_dims, critic_hidden_dims, activation, init_noise_std, resume_path=None):
        self.observation_dim = observation_dim
        default_sets = ["critic"]
        obs_groups = resolve_obs_groups(self.env.dummy_obs, obs_groups, default_sets)
        self.actor_critic = ActorCritic(
            obs=self.env.dummy_obs,
            obs_groups=obs_groups,
            num_actions=action_dim,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            ).to(self.device)
        self.policy_learning_iteration = 0
        if resume_path is not None:
            print(f"[Prepare Policy] Loading policy from {resume_path}.")
            loaded_dict = torch.load(resume_path)
            self.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
            self.policy_learning_iteration = loaded_dict["iter"]

        
    def prepare_algorithm(self, num_learning_epochs, num_mini_batches, clip_param, gamma, lam, value_loss_coef, entropy_coef, learning_rate, max_grad_norm, use_clipped_value_loss, schedule, desired_kl):
        self.alg = PPO(
            self.actor_critic,
            num_learning_epochs,
            num_mini_batches,
            clip_param,
            gamma,
            lam,
            value_loss_coef,
            entropy_coef,
            learning_rate,
            max_grad_norm,
            use_clipped_value_loss,
            schedule,
            desired_kl,
            device=self.device,
            )

    
    def train_policy(self, log_dir, num_steps_per_env, save_interval, max_iterations, export_dir, logger=None):
        print(f"[Train Policy] Training policy for {max_iterations} iterations.")
        policy_training = PolicyTraining(
            log_dir,
            env=self.env,
            alg=self.alg,
            device=self.device,
            num_steps_per_env=num_steps_per_env,
            save_interval=save_interval,
            max_iterations=max_iterations,
            export_dir=export_dir,
            logger=logger,
            )
        policy_training.current_learning_iteration = self.policy_learning_iteration
        policy_training.learn()
        self.policy_learning_iteration += policy_training.max_iterations


def run_experiment(config: BaseConfig, logger: RunLogger):
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_num}"
    log_dir = os.path.join(args_cli.log_root or "logs", config.experiment_name, args_cli.task, run_name)
    logger.set_run_name(run_name)
    os.makedirs(log_dir, exist_ok=True)
    model_experiment = ModelBasedExperiment(**config.experiment_config.to_dict())
    model_experiment.prepare_environment(**config.environment_config.to_dict())
    model_experiment.prepare_model(**config.model_architecture_config.to_dict())
    model_experiment.prepare_policy(**config.policy_architecture_config.to_dict())
    model_experiment.prepare_algorithm(**config.policy_algorithm_config.to_dict())
    model_experiment.prepare_data(**config.data_config.to_dict())
    model_experiment.train_policy(
        log_dir, **config.policy_training_config.to_dict(), logger=logger
    )
    print(f"Training completed. Policy saved to {log_dir}.")


def run(config: BaseConfig, logger: RunLogger):
    run_experiment(config, logger)


def resolve_task_config(task: str):
    if task == "anymal_d_flat":
        config = AnymalDFlatConfig()
        return config
    else:
        raise ValueError(f"Unknown task: {task}")


def _apply_cli_overrides(config: BaseConfig) -> None:
    """Apply the run-shape CLI overrides (device, iterations, smoke) to *config*."""
    config.experiment_config.device = args_cli.device
    if args_cli.max_iterations is not None:
        config.policy_training_config.max_iterations = args_cli.max_iterations
    if args_cli.smoke:
        config.policy_training_config.max_iterations = 1
        config.environment_config.num_envs = min(config.environment_config.num_envs, 64)
        config.environment_config.max_episode_length = min(config.environment_config.max_episode_length, 32)
        config.data_config.file_data_size = min(config.data_config.file_data_size, 1000)
        config.data_config.batch_data_size = min(config.data_config.batch_data_size, 1000)
        config.policy_training_config.num_steps_per_env = min(
            config.policy_training_config.num_steps_per_env, 8
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline (pure PyTorch) policy training with a pretrained world model.")
    parser.add_argument("--task", type=str, default="anymal_d_flat", help="Task to use for the experiment.")
    parser.add_argument("--run_num", type=int, default=None, help="Run number for the experiment on the cluster.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for the offline run.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Override the PPO iteration count.")
    parser.add_argument("--smoke", action="store_true", default=False, help="Small acceptance run (1 iteration).")
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb", "none"], help="Logging backend."
    )
    parser.add_argument("--log_root", type=str, default=None, help="Root directory for the run logs.")
    args_cli = parser.parse_args()
    config = resolve_task_config(args_cli.task)
    _apply_cli_overrides(config)
    logger = RunLogger(
        args_cli.logger,
        log_dir=os.path.join(args_cli.log_root or "logs", config.experiment_name, args_cli.task),
        project=config.experiment_name,
        config=config.to_dict(),
    )
    try:
        run(config, logger)
    finally:
        logger.close()
