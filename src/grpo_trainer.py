import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image
from functools import partial
from tqdm import tqdm

import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from src.config import Config
from src.episode import Episode
from src.episode_batch_repeat_sampler import EpisodeBatchRepeatSampler
from src.episode_dataset import EpisodeDataset
from src.policy.policy_base import PolicyBaseModel
from src.rl_data_record import RLDataRecord
from src.utils import get_color, to_device_collate, top_k_sampling
from src.reward_model import RewardModel


class GRPOTrainer:
    def __init__(
        self,
        config: Config,
        policy: PolicyBaseModel,
        reward_model: RewardModel,
    ):
        assert config is not None
        assert policy is not None
        assert reward_model is not None

        self.config = config
        self.policy = policy
        self.policy.to(config.device)
        self.reward_model = reward_model

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.lr, weight_decay=0.01
        )
        self.writer = SummaryWriter()

    def get_data_loader(self, dataset: EpisodeDataset) -> DataLoader:
        assert dataset is not None

        if dataset.split == "TRAIN":
            batch_size = self.config.train_batch_size
        elif dataset.split == "TEST":
            batch_size = self.config.test_batch_size
        else:
            assert dataset.split == "EVAL"
            batch_size = self.config.eval_batch_size

        batch_repeat_sampler = EpisodeBatchRepeatSampler(
            dataset=dataset,
            batch_size=batch_size,
            group_size=self.config.episode_group_size,
            repeats=self.config.episode_steps,
        )
        # print(f"batch_repeat_sampler: {list(batch_repeat_sampler)}")

        to_device_collate_configurable = partial(to_device_collate, self.config.device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size * self.config.episode_group_size,
            sampler=batch_repeat_sampler,
            # pin_memory=True,
            collate_fn=to_device_collate_configurable,
        )
        return dataloader

    def compute_batch_rewards(
        self,
        batch_rl_data_record: RLDataRecord,
        batch_episode_indices: list[int],
        dataset: EpisodeDataset,
        step: int,
        write_tensorboard: bool = True,
    ) -> torch.tensor:
        assert batch_rl_data_record is not None
        assert batch_episode_indices is not None
        assert len(batch_episode_indices) > 0
        assert dataset is not None

        # 1. get target episodes
        batch_episodes_rewards = self.reward_model.reward(
            batach_cur_pos=batch_rl_data_record.batch_agent_current_pos,
            batch_target_pos=batch_rl_data_record.batch_agent_target_pos,
        )
        # print(f"episodes: {batch_episodes_rewards}")

        # B x G
        batch_group_episodes_rewards = batch_episodes_rewards.view(
            -1,
            self.config.episode_group_size,
        )
        # 1
        batch_mean_group_episode_rewards = torch.mean(batch_group_episodes_rewards)
        if write_tensorboard:
            self.writer.add_scalar(
                f"{dataset.split}:batch_mean_group_episode_rewards",
                batch_mean_group_episode_rewards,
                step,
            )

        # B x 1
        batch_group_r_std, batch_group_r_mean = torch.std_mean(
            batch_group_episodes_rewards, dim=1, keepdim=True
        )
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_r_std", batch_group_r_std, step
            )
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_r_mean", batch_group_r_mean, step
            )
        # if torch.isnan(batch_group_r_std) or batch_group_r_std == 0.0:
        #     # print(f"invalid episode, batch_group_r_std: {r_std}")
        #     return

        batch_group_r_std = torch.nan_to_num(batch_group_r_std, nan=1.0)
        batch_group_r_std[batch_group_r_std == 0.0] = 1.0

        # B x G
        batch_group_reward_advantages = (
            batch_group_episodes_rewards - batch_group_r_mean
        ) / batch_group_r_std
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_reward_advantages",
                batch_group_reward_advantages,
                step,
            )
        # print(f"batch_group_r_std: {batch_group_r_std}, r_mean: {r_mean}, r_advantages: {r_advantages}")

        # 3. Compute the KL-
        # N/A

        # 4. Compute weighted rewards
        # B x G
        batch_group_advantage_weighted_rewards = (
            batch_group_episodes_rewards * batch_group_reward_advantages
        )
        # print(f"batch_group_advantage_weighted_rewards: {batch_group_advantage_weighted_rewards}")

        # S x B x G
        batch_group_episode_log_probs = (
            batch_rl_data_record.batch_logit_prob_history.view(
                self.config.episode_steps,
                -1,
                self.config.episode_group_size,
            )
        )
        batch_group_episode_log_probs = torch.sum(
            torch.log(batch_group_episode_log_probs), dim=0
        )
        # print(f"episode_log_probs: {episode_log_probs}")
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_episode_log_probs",
                batch_group_episode_log_probs,
                step,
            )

        # S x B x G
        batch_group_episode_probs = batch_rl_data_record.batch_top_k_prob_history.view(
            self.config.episode_steps,
            -1,
            self.config.episode_group_size,
        )
        # print(f"batch_group_episode_probs: {batch_group_episode_probs}")
        batch_group_episode_probs = torch.sum(batch_group_episode_probs, dim=0)
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_episode_probs",
                batch_group_episode_probs,
                step,
            )

        # B x G
        batch_group_episode_weighted_rewards = (
            batch_group_episodes_rewards * batch_group_episode_log_probs
        )  # episode_probs
        # print(f"batch_group_episode_weighted_rewards: {batch_group_episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_episode_weighted_rewards",
                batch_group_episode_weighted_rewards,
                step,
            )

        # 1
        batch_mean_episode_weighted_rewards = torch.mean(
            batch_group_episode_weighted_rewards
        )
        # print(f"batch_mean_episode_weighted_rewards: {batch_mean_episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_scalar(
                f"{dataset.split}:batch_mean_episode_weighted_rewards",
                batch_mean_episode_weighted_rewards,
                step,
            )

        # B x G
        batch_group_reward_weighted_advantages = -(
            batch_group_reward_advantages * batch_group_episode_log_probs
        )  # episode_probs
        # print(f"batch_group_episode_weighted_rewards: {batch_group_episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_histogram(
                f"{dataset.split}:batch_group_reward_weighted_advantages",
                batch_group_reward_weighted_advantages,
                step,
            )

        # 1
        batch_mean_group_reward_weighted_advantages = torch.mean(
            batch_group_reward_weighted_advantages
        )
        # print(f"batch_mean_episode_weighted_rewards: {batch_mean_episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_scalar(
                f"{dataset.split}:batch_mean_group_reward_weighted_advantages",
                batch_mean_group_reward_weighted_advantages,
                step,
            )

        return batch_mean_group_reward_weighted_advantages

    def eval_policy(
        self,
        batch_episode_indices: list[int],
        batch_rl_data_record: RLDataRecord,
        dataset: EpisodeDataset,
        step: int,
    ):
        assert batch_episode_indices is not None
        assert len(batch_episode_indices) > 0
        assert dataset is not None
        assert dataset.split == "EVAL"

        # Compute metrics and gradient
        batch_mean_episode_weighted_rewards: torch.tensor = self.compute_batch_rewards(
            batch_rl_data_record=batch_rl_data_record,
            batch_episode_indices=batch_episode_indices,
            dataset=dataset,
            step=step,
        )

    def train_policy(
        self,
        batch_episode_indices: list[int],
        batch_rl_data_record: RLDataRecord,
        dataset: EpisodeDataset,
        step: int,
    ):
        assert batch_episode_indices is not None
        assert len(batch_episode_indices) > 0
        assert dataset is not None
        assert dataset.split == "TRAIN"
        assert self.policy.training

        # Compute metrics and gradient
        batch_mean_group_reward_weighted_advantages: torch.tensor = (
            self.compute_batch_rewards(
                batch_rl_data_record=batch_rl_data_record,
                batch_episode_indices=batch_episode_indices,
                dataset=dataset,
                step=step,
            )
        )

        # Optimization
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()
        batch_mean_group_reward_weighted_advantages.backward()

        # Adjust learning weights
        self.optimizer.step()

        # for name, param in self.policy.brain.named_parameters():
        #     if param.grad is not None and param.grad.size(0) > 0:
        #         self.writer.add_histogram(f"{name}.grad", param.grad, step)

    def train_policy_with_cleanup(
        self,
        batch_episode_indices: list[int],
        batch_rl_data_record: RLDataRecord,
        dataset: EpisodeDataset,
        step: int,
        debug: bool,
    ):
        try:
            self.train_policy(
                batch_episode_indices=batch_episode_indices,
                batch_rl_data_record=batch_rl_data_record,
                dataset=dataset,
                step=step,
            )
        finally:
            target_episodes = dataset.get_episods(
                batch_episode_indices=batch_episode_indices
            )
            for idx, episode in enumerate(target_episodes):
                if idx == 0 and debug:
                    # plt.ioff()
                    # only viz the 1st episode
                    # avoid too much data
                    fig = plt.figure(figsize=self.config.figure_size)
                    ax = fig.add_subplot(1, 1, 1)
                    episode.viz(
                        ax=ax, reward_model=self.reward_model, color=get_color(0)
                    )
                    plot_buf = io.BytesIO()
                    plt.savefig(plot_buf, format="jpeg")
                    plot_buf.seek(0)
                    image = PIL.Image.open(plot_buf)
                    image = ToTensor()(image)
                    self.writer.add_image(
                        f"Train Episod: {episode.episode_id}", image, step
                    )
                    # plt.ion()

                episode.reset()

    def eval_policy_with_cleanup(
        self,
        batch_episode_indices: list[int],
        batch_rl_data_record: RLDataRecord,
        dataset: EpisodeDataset,
        epoch: int,
        batch_idx: int,
        step: int,
        debug: bool,
    ):
        try:
            self.eval_policy(
                batch_episode_indices=batch_episode_indices,
                batch_rl_data_record=batch_rl_data_record,
                dataset=dataset,
                step=step,
            )
        finally:
            target_episodes = dataset.get_episods(
                batch_episode_indices=batch_episode_indices
            )
            for idx, episode in enumerate(target_episodes):
                if idx == 0:  # viz the 1st batch 1st item
                    backend_ = mpl.get_backend()
                    mpl.use("Agg")  # Prevent showing stuff

                    # only viz the 1st episode
                    # avoid too much data
                    fig, axes = plt.subplots(
                        nrows=1,
                        ncols=3,
                        figsize=self.config.triple_figure_size,
                    )
                    episode.viz(
                        ax=axes[0],
                        reward_model=self.reward_model,
                        color=get_color(0),
                    )
                    episode.viz_fov(
                        ax=axes[1],
                    )
                    batch_rl_data_record.viz_fov(
                        ax=axes[2], idx=idx, reward_model=self.reward_model
                    )
                    plot_buf = io.BytesIO()
                    plt.savefig(plot_buf, format="jpeg")
                    plot_buf.seek(0)
                    image = PIL.Image.open(plot_buf)
                    image = ToTensor()(image)
                    self.writer.add_image(
                        f"Eval Episod: {epoch}:{episode.episode_id}", image, step
                    )

                    mpl.use(backend_)  # Reset backend

                episode.reset()

    def execute_1_episode_step(
        self,
        mode: str,
        batch_rl_data_record: RLDataRecord,
        dataset: EpisodeDataset,
        step: int,
        debug: bool = False,
    ):
        assert mode in ["TRAIN", "EVAL"]
        assert batch_rl_data_record is not None
        assert dataset is not None

        # Flaten the fov
        batch_logits = self.policy(batch_rl_data_record=batch_rl_data_record)
        batch_action_idx, batch_logit_prob, batch_top_k_prob = top_k_sampling(
            logits=batch_logits, k=self.config.top_k if mode == "TRAIN" else 1
        )
        # print(
        #     f"batch_action_idx: {batch_action_idx}, batch_logit_prob: {batch_logit_prob}, batch_top_k_prob: {batch_top_k_prob}"
        # )
        batch_rl_data_record.update_step(
            batch_action_idx=batch_action_idx,
            batch_logit_prob=batch_logit_prob,
            batch_top_k_prob=batch_top_k_prob,
            step=step,
            debug=debug,
        )
        return batch_rl_data_record.current_batch_episode_idx

    def _run_eval(
        self,
        epoch: int,
        dataset: EpisodeDataset,
        dataloader: DataLoader,
        step: int,
        debug: bool,
    ):
        assert epoch >= 0
        assert step >= 0
        assert dataset is not None
        assert dataset.split == "EVAL"
        assert dataloader is not None

        self.policy.eval()
        with torch.no_grad():
            with tqdm(dataloader, desc=f"Epoch {epoch + 1}") as t:
                last_batch_episode_idx = None
                batch_rl_data_record = None
                for batch_idx, batch_data_items in enumerate(t):
                    # step = epoch * len(train_dataloader) + batch_idx
                    if batch_rl_data_record is None:
                        batch_rl_data_record = RLDataRecord(
                            config=self.config, batch_data_items=batch_data_items
                        )

                    current_batch_episode_idx = self.execute_1_episode_step(
                        mode="EVAL",
                        batch_rl_data_record=batch_rl_data_record,
                        dataset=dataset,
                        step=((batch_idx + 1) % self.config.episode_steps),
                        debug=debug,
                    )

                    if last_batch_episode_idx is None:
                        last_batch_episode_idx = current_batch_episode_idx

                    is_episode_step_done = (
                        batch_idx + 1
                    ) % self.config.episode_steps == 0
                    if is_episode_step_done:
                        assert last_batch_episode_idx == current_batch_episode_idx
                        # print(
                        #     f"optmized model. current_batch_episode_idx: {current_batch_episode_idx}"
                        # )
                        step += 1
                        self.eval_policy_with_cleanup(
                            batch_episode_indices=current_batch_episode_idx,
                            batch_rl_data_record=batch_rl_data_record,
                            dataset=dataset,
                            epoch=epoch,
                            batch_idx=batch_idx,
                            step=step,
                            debug=debug,
                        )

                        # reset the rl data record for next "TRUE" batch
                        batch_rl_data_record = None

                        # reset last_batch_episode_idx
                        last_batch_episode_idx = None

                        # Update progress bar description (optional)
                        t.set_postfix(
                            {
                                "split": dataset.split,
                                "step": step,
                                "batch_idx": batch_idx,
                                "is_episode_step_done": is_episode_step_done,
                                # "current_batch_episode_idx": current_batch_episode_idx,
                            }
                        )

        return step

    def _run_train(
        self,
        dataset: EpisodeDataset,
        dataloader: DataLoader,
        epoch: int,
        step: int,
        profile: torch.profiler.profile,
        debug: bool,
    ) -> int:
        assert dataset is not None
        assert dataset.split == "TRAIN"
        assert dataloader is not None

        with tqdm(dataloader, desc=f"Epoch {epoch + 1}") as t:
            last_batch_episode_idx = None
            batch_rl_data_record = None
            for batch_idx, batch_data_items in enumerate(t):
                # step = epoch * len(dataloader) + batch_idx
                if batch_rl_data_record is None:
                    batch_rl_data_record = RLDataRecord(
                        config=self.config, batch_data_items=batch_data_items
                    )

                current_batch_episode_idx = self.execute_1_episode_step(
                    mode="TRAIN",
                    batch_rl_data_record=batch_rl_data_record,
                    dataset=dataset,
                    step=((batch_idx + 1) % self.config.episode_steps),
                    debug=debug,
                )

                if last_batch_episode_idx is None:
                    last_batch_episode_idx = current_batch_episode_idx

                is_episode_step_done = (batch_idx + 1) % self.config.episode_steps == 0
                if is_episode_step_done:
                    assert last_batch_episode_idx == current_batch_episode_idx
                    # print(
                    #     f"optmized model. current_batch_episode_idx: {current_batch_episode_idx}"
                    # )
                    step += 1
                    self.train_policy_with_cleanup(
                        batch_episode_indices=current_batch_episode_idx,
                        batch_rl_data_record=batch_rl_data_record,
                        dataset=dataset,
                        step=step,
                        debug=debug,
                    )

                    # reset the rl data record for next "TRUE" batch
                    batch_rl_data_record = None

                    # reset last_batch_episode_idx
                    last_batch_episode_idx = None

                    # Profile one step
                    if profile is not None:
                        profile.step()

                # Update progress bar description (optional)
                t.set_postfix(
                    {
                        "split": dataset.split,
                        "step": step,
                        "batch_idx": batch_idx,
                        "is_episode_step_done": is_episode_step_done,
                        # "current_batch_episode_idx": current_batch_episode_idx,
                    }
                )

            if last_batch_episode_idx is not None:
                assert batch_rl_data_record is not None
                # print(f"optmized model. last_batch_episode_idx: {last_batch_episode_idx}")
                step += 1

                self.train_policy_with_cleanup(
                    batch_episode_indices=last_batch_episode_idx,
                    batch_rl_data_record=batch_rl_data_record,
                    dataset=dataset,
                    step=step,
                    debug=debug,
                )
        return step

    def run(
        self,
        train_dataset: EpisodeDataset,
        eval_dataset: EpisodeDataset,
        run_profile: bool = False,
        debug: bool = False,
    ):
        assert train_dataset is not None
        assert len(train_dataset) > 0
        assert train_dataset.split == "TRAIN"
        assert eval_dataset is not None
        assert len(eval_dataset) > 0
        assert eval_dataset.split == "EVAL"

        train_dataloader = self.get_data_loader(dataset=train_dataset)
        eval_dataloader = self.get_data_loader(dataset=eval_dataset)

        print(
            f"train_dataloader: {len(train_dataloader)}, eval_dataloader: {len(eval_dataloader)}"
        )

        def _run_internal(profile: torch.profiler.profile = None):
            # torch.autograd.set_detect_anomaly(True)
            self.policy.train()
            eval_step = 0
            train_step = 0
            for epoch in range(self.config.epoches):
                train_step = self._run_train(
                    dataset=train_dataset,
                    dataloader=train_dataloader,
                    epoch=epoch,
                    step=train_step,
                    profile=profile,
                    debug=debug,
                )

                # Eval - each epoch
                try:
                    eval_step = self._run_eval(
                        dataset=eval_dataset,
                        dataloader=eval_dataloader,
                        epoch=epoch,
                        step=eval_step,
                        debug=debug,
                    )
                finally:
                    # reset to the train mode
                    self.policy.train()

        if not run_profile:
            _run_internal()
        else:
            # Use the profiler to analyze the execution
            activities = [
                torch.profiler.ProfilerActivity.CPU,
            ]
            with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./runs/profile"
                ),
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
                with_modules=True,
            ) as profile:
                _run_internal(profile=profile)
