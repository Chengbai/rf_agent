import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from src.config import Config
from src.episode import Episode
from src.episode_batch_reat_sampler import EpisodeBatchRepeatSampler
from src.episode_dataset import EpisodeDataset
from src.policy import Policy
from src.utils import get_color, to_device_collate, top_k_sampling
from src.reward_model import RewardModel


class GRPOTrainer:
    def __init__(
        self,
        config: Config,
        policy: Policy,
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
            repeats=self.config.episode_steps,
        )
        print(f"batch_repeat_sampler: {list(batch_repeat_sampler)}")

        to_device_collate_configurable = partial(to_device_collate, self.config.device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=batch_repeat_sampler,
            pin_memory=True,
            collate_fn=to_device_collate_configurable,
        )
        return dataloader

    def compute_batch_rewards(
        self,
        batch_episode_idices: list[int],
        dataset: EpisodeDataset,
        step: int,
        write_tensorboard: bool = True,
    ) -> torch.tensor:
        assert batch_episode_idices is not None
        assert len(batch_episode_idices) > 0
        assert dataset is not None

        # 1. get target episodes
        target_episodes = dataset.get_episods(batch_episode_idices=batch_episode_idices)

        # 2. Compute the Advantages
        episodes_rewards = torch.tensor(
            [
                episode.reward(reward_model=self.reward_model)
                for episode in target_episodes
            ],
            device=self.config.device,
        )
        # print(f"episodes rewards: {episodes_rewards}")

        r_std, r_mean = torch.std_mean(episodes_rewards)
        if write_tensorboard:
            self.writer.add_scalar("r_std", r_std, step)
            self.writer.add_scalar("r_mean", r_mean, step)
        if torch.isnan(r_std) or r_std == 0.0:
            # print(f"invalid episode, r_std: {r_std}")
            return

        r_advantages = (episodes_rewards - r_mean) / r_std
        # print(f"r_std: {r_std}, r_mean: {r_mean}, r_advantages: {r_advantages}")

        # 3. Compute the KL-
        # N/A

        # 4. Compute weighted rewards
        advantage_weighted_rewards = episodes_rewards * r_advantages
        # print(f"advantage_weighted_rewards: {advantage_weighted_rewards}")

        episode_log_probs = torch.concat(
            [episode.log_reward_prob() for episode in target_episodes]
        ).to(self.config.device)
        # print(f"episode_log_probs: {episode_log_probs}")
        if write_tensorboard:
            self.writer.add_histogram(f"episode_log_probs", episode_log_probs, step)

        episode_probs = torch.concat(
            [episode.reward_prob() for episode in target_episodes]
        )
        # print(f"episode_probs: {episode_probs}")
        if write_tensorboard:
            self.writer.add_histogram(f"episode_probs", episode_probs, step)

        episode_weighted_rewards = (
            advantage_weighted_rewards * episode_log_probs
        )  # episode_probs
        # print(f"episode_weighted_rewards: {episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_histogram(
                f"episode_weighted_rewards", episode_weighted_rewards, step
            )

        mean_episode_weighted_rewards = torch.mean(episode_weighted_rewards)
        # print(f"mean_episode_weighted_rewards: {mean_episode_weighted_rewards}")
        if write_tensorboard:
            self.writer.add_scalar(
                "mean_episode_weighted_rewards", mean_episode_weighted_rewards, step
            )
        return mean_episode_weighted_rewards

    def train_policy(
        self,
        batch_episode_idices: list[int],
        dataset: EpisodeDataset,
        step: int,
    ):
        assert batch_episode_idices is not None
        assert len(batch_episode_idices) > 0
        assert dataset is not None
        assert dataset.split == "TRAIN"

        # Compute metrics and gradient
        mean_episode_weighted_rewards: torch.tensor = self.compute_batch_rewards(
            batch_episode_idices=batch_episode_idices, dataset=dataset, step=step
        )

        # Optimization
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()
        mean_episode_weighted_rewards.backward()

        # Adjust learning weights
        self.optimizer.step()

        for name, param in self.policy.brain.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"{name}.grad", param.grad, step)

    def train_policy_with_cleanup(
        self,
        batch_episode_idices: list[int],
        dataset: EpisodeDataset,
        step: int,
        debug: bool,
    ):
        try:
            self.train_policy(
                batch_episode_idices=batch_episode_idices,
                dataset=dataset,
                step=step,
            )
        finally:
            target_episodes = dataset.get_episods(
                batch_episode_idices=batch_episode_idices
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
                    self.writer.add_image(f"Train Episod: {episode.id}", image, step)
                    # plt.ion()

                episode.reset()

    def run_1_batch(
        self,
        batch_idx: list,
        batch_data_items: dict,
        dataset: EpisodeDataset,
        debug: bool = False,
    ):
        assert batch_idx >= 0
        assert batch_data_items
        assert dataset is not None

        # to be optimized `episode_idx`
        current_batch_episode_idx = batch_data_items["episode_idx"]
        last_batch_agent_start_pos = batch_data_items["agent_start_pos"]
        last_batch_agent_target_pos = batch_data_items["agent_target_pos"]
        last_batch_agent_current_pos = batch_data_items["agent_current_pos"]

        if debug:
            print(
                f"current_batch_episode_idx: {current_batch_episode_idx}, last_batch_agent_start_pos: {last_batch_agent_start_pos}, last_batch_agent_target_pos: {last_batch_agent_target_pos}, last_batch_agent_current_pos: {last_batch_agent_current_pos}"
            )
        # print(
        #     f"epoch: {epoch}, batch_idx: {batch_idx}, episode_idx: {batch_data_items['episode_idx']}"
        # )

        # print(
        #     {
        #         k: v.size()
        #         for k, v in batch_data_items.items()
        #         if isinstance(v, torch.Tensor)
        #     }
        # )
        batch_fov = batch_data_items["fov"]
        B, FOV_H, FOV_W = batch_fov.shape

        batch_fov = batch_fov.reshape(shape=(B, -1))

        # feature: [B, fov]
        features = batch_fov
        # feature: [B, current_pos, target_pos, fov]
        # features = torch.concat(
        #     [
        #         last_batch_agent_current_pos,
        #         last_batch_agent_target_pos,
        #         batch_fov,
        #     ],
        #     dim=1,
        # )
        batch_logits = self.policy(features)
        # print(f"batch_logits: {batch_logits.shape}")
        batch_action_idx, batch_logit_prob, batch_top_k_prob = top_k_sampling(
            logits=batch_logits, k=self.config.top_k
        )
        # print(
        #     f"batch_action_idx: {batch_action_idx}, batch_logit_prob: {batch_logit_prob}, batch_top_k_prob: {batch_top_k_prob}"
        # )
        dataset.update_step(
            batch_episode_idices=batch_data_items["episode_idx"],
            batch_action_idx=batch_action_idx,
            batch_logit_prob=batch_logit_prob,
            batch_top_k_prob=batch_top_k_prob,
        )
        return current_batch_episode_idx

    def eval_policy(
        self,
        dataset: EpisodeDataset,
        dataloader: DataLoader,
        epoch: int,
        step: int,
        debug: bool,
    ):
        self.policy.eval()
        with torch.no_grad():
            eval_mean_episode_weighted_rewards = torch.tensor(0.0)
            for batch_idx, batch_data_items in enumerate(dataloader):
                try:
                    # step = epoch * len(train_dataloader) + batch_idx
                    current_batch_episode_idx = self.run_1_batch(
                        batch_idx=batch_idx,
                        batch_data_items=batch_data_items,
                        dataset=dataset,
                        debug=debug,
                    )
                    mean_episode_weighted_rewards: torch.tensor = (
                        self.compute_batch_rewards(
                            batch_episode_idices=current_batch_episode_idx,
                            dataset=dataset,
                            step=step,
                            write_tensorboard=False,
                        )
                    )

                    mean_episode_weighted_rewards = torch.nan_to_num(
                        mean_episode_weighted_rewards, nan=0.0
                    )
                    eval_mean_episode_weighted_rewards += mean_episode_weighted_rewards
                finally:
                    target_episodes = dataset.get_episods(
                        batch_episode_idices=current_batch_episode_idx
                    )
                    for idx, episode in enumerate(target_episodes):
                        if batch_idx == 0 and idx == 0:  # viz the 1st batch 1st item
                            backend_ = mpl.get_backend()
                            mpl.use("Agg")  # Prevent showing stuff

                            # only viz the 1st episode
                            # avoid too much data
                            fig, axes = plt.subplots(
                                nrows=1, ncols=2, figsize=self.config.double_figure_size
                            )
                            episode.viz(
                                ax=axes[0],
                                reward_model=self.reward_model,
                                color=get_color(0),
                            )
                            episode.viz_fov(
                                ax=axes[1],
                            )
                            plot_buf = io.BytesIO()
                            plt.savefig(plot_buf, format="jpeg")
                            plot_buf.seek(0)
                            image = PIL.Image.open(plot_buf)
                            image = ToTensor()(image)
                            self.writer.add_image(
                                f"Eval Episod: {epoch}:{episode.id}", image, step
                            )

                            mpl.use(backend_)  # Reset backend

                        episode.reset()
            self.writer.add_scalar(
                "Eval: mean_episode_weighted_rewards",
                eval_mean_episode_weighted_rewards / len(dataloader),
                step,
            )

    def run(
        self,
        train_dataset: EpisodeDataset,
        eval_dataset: EpisodeDataset,
        debug: bool = False,
    ):
        assert train_dataset is not None
        assert len(train_dataset) > 0
        assert eval_dataset is not None
        assert len(eval_dataset) > 0

        train_dataloader = self.get_data_loader(dataset=train_dataset)
        eval_dataloader = self.get_data_loader(dataset=eval_dataset)

        step = 0

        for epoch in range(self.config.epoches):
            last_batch_episode_idx = None
            for batch_idx, batch_data_items in enumerate(train_dataloader):
                # step = epoch * len(train_dataloader) + batch_idx
                current_batch_episode_idx = self.run_1_batch(
                    batch_idx=batch_idx,
                    batch_data_items=batch_data_items,
                    dataset=train_dataset,
                    debug=debug,
                )

                if last_batch_episode_idx is None:
                    last_batch_episode_idx = current_batch_episode_idx

                if (batch_idx + 1) % self.config.episode_steps == 0:
                    assert last_batch_episode_idx == current_batch_episode_idx
                    # print(
                    #     f"optmized model. current_batch_episode_idx: {current_batch_episode_idx}"
                    # )
                    step += 1
                    self.train_policy_with_cleanup(
                        batch_episode_idices=current_batch_episode_idx,
                        dataset=train_dataset,
                        step=step,
                        debug=debug,
                    )

                    # reset last_batch_episode_idx
                    last_batch_episode_idx = None

            if last_batch_episode_idx is not None:
                # print(f"optmized model. last_batch_episode_idx: {last_batch_episode_idx}")
                step += 1

                self.train_policy_with_cleanup(
                    batch_episode_idices=last_batch_episode_idx,
                    dataset=train_dataset,
                    step=step,
                    debug=debug,
                )

            # Eval - each epoch
            try:
                self.eval_policy(
                    dataset=eval_dataset,
                    dataloader=eval_dataloader,
                    step=step,
                    epoch=epoch,
                    debug=debug,
                )
            finally:
                # reset to the train mode
                self.policy.train()
