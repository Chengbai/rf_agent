import io
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

        self.optimizer = torch.optim.AdamW(policy.parameters(), lr=1, weight_decay=0.01)
        self.writer = SummaryWriter()

    def get_data_loader(self, dataset: EpisodeDataset) -> DataLoader:
        train_batch_repeat_sampler = EpisodeBatchRepeatSampler(
            dataset=dataset,
            batch_size=self.config.train_batch_size,
            repeats=self.config.episode_steps,
        )
        print(f"train_batch_repeat_sampler: {list(train_batch_repeat_sampler)}")

        to_device_collate_configurable = partial(to_device_collate, self.config.device)

        train_dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            sampler=train_batch_repeat_sampler,
            pin_memory=True,
            collate_fn=to_device_collate_configurable,
        )
        return train_dataloader

    def optimized_policy(
        self,
        batch_episode_idices: list[int],
        dataset: EpisodeDataset,
        step: int,
    ):
        assert batch_episode_idices is not None
        assert len(batch_episode_idices) > 0
        assert dataset is not None
        assert dataset.split == "TRAIN"

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
        self.writer.add_histogram(f"episode_log_probs", episode_log_probs, step)

        episode_probs = torch.concat(
            [episode.reward_prob() for episode in target_episodes]
        )
        # print(f"episode_probs: {episode_probs}")
        self.writer.add_histogram(f"episode_probs", episode_probs, step)

        episode_weighted_rewards = (
            advantage_weighted_rewards * episode_log_probs
        )  # episode_probs
        # print(f"episode_weighted_rewards: {episode_weighted_rewards}")
        self.writer.add_histogram(
            f"episode_weighted_rewards", episode_weighted_rewards, step
        )

        mean_episode_weighted_rewards = torch.mean(episode_weighted_rewards)
        # print(f"mean_episode_weighted_rewards: {mean_episode_weighted_rewards}")
        self.writer.add_scalar(
            "mean_episode_weighted_rewards", mean_episode_weighted_rewards, step
        )

        # 5. optimization
        # Zero your gradients for every batch!
        self.optimizer.zero_grad()
        mean_episode_weighted_rewards.backward()
        # for name, param in policy.brain[0].named_parameters():
        #     print(name, param)

        # Adjust learning weights
        self.optimizer.step()

        for name, param in self.policy.brain.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f"{name}.grad", param.grad, step)

    def optimized_policy_with_cleanup(
        self,
        batch_episode_idices: list[int],
        dataset: EpisodeDataset,
        step: int,
        debug: bool,
    ):
        try:
            self.optimized_policy(
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
                    # only viz the 1st episode
                    # avoid too much data
                    fig = plt.figure(figsize=self.config.figure_size)
                    ax = fig.add_subplot(1, 1, 1)
                    episode.viz(ax=ax, color=get_color(0))
                    plot_buf = io.BytesIO()
                    plt.savefig(plot_buf, format="jpeg")
                    plot_buf.seek(0)
                    image = PIL.Image.open(plot_buf)
                    image = ToTensor()(image)
                    self.writer.add_image(f"Episod: {episode.id}", image, step)
                episode.reset()

    def train(
        self,
        dataset: EpisodeDataset,
        debug: bool = False,
    ):
        assert dataset is not None
        assert len(dataset) > 0

        train_dataloader = self.get_data_loader(dataset=dataset)

        step = 0

        for epoch in range(self.config.epoches):
            last_episode_idx = None
            for batch_idx, batch_data_items in enumerate(train_dataloader):
                # step = epoch * len(train_dataloader) + batch_idx

                # to be optimized `episode_idx`
                last_batch_episode_idx = batch_data_items["episode_idx"]
                last_batch_agent_start_pos = batch_data_items["agent_start_pos"]
                last_batch_agent_target_pos = batch_data_items["agent_target_pos"]
                last_batch_agent_current_pos = batch_data_items["agent_current_pos"]

                if debug:
                    print(
                        f"last_batch_episode_idx: {last_batch_episode_idx}, last_batch_agent_start_pos: {last_batch_agent_start_pos}, last_batch_agent_target_pos: {last_batch_agent_target_pos}, last_batch_agent_current_pos: {last_batch_agent_current_pos}"
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

                # feature: [B, current_pos, target_pos, fov]
                features = torch.concat(
                    [
                        last_batch_agent_current_pos,
                        last_batch_agent_target_pos,
                        batch_fov,
                    ],
                    dim=1,
                )
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

                if (batch_idx + 1) % self.config.episode_steps == 0:
                    # print(
                    #     f"optmized model. last_batch_episode_idx: {last_batch_episode_idx}"
                    # )
                    step += 1
                    self.optimized_policy_with_cleanup(
                        batch_episode_idices=last_batch_episode_idx,
                        dataset=dataset,
                        step=step,
                        debug=debug,
                    )
                    # reset last_batch_episode_idx
                    last_batch_episode_idx = None

                # break

            if last_batch_episode_idx is not None:
                # print(f"optmized model. last_batch_episode_idx: {last_batch_episode_idx}")
                step += 1

                self.optimized_policy_with_cleanup(
                    batch_episode_idices=last_batch_episode_idx,
                    dataset=dataset,
                    step=step,
                    debug=debug,
                )
