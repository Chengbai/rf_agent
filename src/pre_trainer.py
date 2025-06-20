# 2025 / 06

import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import PIL.Image
from functools import partial
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.episode import Episode
from src.episode_dataset import EpisodeRLDataset
from src.policy.policy_base import PolicyBaseModel
from src.rl_data_record import RLDataRecord
from src.utils import get_color, to_device_collate, top_k_sampling
from src.reward_model import RewardModel


class PreTrainer:
    def __init__(
        self,
        config: Config,
        policy: PolicyBaseModel,
    ):
        assert config is not None
        assert policy is not None

        self.config = config
        self.policy = policy
        self.policy.to(config.device)

        self.optimizer = torch.optim.AdamW(
            policy.parameters(), lr=config.lr, weight_decay=0.01
        )
        self.learning_rate_scheduler = None
        self.criterion = None
        self.writer = SummaryWriter()

    def get_data_loader(self, dataset: EpisodeRLDataset) -> DataLoader:
        assert dataset is not None

        shuffle = True
        if dataset.split == "TRAIN":
            batch_size = self.config.train_batch_size
        elif dataset.split == "TEST":
            shuffle = False
            batch_size = self.config.test_batch_size
        else:
            assert dataset.split == "EVAL"
            shuffle = False
            batch_size = self.config.eval_batch_size

        to_device_collate_configurable = partial(to_device_collate, self.config.device)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size * self.config.episode_group_size,
            # pin_memory=True,
            collate_fn=to_device_collate_configurable,
            shuffle=shuffle,
        )
        return dataloader

    def _run_train(
        self,
        dataset: EpisodeRLDataset,
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
            for batch_idx, batch_data in enumerate(t):
                step += 1
                batch_fov: torch.Tensor = batch_data["fov"]
                batch_cur_position: torch.Tensor = batch_data["agent_current_pos"]
                batch_target_position: torch.Tensor = batch_data["agent_target_pos"]
                batch_best_next_pos: torch.Tensor = batch_data["best_next_pos"]
                batch_best_next_action: torch.Tensor = batch_data["best_next_action"]
                batch_logits = self.policy(
                    batch_fov=batch_fov,
                    batch_cur_position=batch_cur_position,
                    batch_target_position=batch_target_position,
                )
                batch_probability = F.softmax(batch_logits, dim=1)
                if self.writer is not None:
                    self.writer.add_histogram(
                        f"{dataset.split}:batch_probability", batch_probability, step
                    )
                # print(f"batch_logits: {batch_logits.shape}")
                # print(f"batch_probability: {batch_probability.shape}")

                loss = self.criterion(batch_logits, batch_best_next_action)
                if self.writer is not None:
                    self.writer.add_scalar(
                        f"{dataset.split}:loss",
                        loss,
                        step,
                    )
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.learning_rate_scheduler.step()
                current_lr = self.learning_rate_scheduler.get_last_lr()[0]

                t.set_postfix(
                    {
                        "Epoch": epoch,
                        "Loss": f"{loss:.4f}",
                        "Lr": f"{current_lr:.6f}",
                        "split": dataset.split,
                        "step": step,
                        "batch_idx": batch_idx,
                    }
                )
        return step

    def _run_eval(
        self,
        epoch: int,
        dataset: EpisodeRLDataset,
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
                for batch_idx, batch_data in enumerate(t):
                    step += 1
                    batch_fov: torch.Tensor = batch_data["fov"]
                    batch_cur_position: torch.Tensor = batch_data["agent_current_pos"]
                    batch_target_position: torch.Tensor = batch_data["agent_target_pos"]
                    batch_best_next_pos: torch.Tensor = batch_data["best_next_pos"]
                    batch_best_next_action: torch.Tensor = batch_data[
                        "best_next_action"
                    ]
                    batch_logits = self.policy(
                        batch_fov=batch_fov,
                        batch_cur_position=batch_cur_position,
                        batch_target_position=batch_target_position,
                    )
                    batch_probability = F.softmax(batch_logits, dim=1)
                    if self.writer is not None:
                        self.writer.add_histogram(
                            f"{dataset.split}:batch_probability",
                            batch_probability,
                            step,
                        )
                    # print(f"batch_logits: {batch_logits.shape}")
                    # print(f"batch_probability: {batch_probability.shape}")

                    loss = self.criterion(batch_logits, batch_best_next_action)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            f"{dataset.split}:loss",
                            loss,
                            step,
                        )

                    # Update progress bar description (optional)
                    t.set_postfix(
                        {
                            "Epoch": epoch,
                            "Loss": f"{loss:.4f}",
                            "split": dataset.split,
                            "step": step,
                            "batch_idx": batch_idx,
                        }
                    )
        return step

    def run(
        self,
        train_dataset: EpisodeRLDataset,
        eval_dataset: EpisodeRLDataset,
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

        total_steps = self.config.epoches * len(train_dataloader)
        warmup_steps = 0
        # Cosine annealing scheduler
        self.learning_rate_scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps - warmup_steps
        )
        self.criterion = nn.CrossEntropyLoss()

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
