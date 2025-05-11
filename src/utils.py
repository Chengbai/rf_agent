import torch
import torch.nn.functional as F


def normalize_min_max(batch_logits):
    """
    Normalizes a batch of tensors to the range [0, 1] using min-max scaling.

    Args:
        batch_tensor (torch.Tensor): A tensor of shape (batch_size, *).

    Returns:
        torch.Tensor: A normalized tensor of the same shape.
    """
    min_values = batch_logits.min(dim=-1, keepdim=True)[0]
    # print(f"min_value: {min_values}")

    max_values = batch_logits.max(dim=-1, keepdim=True)[0]
    # print(f"max_values: {max_values}")

    norm_logits = (batch_logits - min_values) / (max_values - min_values)
    # print(norm_logits)

    norm_batch_logits = torch.nan_to_num(norm_logits, nan=1.0)

    return norm_batch_logits


def top_k_sampling(
    logits: torch.tensor, k: int = 10
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    # normailize the logits
    # logits_mean, logits_std = torch.std_mean(logits)
    # logits = (logits - logits_mean) / logits_std
    # print(f"logits: {logits}")
    norm_logits = normalize_min_max(logits)
    # print(f"norm_logits: {norm_logits}")
    logits_probs = F.softmax(norm_logits, dim=-1)
    # print(f"logits_probs: {logits_probs}")

    values, indices = torch.topk(norm_logits, k, dim=-1)
    # print(f"topk - k: {k}, values: {values}, indices: {indices}")
    # masked_logits = logits.clone().masked_fill(
    #     ~torch.isin(logits, values, dim=-1), float("-inf")
    # )
    # print(f"masked_logits: {masked_logits}")
    topk_probs = F.softmax(values, dim=-1)
    # print(f"topk_probs: {topk_probs}")

    next_token = torch.multinomial(topk_probs, num_samples=1)
    # print(f"next_token: {next_token}")
    return (
        indices[torch.arange(next_token.size(0)).unsqueeze(1), next_token],
        logits_probs[
            torch.arange(logits_probs.size(0)).unsqueeze(1),
            indices[torch.arange(next_token.size(0)).unsqueeze(1), next_token],
        ],
        topk_probs[torch.arange(next_token.size(0)).unsqueeze(1), next_token],
    )


def get_color(idx: int):
    colors = ["red", "green", "blue", "gray"]
    return colors[idx % len(colors)]


def to_device_collate(device, batch: list[dict]):
    batched_data = {}
    for key in batch[0].keys():
        # Check if elements are tensors, otherwise try to stack them directly
        if isinstance(batch[0][key], torch.Tensor):
            batched_data[key] = torch.stack([item[key].to(device) for item in batch])
        else:
            batched_data[key] = [item[key] for item in batch]
    return batched_data
