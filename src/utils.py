import torch
import torch.nn.functional as F


def normalize_tensor(tensor):
    """Normalizes a tensor to the range [0, 1].

    Args:
      tensor: The input tensor.

    Returns:
      The normalized tensor.
    """
    min_val = tensor.min()
    max_val = tensor.max()
    if min_val == max_val:
        normalized_tensor = torch.ones_like(tensor)
    else:
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def top_k_sampling(
    logits: torch.tensor, k: int = 10
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    # normailize the logits
    # logits_mean, logits_std = torch.std_mean(logits)
    # logits = (logits - logits_mean) / logits_std
    logits = normalize_tensor(logits)
    logits_probs = F.softmax(logits, dim=-1)

    values, indices = torch.topk(logits, k)
    # print(f"topk - k: {k}, values: {values}, indices: {indices}")
    masked_logits = logits.clone().masked_fill(
        ~torch.isin(logits, values), float("-inf")
    )
    # print(f"masked_logits: {masked_logits}")
    topk_probs = F.softmax(masked_logits, dim=-1)
    # print(f"topk_probs: {topk_probs}")

    next_token = torch.multinomial(topk_probs, num_samples=1)
    return next_token, logits_probs[next_token], topk_probs[next_token]
