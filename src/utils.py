import torch
import torch.nn.functional as F


def top_k_sampling(
    logits: torch.tensor, k=10
) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    logits_probs = F.softmax(logits, dim=-1)

    values, indices = torch.topk(logits, k)
    masked_logits = logits.clone().masked_fill(
        ~torch.isin(logits, values), float("-inf")
    )

    topk_probs = F.softmax(masked_logits, dim=-1)

    next_token = torch.multinomial(topk_probs, num_samples=1)
    return next_token, logits_probs[next_token], topk_probs[next_token]
