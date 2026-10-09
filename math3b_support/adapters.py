from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any, Callable, Literal, TYPE_CHECKING

import torch
from torch import Tensor
from torch.utils.data import Dataset
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
import torch.nn.functional as F

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).
    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.
    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    import torch
    
    batch_size = len(prompt_strs)
    
    # Tokenize prompts and outputs separately
    prompt_tokens = []
    output_tokens = []
    
    for prompt_str, output_str in zip(prompt_strs, output_strs):
        # Tokenize prompt and output separately
        prompt_tok = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        output_tok = tokenizer(output_str, add_special_tokens=False, return_tensors="pt")["input_ids"].squeeze(0)
        
        prompt_tokens.append(prompt_tok)
        output_tokens.append(output_tok)
    
    # Concatenate prompt and output tokens for each example
    concatenated_tokens = []
    prompt_lens = []
    
    for prompt_tok, output_tok in zip(prompt_tokens, output_tokens):
        # Concatenate prompt and output
        full_sequence = torch.cat([prompt_tok, output_tok], dim=0)
        concatenated_tokens.append(full_sequence)
        prompt_lens.append(len(prompt_tok))
    
    # Find max length and pad sequences
    max_len = max(len(seq) for seq in concatenated_tokens)
    
    # Create padded tensors
    input_ids_list = []
    response_masks = []
    
    for i, (full_seq, prompt_len) in enumerate(zip(concatenated_tokens, prompt_lens)):
        seq_len = len(full_seq)
        
        # Pad sequence to max_len
        PAD = tokenizer.pad_token_id  # for Qwen this is 151643
        padded_seq = torch.full((max_len,), PAD, dtype=torch.long)
        padded_seq[:seq_len] = full_seq
        
        # Build response mask in LABEL space (length max_len-1), boolean
        label_mask = torch.zeros(max_len - 1, dtype=torch.bool)
        start = max(prompt_len - 1, 0)   # first output token after shift
        end   = max(seq_len - 1, 0)      # labels end at L-2
        if start < end:
            label_mask[start:end] = True
        
        input_ids_list.append(padded_seq)
        response_masks.append(label_mask)
    
    input_ids_full = torch.stack(input_ids_list)      # (batch_size, max_len)
    input_ids = input_ids_full[:, :-1]
    labels    = input_ids_full[:, 1:]
    response_mask = torch.stack(response_masks)       # (batch_size, max_len - 1), bool


    return {
        "input_ids": input_ids,
        "labels": labels, 
        "response_mask": response_mask
    }












def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    raw_rewards = torch.tensor(
        [float(reward_fn(r, g).get("reward", 0.0))
         for r, g in zip(rollout_responses, repeated_ground_truths)],
        dtype=torch.float32,
    )
    G = raw_rewards.numel() // group_size
    Rg = raw_rewards.view(G, group_size)

    mean = Rg.mean(dim=1, keepdim=True)
    if normalize_by_std:
        std = Rg.std(dim=1, unbiased=True, keepdim=True)  # <-- CHANGED
        normalized = ((Rg - mean) / (std + advantage_eps)).reshape(-1)
    else:
        normalized = (Rg - mean).reshape(-1)

    metadata = {
        "rollout_batch_size": float(raw_rewards.numel()),
        "group_size": float(group_size),
        "num_groups": float(G),
        "normalize_by_std": float(bool(normalize_by_std)),
        "advantage_eps": float(advantage_eps),
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=True).item()),
        "reward_min": float(raw_rewards.min().item()),
        "reward_max": float(raw_rewards.max().item()),
        "adv_mean": float(normalized.mean().item()),
        "adv_std": float(normalized.std(unbiased=True).item()),
    }
    return normalized, raw_rewards, metadata











def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.log_softmax(logits, dim=-1)   # stable
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool= False,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids=input_ids).logits                  # (B, L, V)
    log_probs_all = F.log_softmax(logits, dim=-1)               # stable
    # select log p(label_t | prefix ≤ t-1)
    log_probs = torch.gather(log_probs_all, -1, labels.unsqueeze(-1)).squeeze(-1)  # (B, L)

    out = {"log_probs": log_probs}
    if return_token_entropy:
        probs = torch.exp(log_probs_all)
        token_entropy = -(probs * log_probs_all).sum(dim=-1)    # (B, L)
        out["token_entropy"] = token_entropy

    return out




def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
 # raw_rewards_or_advantages: (B, 1) or (B,)
    if raw_rewards_or_advantages.dim() == 1:
        raw_rewards_or_advantages = raw_rewards_or_advantages.unsqueeze(1)

    # Broadcast A over sequence length and match dtype/device
    A = raw_rewards_or_advantages.to(dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    A = A.expand_as(policy_log_probs)  # (B, T)

    # Per-token naive policy-gradient loss:  -A_t * log p_t
    loss = -A * policy_log_probs  # (B, T)
    return loss








def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """

 # broadcast A over sequence length
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    A = advantages.to(device=policy_log_probs.device, dtype=policy_log_probs.dtype).expand_as(policy_log_probs)
    # r_t = exp(log πθ - log π_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # per-token GRPO-Clip loss:  -min(r*A, clip(r)*A)
    loss = -torch.minimum(ratio * A, clipped_ratio * A)

    # single metadata dict (simple & useful)
    clipped_mask = ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).to(policy_log_probs.dtype)
    metadata = {
        "ratio": ratio.detach(),
        "clipped_mask": clipped_mask,           # 1.0 where clipped, 0.0 otherwise
    }
    return loss, metadata


def run_compute_grpo_clip_KL_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    cliprange: float,
    beta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss with a KL penalty to a reference policy.

    Args:
        advantages: (batch_size, 1) or (batch_size,) advantages per rollout.
        policy_log_probs: (batch_size, seq_len) log π_θ(o_i|q).
        old_log_probs: (batch_size, seq_len) log π_θ_old(o_i|q).
        ref_log_probs: (batch_size, seq_len) log π_ref(o_i|q).
        cliprange: float, ε in the paper (ratio clip range).
        beta: float, β coefficient for the KL penalty.

    Returns:
        loss: (batch_size, seq_len) per-token GRPO-Clip-KL loss.
        metadata: dict with useful diagnostics.
    """

    # Broadcast A over sequence length
    if advantages.dim() == 1:
        advantages = advantages.unsqueeze(1)
    A = advantages.to(device=policy_log_probs.device,
                      dtype=policy_log_probs.dtype).expand_as(policy_log_probs)

    # r_t = exp(log πθ - log π_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # Base GRPO-Clip loss:  -min(r*A, clip(r)*A)
    base_loss = -torch.minimum(ratio * A, clipped_ratio * A)

    # D_KL^i(πθ || π_ref) = π_ref/πθ - log(π_ref/πθ) - 1
    # where π_ref/πθ = exp(log π_ref - log π_θ)
    log_ratio_ref_policy = ref_log_probs - policy_log_probs
    ratio_ref_policy = torch.exp(log_ratio_ref_policy)
    kl_per_token = ratio_ref_policy - log_ratio_ref_policy - 1.0

    # Full loss: -[min(r*A, clip(r)*A) - β D_KL] = base_loss + β D_KL
    loss = base_loss + beta * kl_per_token

    clipped_mask = ((ratio < 1.0 - cliprange) | (ratio > 1.0 + cliprange)).to(
        policy_log_probs.dtype
    )

    metadata = {
        "ratio": ratio.detach(),
        "clipped_mask": clipped_mask,         # 1.0 where clipped, 0.0 otherwise
        "kl_per_token": kl_per_token.detach(),
        "kl_mean": kl_per_token.mean().detach(),
    }

    return loss, metadata

















"""
    Wrapper that delegates to the appropriate policy gradient loss function above.
"""
def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ref_log_probs: torch.Tensor | None = None,
    beta: float | None = None,   # ← optional
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {"loss_type": "no_baseline"}
    elif loss_type == "reinforce_with_baseline":
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_type": "reinforce_with_baseline"}
    elif loss_type == "grpo_clip":
        loss, md = run_compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
        metadata = {"loss_type": "grpo_clip", **md}
    elif loss_type == "grpo_clip_KL":
        if ref_log_probs is None:
            raise ValueError("ref_log_probs is required when loss_type='grpo_clip_KL'")
        loss, md = run_compute_grpo_clip_KL_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            ref_log_probs,
            cliprange,
            beta,
        )
        metadata = {"loss_type": "grpo_clip_KL", **md}

    else:
        # minimal fallback
        loss = run_compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {"loss_type": loss_type}
    return loss, metadata








def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    m = mask.to(dtype=tensor.dtype, device=tensor.device)
    if dim is None:
        den = m.sum()
        if den == 0:
            return torch.tensor(float('nan'), dtype=tensor.dtype, device=tensor.device)
        num = (tensor * m).sum()
        return num / den
    else:
        den = m.sum(dim=dim)
        num = (tensor * m).sum(dim=dim)
        # NaN where den == 0
        out = torch.where(
            den > 0,
            num / den,
            torch.full_like(num, float('nan'))
        )
        return out









def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    mask = response_mask.to(dtype=policy_log_probs.dtype)
    nll = -policy_log_probs  # (B, T)

    # choose denominator (if None, use number of valid tokens)
    denom = mask.sum() if normalize_constant is None else normalize_constant

    # masked normalized sum (scalar)
    loss = run_masked_normalize(nll, mask, normalize_constant=denom)

    # also average across batch and scale for grad accumulation
    B = policy_log_probs.shape[0]
    loss = loss / B / float(gradient_accumulation_steps)

    # accumulate gradients on policy_log_probs (or upstream parameters)
    loss.backward()


    metadata = {
        "num_response_tokens": mask.sum().detach(),
        "batch_size": torch.tensor(B, device=policy_log_probs.device),
        "normalize_constant": torch.as_tensor(denom, device=policy_log_probs.device).detach(),
    }
    return loss, metadata



    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_clip_KL"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    ref_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    normalization="mean",
    beta: float | None = None,
    profile_phase: Callable[[str], Any] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        ref_log_probs: torch.Tensor | None, the log-probs of the reference policy.
            Needed for loss_type="grpo_clip_KL".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).
        profile_phase: Optional context-manager factory used to measure the
            loss reduction and backward pass separately. The default adds no
            profiler dependency and preserves existing call sites.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    phase = profile_phase or (lambda _name: nullcontext())

    with phase("loss"):
        # 1) per-token loss via the wrapper
        per_token_loss, md = run_compute_policy_gradient_loss(
            policy_log_probs=policy_log_probs,
            loss_type=loss_type,
            raw_rewards=raw_rewards,
            advantages=advantages,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            cliprange=cliprange if cliprange is not None else 0.0,
            beta=beta if beta is not None else 0.0,
        )  # shape: (B, T)

        # 2) reduce to per-example/sequence scalar with masked mean over tokens
        #    (mask==1 positions included)
        if normalization == "mean":
            per_example_loss = run_masked_mean(per_token_loss, response_mask, dim=1)
        elif normalization == "normalize":
            max_gen_len = response_mask.shape[1]
            per_example_loss = run_masked_normalize(
                per_token_loss,
                response_mask,
                dim=1,
                normalize_constant=max_gen_len,
            )
        else:
            raise ValueError(f"Unsupported normalization: {normalization}")

        # 3) average over batch and scale for gradient accumulation
        loss = per_example_loss.mean() / float(gradient_accumulation_steps)

    # 4) backprop to accumulate gradients on policy_log_probs (or upstream params)
    with phase("backward"):
        loss.backward()

    # 6) return a detached scalar (for logging) and metadata
    metadata = {"loss_type": torch.tensor(0), **{k: v for k, v in md.items()}}
    return loss, metadata



def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
 # include only mask==1 (supports bool or 0/1 masks)
    masked = tensor * mask.to(dtype=tensor.dtype)

    # sum over the requested dimension (or all)
    summed = masked.sum() if dim is None else masked.sum(dim=dim)

    # divide by the provided constant
    denom = torch.tensor(normalize_constant, dtype=summed.dtype, device=summed.device)
    return summed / denom










"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
