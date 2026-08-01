"""Your solution. See task_prompt.md for the full contract.

Implement a fast, batched, KV-cached greedy decoder for TinyGPT that matches
the reference generate() token-for-token (per sequence, independently) and is
at least 8x faster, measured live.
"""


def generate_cached(model, prompt_ids, max_new_tokens):
    """prompt_ids: list of B 1-D LongTensors (ragged). Returns
    (new_tokens LongTensor[B, N], step_logits FloatTensor[B, N, 66])."""
    raise NotImplementedError
