import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)

        # Top-k sampling: keep only top-k tokens
        if top_k is not None and top_k > 0:
            # Get indices of elements outside top-k and set to -inf
            mask_indices = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[
                :, top_k:
            ]
            logprobs = mx.array(logprobs)
            logprobs[:, mask_indices] = -mx.inf

        # Top-p (Nucleus) sampling: keep tokens with cumulative prob <= top_p
        if top_p is not None and top_p > 0:
            # Sort from highest to lowest
            sorted_idx = mx.argsort(-logprobs, axis=-1)
            sorted_logprobs = logprobs[:, sorted_idx]
            # Cumulative sum of probabilities (after softmax)
            cumsum = mx.cumsum(mx.exp(sorted_logprobs), axis=-1)
            # Mask tokens where cumsum >= top_p (keep those where cumsum < top_p)
            mask_elements = cumsum < top_p
            # Always keep at least the first token
            mask_elements[..., 0] = True
            # Apply mask: set to -inf
            logprobs[:, sorted_idx] = mx.where(
                mask_elements,
                sorted_logprobs,
                -mx.inf * mx.ones_like(logprobs),
            )

        # Temperature sampling: scale by temperature and sample
        scaled_logprobs = logprobs / temp
        return mx.random.categorical(scaled_logprobs)

    return sample
