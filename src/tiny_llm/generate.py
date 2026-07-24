import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> None:
    def _step(model, y):
        logits = model(y[None])
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        if sampler is None:
            return mx.argmax(logprobs, axis=-1)
        return sampler(logprobs)

    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    while True:
        token = _step(model, tokens)
        mx.eval(token)
        tokens = mx.concat([tokens, token])
        if token.item() == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

    detokenizer.finalize()
    return detokenizer.text


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    kv_cache = model.create_kv_cache()

    def _step(model, y, offset, kv_cache):
        logits = model(y[None], offset, kv_cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        token = mx.argmax(logprobs, axis=-1)
        return token, logprobs.squeeze(0)

    try:
        tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))
        detokenizer = tokenizer.detokenizer
        detokenizer.reset()
        offset = 0

        while True:
            token, _ = _step(model, tokens, offset, kv_cache)
            mx.eval(token)
            if token.item() == tokenizer.eos_token_id:
                break
            detokenizer.add_token(token.item())
            print(detokenizer.last_segment, end="", flush=True)
            offset += tokens.size
            tokens = token

        detokenizer.finalize()
        return detokenizer.text
    finally:
        _release_kv_cache(kv_cache)


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
