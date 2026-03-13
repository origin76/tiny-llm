import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from .sampler import make_sampler
from .kv_cache import TinyKvCache, TinyKvFullCache
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    # y: N.. x S, where in week 1 we don't implement batch, so N.. = 1
    # output_logits: N.. x S x vocab_size
    # Use make_sampler if no sampler provided
    if sampler is None:
        sampler = make_sampler(0.0, 1.0, None)

    def _step(model, y):
        # y: (1, S) -> output_logits: (1, S, vocab_size)
        output_logits = model(y)
        # 只取最后一个位置的 logits: (1, vocab_size)
        logits = output_logits[:, -1, :]
        # 采样解码
        next_token = sampler(logits)
        return next_token  # shape: (1,)
    
    print(f"Prompt: {prompt}")
    input_ids = tokenizer.encode(prompt)
    y = mx.array(input_ids)[None]  # (1, S)

    # 2. Prefill：用整个 prompt 预测第一个新 token
    next_token = _step(model, y)
    mx.eval(next_token)

    generated = [next_token.item()]

    # 3. 逐步生成，直到 EOS 或达到最大长度
    max_new_tokens = 512
    eos_token_id = tokenizer.eos_token_id

    for i in range(max_new_tokens - 1):
        if generated[-1] == eos_token_id:
            break
        # 将新 token 追加到序列末尾
        y = mx.array(input_ids + generated)[None]  # (1, S + generated_so_far)
        next_token = _step(model, y)
        mx.eval(next_token)
        generated.append(next_token.item())
    
    # 4. Decode
    result = tokenizer.decode(generated)
    print(f"DEBUG: 解码后的最终字符串: '{result}'")
    
    return result


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    # Initialize KV cache for each layer
    kv_cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    def _step(model, y, offset, kv_cache):
        # y: (S,) token ids
        # offset: current sequence length (position of last token processed)
        # Returns: next token, logprobs
        logits = model(y[None], offset, kv_cache)  # (1, 1, vocab_size)
        logits = logits[:, -1, :]  # (1, vocab_size)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampler = lambda x: mx.argmax(x, axis=-1)
        y = sampler(logprobs)
        return y, logprobs.squeeze(0)

    # Encode prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))

    # Initialize detokenizer for streaming output
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()

    offset = 0

    # First iteration is prefill - process entire prompt
    while True:
        token, _ = _step(model, tokens, offset, kv_cache)
        mx.eval(token)

        if token.item() == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token.item())
        print(detokenizer.last_segment, end="", flush=True)

        # Update offset: prefill uses prompt length, decode uses 1
        offset += tokens.size
        tokens = token
    
    detokenizer.finalize()

    return detokenizer.text


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
