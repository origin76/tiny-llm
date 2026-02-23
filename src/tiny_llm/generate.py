import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    # y: N.. x S, where in week 1 we don't implement batch, so N.. = 1
    # output_logits: N.. x S x vocab_size
    def _step(model, y):
        # y: (1, S) -> output_logits: (1, S, vocab_size)
        output_logits = model(y)
        # 只取最后一个位置的 logits: (1, vocab_size)
        logits = output_logits[:, -1, :]
        # 贪婪解码：取概率最高的 token
        if sampler is not None:
            next_token = sampler(logits)
        else:
            next_token = mx.argmax(logits, axis=-1)
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
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
