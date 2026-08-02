pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

Time: 6.29s, Output throughput: 10.33 tok/s
Total throughput (prompt+output): 30.68 tok/s
Prefill throughput: 377.45 tok/s
Decode throughput: 10.75 tok/s

pdm run bench --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model qwen3-4b \
  --num-seqs 1 --min-input-len 128 --max-input-len 128 \
  --min-output-len 65 --max-output-len 65 --warmup 2

Requests: 1, Prompt tokens: 128, Generated tokens: 65
Time: 4.75s, Output throughput: 13.69 tok/s
Total throughput (prompt+output): 40.66 tok/s
Prefill throughput: 46.27 tok/s
Decode throughput: 32.34 tok/s