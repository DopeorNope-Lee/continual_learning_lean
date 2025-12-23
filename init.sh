python src/subword_initialization.py \
  --model_path Qwen/Qwen3-8B \
  --base_tokenizer ./qwen3_tokenizer \
  --new_tokenizer ./expanded_qwen3_tokenizer_safe \
  --save_dir ./qwen3_8b_expanded \
  --dtype bfloat16 \
  --device cpu