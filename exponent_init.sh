python ./src/exponent_init.py \
  --base_model Qwen/Qwen3-8B \
  --base_tokenizer Qwen/Qwen3-8B \
  --ext_tokenizer DopeorNope/FFT-expanded-naive \
  --output_dir ./qwen3-8b-expanded-exponent-init \
  --alpha 2.0 \
  --device_map cpu \
  --dtype bf16 \
  --dump_json ./res/exponent_new_token_decomp.json