export CUDA_VISIBLE_DEVICES="0,1,2,3"

torchrun --nproc_per_node=4 src/train_nl2fl.py \
  --deepspeed //data/SIML/sy/group_theory/Continual_learning/config/zero3_bf16.json \
  --output_dir ./FFT-naive \
  --model_name_or_path Qwen/Qwen3-8B \
  --max_seq_length 4096 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 2 \
  --num_train_epochs 1 \
  --wandb_project "continual-learning" \
  --wandb_name "naive-fft" \
  --wandb_tags "fft,naive"