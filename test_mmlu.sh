export CUDA_VISIBLE_DEVICES="0,1,2,3"
export model="Qwen/Qwen3-8B"

export outdir="stage2_mmlu_5shot"

python eval_mmlu.py \
  --model ${model} \
  --data_file ./data/mmlu_test.jsonl \
  --dev_file  ./data/mmlu_dev.jsonl \
  --nshot 5 \
  --tensor_parallel_size 4 \
  --batch_size 256 \
  --seq_len 4096 \
  --outdir ${outdir}


python test/parsing_mmlu.py --data_file ${outdir}.csv --use_second_model --tensor_parallel_size 4
