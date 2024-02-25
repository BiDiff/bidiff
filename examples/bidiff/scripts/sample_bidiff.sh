
export OUTPUT_DIR="./log/__test_sample__/sample_tmp"
export CONFIG="./configs/bidiff_if_objaverse.yaml"

CUDA_VISIBLE_DEVICES=0 python sample_bidiff.py \
 --output_dir=$OUTPUT_DIR \
 --cfg=$CONFIG \
 --seed=1234 \
 --interactive_mode \
 --sample_config_file="./configs/sample_cfg1.json" \
 --ckpt_path="./log/XXX/checkpoint-XXX/bidiff"
