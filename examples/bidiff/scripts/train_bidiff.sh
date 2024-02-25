
export MODEL_DIR="DeepFloyd/IF-I-XL-v1.0"
export OUTPUT_DIR="./log/debug0"
export CONFIG="./configs/bidiff_if_objaverse_debug.yaml"

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port=29500 --multi_gpu train_bidiff.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --cfg=$CONFIG \
 --pre_compute_text_embeddings \
 --resolution=64 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=1 \
 --learning_rate=2e-6 \
 --mv_learning_rate=5e-5 \
 --snr_gamma=5 \
 --validation_steps 4 \
 --text_encoder_use_attention_mask \
 --tokenizer_max_length 77 \
 --set_grads_to_none \
 --skip_save_text_encoder \
 --max_train_steps 400000 \
 --checkpointing_steps 5 \
 --checkpoints_total_limit 2 \
 --use_8bit_adam \
 --gradient_checkpointing \
#  --mixed_precision="fp16" \
#  --enable_xformers_memory_efficient_attention \
#  --lr_scheduler="cosine_with_restarts" \
#  --lr_num_cycles=10 \
#  --push_to_hub