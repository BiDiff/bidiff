export MODEL_DIR="DeepFloyd/IF-I-XL-v1.0"
export OUTPUT_DIR="./log/lr56g2"

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port=29950 --multi_gpu train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --instance_prompt="colored circle with colored background" \
 --dataset_name=fusing/fill50k \
 --resolution=64 \
 --learning_rate=5e-6 \
 --lr_scheduler "constant_with_warmup" \
 --lr_warmup_steps=2000 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=4 \
 --gradient_accumulation_steps=1 \
 --text_encoder_use_attention_mask \
 --tokenizer_max_length 77 \
 --set_grads_to_none \
 --skip_save_text_encoder \
 --max_train_steps 200000 \
 --checkpointing_steps 12000 \
 --validation_steps 200 \
 --report_to=wandb \
 --use_8bit_adam \
#  --enable_xformers_memory_efficient_attention \
#  --pre_compute_text_embeddings \
#  --push_to_hub
