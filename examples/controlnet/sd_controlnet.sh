export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="./log"

CUDA_VISIBLE_DEVICES=7 accelerate launch train_controlnet_copy.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
#   --push_to_hub