export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="projects/$1/training"
export OUTPUT_DIR="projects/$1/model"
export HF_REPO_NAME="$1-lora-model"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks $1" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks $1" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
