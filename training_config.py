train_file="train_and_eval_set/train_format_2/new_trainset_10k.json"

model_path = "../Mistral-7B-Instruct-v0.2"  # base model

new_model = "Mistral-7B-Instruct-cfg-new-FT"  # new model

output_dir = "./results"  # tensorboard result

lora_r = 16  # LoRA attention dimension

lora_alpha = lora_r * 2  # Alpha parameter for LoRA scaling

lora_dropout = 0.05  # Dropout probability for LoRA layers

bnb_4bit_compute_dtype = "bfloat16"  # Compute dtype for 4-bit base models

output_dir = "./results"  # Output directory where the model predictions and checkpoints will be stored

num_train_epochs = 2  # Number of training epochs

# Enable fp16/bf16 training
fp16 = False
bf16 = True

per_device_train_batch_size = 4  # Batch size per GPU for training

per_device_eval_batch_size = 1  # Batch size per GPU for evaluation

gradient_accumulation_steps = (
    1  # Number of update steps to accumulate the gradients for
)

gradient_checkpointing = True  # Enable gradient checkpointing

max_grad_norm = 0.3  # Maximum gradient normal (gradient clipping)

learning_rate = 5e-5  # Initial learning rate (AdamW optimizer)

weight_decay = (
    0.001  # Weight decay to apply to all layers except bias/LayerNorm weights
)

optim = "paged_adamw_32bit"  # Optimizer to use

lr_scheduler_type = "cosine"  # Learning rate schedule

max_steps = -1  # Number of training steps (overrides num_train_epochs)

warmup_ratio = 0.03  # Ratio of steps for a linear warmup (from 0 to learning rate)

# Save checkpoint every X updates steps
# save_steps = 1000
save_strategy = "epoch"

logging_steps = 25  # Log every X updates steps

max_seq_length = 2048  # Maximum sequence length to use

packing = False  # Pack multiple short examples in the same input sequence to increase efficiency

device_map = "auto"  # Load the entire model to the GPU auto, if want to specific, use `CUDA_VISIBLE_DEVICES=0 python training.py` in terminal
