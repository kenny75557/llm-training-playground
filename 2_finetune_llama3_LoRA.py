import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from datasets import  load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_from_disk

torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")


# transform data to llm format
def transform_data_format(data):
    # transform data
    instruction="""You're an intelligent language recognizer, and I want to play a game with you.\n
    The rules are: I'll provide sentence pairs for you to observe their relationship. \n
    Your task is to guess the structure of the second sentence based on the first one.\n
    Please provide the sentences without restating my question, and keep the capitalization unchanged\n
    """
    data["formated"] = (
        f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

{data['origin_sentence']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{data['transform_sentence']}<|eot_id|><|end_of_text|>"""
    )

    return data


# Sampling
def sampling_data(dataset, ratio=0.1):
    return dataset.select(range(int(len(dataset) * ratio)))

# data path
#old 500
datasize="10k"
train_dataset_path = "train_and_eval_set/train_format_2/new_trainset_10k.json"
# "train_and_eval_set/train_format_1/former_trainset_1k.json"
test_dataset_path = "train_and_eval_set/testset/reverse_format_testset.json"
# reverse_format_testset.json /init_format_testset.json


dataset = load_dataset(
    'json',data_files={
    "train":train_dataset_path,
    "test":test_dataset_path
    }
)  # load dataset

dataset = dataset.shuffle(seed=42)  # shuffle dataset

# dataset = sampling_data(dataset, ratio=0.1)  # sampling dataset

dataset = dataset.map(
    transform_data_format
)  
# remove_columns=["input", "instruction", "output"]
# transform data to llm format

model_path = "../Meta-Llama-3-8B-Instruct"  # base model

new_model = "Llama-3-8B-Instruct_cfg_new_FT"+datasize  # new model

output_dir = "./llama-3_results"+datasize  # tensorboard 結果

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

# Check GPU compatibility with bfloat16
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
if compute_dtype == torch.bfloat16:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print(f"Now bf16 is set to {bf16}")
        print("=" * 80)

# QLoRA configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    # device_map="cuda:0",
    
    device_map="auto",
    # torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # optional
    trust_remote_code=True,  # MUST
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# if tokenizer.pad_token == tokenizer.eos_token:
#     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#     model.resize_token_embeddings(len(tokenizer))
#     print(f"Set {tokenizer.pad_token} as pad_token")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
print(f"Set {tokenizer.pad_token} as pad_token")

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)  # Add LoRA to the model
# model.print_trainable_parameters()  # print trainable parameters

# Dataset.filter for long sequence, filt MAX_LENGTH，是 tokenize 後的長度，超過就丟掉
print("=" * 80)
print("Before filter")
print(dataset)
print("=" * 80)
dataset = dataset.filter(
    lambda example: len(tokenizer(example["formated"])["input_ids"]) < max_seq_length
)  # filter dataset by max_seq_length
print("=" * 80)
print("After filter")
print(dataset)
print("=" * 80)

# DataCollatorForCompletionOnlyLM 會把 instruction_template 到 response_template 之間的 label 設成 -100，所以就不會計算 loss
instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    # auto_find_batch_size=True,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    # save_steps=save_steps,
    save_strategy=save_strategy,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    # group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    # shuffle datasets for each epoch
    data_seed=42,
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    dataset_text_field="formated",  # process data field
    data_collator=collator,  # Set supervised collactor
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

### 張量看板
# tensorboard --logdir=results --host 0.0.0.0 --port 11204
print("=" * 80 + "\nTraining\n" + "=" * 80)
trainer.train()
trainer.model.save_pretrained(new_model)  # 儲存lora參數

# Merge LoRA weights to the base model
print("=" * 80 + "\nMerge LoRA weights to the base model\n" + "=" * 80)

torch.cuda.empty_cache()  # Dealing with -> ValueError: weight is on the meta device, we need a `value` to put in on 1.

base_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token == tokenizer.eos_token:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Set {tokenizer.pad_token} as pad_token")


model = PeftModel.from_pretrained(base_model, new_model)  # 載入lora參數
model = model.merge_and_unload()  # Merge LoRA with origin model

model.save_pretrained(new_model + "-full")  # 儲存完整模型
tokenizer.save_pretrained(new_model + "-full")  # 儲存完整模型的tokenizer