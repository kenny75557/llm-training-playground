import os
import argparse
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
import training_config

torch.cuda.empty_cache()
print(f"GPU: {torch.cuda.get_device_name(0)}")

# transform data to llm format
def transform_data_format(data):
    # transform data
    instruction="""You're an intelligent language recognizer, and I want to play a game with you.
    The rules are: I'll provide sentence pairs for you to observe their relationship. 
    Your task is to guess the structure of the second sentence based on the first one.
    Please provide the sentences without restating my question, and keep the capitalization unchanged. origin_sentence:
    """
    data["formated"] = (
        f"""[INST]{instruction}{data['origin_sentence']}\n transform_sentence: [/INST]{data['transform_sentence']}</s>"""
    )

    return data

# Sampling
def sampling_data(dataset, ratio=0.1):
    return dataset.select(range(int(len(dataset) * ratio)))


def load_training_data(file_path):
    datasize="10k"
    train_dataset_path = "train_and_eval_set/train_format_2/new_trainset_10k.json"
    # "train_and_eval_set/train_format_1/former_trainset_1k.json"
    # test_dataset_path = "train_and_eval_set/testset/reverse_format_testset.json"
    # reverse_format_testset.json /init_format_testset.json

    dataset = load_dataset(
        'json',data_files={
        "train":train_dataset_path,
        # "test":test_dataset_path
        }
    )  # load dataset

    dataset = dataset.shuffle(seed=42)  # shuffle dataset

    # dataset = sampling_data(dataset, ratio=0.1)  # sampling dataset if needed

    dataset = dataset.map(
        transform_data_format
    )  
    # remove_columns=["input", "instruction", "output"]
    # transform data to llm format
    return dataset
def load_model_and_tokenizer(model_path:str):
    # Check GPU compatibility with bfloat16
    compute_dtype = getattr(torch, training_config.bnb_4bit_compute_dtype)
    if compute_dtype == torch.bfloat16:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print(f"Now bf16 is set to {training_config.bf16}")
            print("=" * 80)
     # QLoRA configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
     # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        # torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # optional
        trust_remote_code=True,  # MUST
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # This is common between both models


def set_trainer(tokenizer,output_dir,model,dataset,peft_config):
     # DataCollatorForCompletionOnlyLM 會把 instruction_template 到 response_template 之間的 label 設成 -100，所以就不會計算 loss
    response_template = "transform_sentence: [/INST]"
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer.encode(response_template, add_special_tokens = False)[2:],
        tokenizer=tokenizer)
    # collator = DataCollatorForCompletionOnlyLM(
    #     # instruction_template=instruction_template,
    #     response_template=response_template,
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        # auto_find_batch_size=True,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        optim=training_config.optim,
        # save_steps=save_steps,
        save_strategy=training_config.save_strategy,
        logging_steps=training_config.logging_steps,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        max_grad_norm=training_config.max_grad_norm,
        max_steps=training_config.max_steps,
        warmup_ratio=training_config.warmup_ratio,
        # group_by_length=group_by_length,
        lr_scheduler_type=training_config.lr_scheduler_type,
        report_to="tensorboard",
        # shuffle datasets for each epoch
        data_seed=42,
    )
    return SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=peft_config,
        dataset_text_field="formated",  # process data field
        data_collator=collator,  # Set supervised collactor
        max_seq_length=training_config.max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=training_config.packing,
    )
    # Set supervised fine-tuning parameters
    # trainer = 




def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune Llama model with LoRA")
    parser.add_argument("--model_path", type=str, default=training_config.model_path, help="Path to the pre-trained model")
    parser.add_argument("--train_file", type=str, default=training_config.train_file, help="Path to the training data file")
    parser.add_argument("--output_dir", type=str, default=training_config.output_dir, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_train_epochs", type=int, default=training_config.num_train_epochs, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=training_config.learning_rate, help="Learning rate for training")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Load dataset
    dataset=load_training_data(args.train_file)
    # Load tokenizer and Model set QLoRA config 
    model, tokenizer=load_model_and_tokenizer(args.model_path)    
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=training_config.lora_alpha,
        lora_dropout=training_config.lora_dropout,
        r=training_config.lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)  # Add LoRA to the model
    # model.print_trainable_parameters()  # print trainable parameters
    # set tainer and training arguments
    trainer=set_trainer(tokenizer,args.output_dir,model,dataset,peft_config)


    ### 張量看板
    # tensorboard --logdir=results --host 0.0.0.0 --port 11204
    print("=" * 80 + "\nTraining\n" + "=" * 80)
    trainer.train(tokenizer,args.output_dir)
    trainer.model.save_pretrained(training_config.new_model)  # 儲存lora參數

    # Merge LoRA weights to the base model
    print("=" * 80 + "\nMerge LoRA weights to the base model\n" + "=" * 80)
    torch.cuda.empty_cache()  # Dealing with -> ValueError: weight is on the meta device, we need a `value` to put in on 1.

    base_model = AutoModelForCausalLM.from_pretrained(
        training_config.model_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map=training_config.device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(training_config.model_path, trust_remote_code=True)
    # if tokenizer.pad_token == tokenizer.eos_token:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    #     base_model.resize_token_embeddings(len(tokenizer))
    #     print(f"Set {tokenizer.pad_token} as pad_token")


    model = PeftModel.from_pretrained(base_model, training_config.new_model)  # 載入lora參數
    model = model.merge_and_unload()  # Merge LoRA with origin model

    model.save_pretrained(training_config.new_model + "-full")  # 儲存完整模型
    tokenizer.save_pretrained(training_config.new_model + "-full")  # 儲存完整模型的tokenizer

if __name__ == "__main__":
    main()