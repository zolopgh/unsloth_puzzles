import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# Set environment variables for verbose compilation logging
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCH_COMPILE_DEBUG"] = "1"

# Compilation config with just options (no mode)
torch_compile_config = {
    "fullgraph": True,
    "dynamic": True,
    "options": {
        "epilogue_fusion": True,
        "max_autotune": True,
        "trace.enabled": True,
        "triton.cudagraphs": True,
    }
}

# Patch attention mechanism
def patch_attention():
    from transformers.models.llama.modeling_llama import LlamaAttention
    
    original_forward = LlamaAttention.forward
    
    @torch.compile(**torch_compile_config)
    def compiled_forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False):
        return original_forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache)
    
    LlamaAttention.forward = compiled_forward

# Patch MLP
def patch_mlp():
    from transformers.models.llama.modeling_llama import LlamaMLP
    
    original_forward = LlamaMLP.forward
    
    @torch.compile(**torch_compile_config)
    def compiled_forward(self, x):
        return original_forward(self, x)
    
    LlamaMLP.forward = compiled_forward

def setup_model(model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit"):
    print("Loading model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    # Try loading with local GPU device
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": 0},  # Force local GPU
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    
    print("Patching model components...")
    # Apply patches before LoRA
    patch_attention()
    patch_mlp()
    
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
    
    model.enable_input_require_grads()
    return model

def main():
    model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "right"
    
    print("Setting up model...")
    model = setup_model(model_name)
    
    print("Loading dataset...")
    # Load a small subset of data for testing
    dataset = load_dataset(
        "json", 
        data_files={"train": "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"}, 
        split="train[:100]"
    )
    
    training_args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        warmup_steps=1,
        max_steps=10,
        logging_steps=1,
        fp16=True,
        optim="adamw_torch",
        learning_rate=2e-4,
        report_to="none",
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=512,
    )
    
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
