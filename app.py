import json

data = [
    {"problem": "🍎 + 🍎 + 🍎 = 12", "solution": "🍎 = 4", "explanation": "Dividing both sides by 3: 12 ÷ 3 = 4."},
    {"problem": "🍌 + 🍌 = 10", "solution": "🍌 = 5", "explanation": "Dividing both sides by 2: 10 ÷ 2 = 5."},
    {"problem": "🍪 + 🍪 + 🍪 = 9", "solution": "🍪 = 3", "explanation": "Dividing both sides by 3: 9 ÷ 3 = 3."},
    {"problem": "🍉 - 🍋 = 2 (🍉 = 7)", "solution": "🍋 = 5", "explanation": "Subtracting 5 from both sides: 7 - 5 = 2."},
    {"problem": "🍍 × 3 = 15", "solution": "🍍 = 5", "explanation": "Dividing both sides by 3: 15 ÷ 3 = 5."},
    {"problem": "🥕 × 4 = 20", "solution": "🥕 = 5", "explanation": "Dividing both sides by 4: 20 ÷ 4 = 5."},
    {"problem": "🍊 + 🍊 = 16", "solution": "🍊 = 8", "explanation": "Dividing both sides by 2: 16 ÷ 2 = 8."},
    {"problem": "🥑 + 🥑 + 🥑 = 18", "solution": "🥑 = 6", "explanation": "Dividing both sides by 3: 18 ÷ 3 = 6."},
    {"problem": "🥒 - 🍒 = 3 (🥒 = 9)", "solution": "🍒 = 6", "explanation": "Subtracting 6 from both sides: 9 - 6 = 3."},
    {"problem": "🍄 × 2 = 14", "solution": "🍄 = 7", "explanation": "Dividing both sides by 2: 14 ÷ 2 = 7."},
    {"problem": "🍞 ÷ 2 = 4", "solution": "🍞 = 8", "explanation": "Multiplying both sides by 2: 4 × 2 = 8."},
    {"problem": "🌽 + 🌽 + 🌽 = 15", "solution": "🌽 = 5", "explanation": "Dividing both sides by 3: 15 ÷ 3 = 5."},
    {"problem": "🥔 × 5 = 25", "solution": "🥔 = 5", "explanation": "Dividing both sides by 5: 25 ÷ 5 = 5."},
    {"problem": "🍑 + 🍍 = 14 (🍑 = 6)", "solution": "🍍 = 8", "explanation": "Subtracting 6 from both sides: 14 - 6 = 8."},
    {"problem": "🍆 + 🍅 + 🍆 = 21 (🍆 = 7)", "solution": "🍅 = 7", "explanation": "Subtracting 14 from both sides: 21 - 14 = 7."},
    {"problem": "🥜 × 3 = 18", "solution": "🥜 = 6", "explanation": "Dividing both sides by 3: 18 ÷ 3 = 6."},
    {"problem": "🍋 + 🍍 = 16 (🍋 = 7)", "solution": "🍍 = 9", "explanation": "Subtracting 7 from both sides: 16 - 7 = 9."},
    {"problem": "🥝 - 🍉 = 4 (🥝 = 10)", "solution": "🍉 = 6", "explanation": "Subtracting 6 from both sides: 10 - 6 = 4."},
    {"problem": "🍰 + 🍰 + 🍰 = 27", "solution": "🍰 = 9", "explanation": "Dividing both sides by 3: 27 ÷ 3 = 9."},
    {"problem": "🥥 ÷ 2 = 5", "solution": "🥥 = 10", "explanation": "Multiplying both sides by 2: 5 × 2 = 10."},
    {"problem": "🍜 × 3 = 24", "solution": "🍜 = 8", "explanation": "Dividing both sides by 3: 24 ÷ 3 = 8."},
    {"problem": "🥭 + 🍒 = 13 (🥭 = 6)", "solution": "🍒 = 7", "explanation": "Subtracting 6 from both sides: 13 - 6 = 7."},
    {"problem": "🥒 ÷ 2 = 6", "solution": "🥒 = 12", "explanation": "Multiplying both sides by 2: 6 × 2 = 12."},
    {"problem": "🍠 + 🍠 = 18", "solution": "🍠 = 9", "explanation": "Dividing both sides by 2: 18 ÷ 2 = 9."},
    {"problem": "🥐 + 🥐 + 🥐 = 30", "solution": "🥐 = 10", "explanation": "Dividing both sides by 3: 30 ÷ 3 = 10."},
    {"problem": "🍏 × 4 = 32", "solution": "🍏 = 8", "explanation": "Dividing both sides by 4: 32 ÷ 4 = 8."},
    {"problem": "🍔 + 🍟 = 15 (🍔 = 7)", "solution": "🍟 = 8", "explanation": "Subtracting 7 from both sides: 15 - 7 = 8."},
    {"problem": "🍩 × 2 = 14", "solution": "🍩 = 7", "explanation": "Dividing both sides by 2: 14 ÷ 2 = 7."},
    {"problem": "🥦 + 🥦 = 20", "solution": "🥦 = 10", "explanation": "Dividing both sides by 2: 20 ÷ 2 = 10."},
    {"problem": "🥜 ÷ 5 = 2", "solution": "🥜 = 10", "explanation": "Multiplying both sides by 5: 2 × 5 = 10."}
]

with open("emoji_math_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
import os
os.environ["HF_TOKEN"] = #simply paste your Hugging face token here


from huggingface_hub import whoami
print(whoami())from unsloth import FastLanguageModel
import torch

model_id = "deepseek-ai/deepseek-math-7b-instruct"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_id,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True
)

tokenizer.pad_token = tokenizer.eos_token
from transformers import AutoTokenizer



model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",    
    use_gradient_checkpointing = "unsloth", 
    random_state = 3977,
    use_rslora = False,  
    loftq_config = None, 
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deepseek-math-fine",
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,  
    optim="adamw_8bit",  
    save_steps=500,
    eval_strategy="steps",  
    eval_steps=500,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,  
    num_train_epochs=3,
    push_to_hub=False,
    save_total_limit=2,  
    report_to="none", 
    run_name="deepseek-math-finetune"  
)


from datasets import load_dataset

dataset = load_dataset("json", data_files="emoji_math_dataset.json")

EOS_TOKEN = tokenizer.eos_token  

def formatting_prompts_func(examples):
    texts = [prob + " -> " + sol + EOS_TOKEN for prob, sol in zip(examples["problem"], examples["solution"])]
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

def tokenize_function(examples):
    model_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=2048)
    model_inputs["labels"] = model_inputs["input_ids"].copy()  
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

trainer.train()


import torch

def test_model(model, tokenizer, test_cases):
    model.eval()
    results = []

    for test in test_cases:
        input_text = test + " ->"
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)

        solved_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append((test, solved_text))

    return results

test_cases = [
    "🍎 + 🍎 + 🍎 = 15",
    "🍌 × 4 = 20",
    "🍪 + 🍪 = 8",
    "🍉 - 🍋 = 3 (🍉 = 10)"
]

results = test_model(model, tokenizer, test_cases)

for problem, solution in results:
    print(f"Problem: {problem}")
    print(f"Solution: {solution}\n")
