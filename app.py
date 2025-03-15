import streamlit as st
import torch
from deepseke_r1 import DeepSeKeR1Tokenizer, DeepSeKeR1ForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset

emoji_math_problems = [
    {"problem": "🍎 + 🍎 + 🍎 = 12", "solution": "🍎 = 4"},
    {"problem": "🍌 + 🍌 = 8", "solution": "🍌 = 4"},
    {"problem": "🍎 + 🍌 = 10, 🍎 = 6", "solution": "🍌 = 4"},
    {"problem": "🍋 × 2 = 14", "solution": "🍋 = 7"}
]

def train_model():
    dataset = Dataset.from_dict({"problem": [p["problem"] for p in emoji_math_problems], "solution": [p["solution"] for p in emoji_math_problems]})
    tokenizer = DeepSeKeR1Tokenizer.from_pretrained("deepseke-r1")
    
    def tokenize_function(examples):
        return tokenizer(examples["problem"], text_target=examples["solution"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    model = DeepSeKeR1ForSeq2SeqLM.from_pretrained("deepseke-r1")
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets
    )
    trainer.train()
    model.save_pretrained("./emoji_math_solver")
    tokenizer.save_pretrained("./emoji_math_solver")
    return model, tokenizer

def solve_emoji_math(prompt, model, tokenizer):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)
    return tokenizer.decode(output[0], skip_special_tokens=True)

st.title("Emoji Math Solver")
if st.button("Fine-Tune Model"):
    model, tokenizer = train_model()
    st.success("Model fine-tuned successfully!")

prompt = st.text_input("Enter an emoji-based math problem:")
if st.button("Solve Problem") and prompt:
    model = DeepSeKeR1ForSeq2SeqLM.from_pretrained("./emoji_math_solver")
    tokenizer = DeepSeKeR1Tokenizer.from_pretrained("./emoji_math_solver")
    solution = solve_emoji_math(prompt, model, tokenizer)
    st.write(solution)


