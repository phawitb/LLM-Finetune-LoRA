from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# === Load base model + apply LoRA ===
model_path = "./finetuned-distilgpt2-lora"
tokenizer = AutoTokenizer.from_pretrained(model_path)

base_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
model = PeftModel.from_pretrained(base_model, model_path)

# === Use GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Prompts ===
prompts = [
    "Mars is known for its massive volcanoes, such as",
    "Saturn's ring system is made primarily of",
    "Voyager 1 was launched in 1977 and",
    "All planets in the Solar System orbit the Sun in",
    "Why is Earth able to support life?",
    "The Kuiper Belt is a region beyond Neptune where",
    "Neptune has the strongest winds in the Solar System, reaching",
    "Jupiter’s moon Europa may harbor life because"
]

# === Inference loop ===
for i, prompt in enumerate(prompts, start=1):
    print(f"\nPrompt {i}: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated:")
    print(generated_text)
