from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === Load fine-tuned model ===
model_path = "./finetuned-distilgpt2-full"  # เปลี่ยนเป็น LoRA model path ได้
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# === Use GPU if available ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === Prompt list ===
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
    
    # Encode input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,       # ป้องกันการวนคำซ้ำ
            no_repeat_ngram_size=3,       # ไม่ให้ซ้ำ n-gram 3 คำ
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode result
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated:")
    print(generated_text)
