from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 로드
model_name = "PygmalionAI/pygmalion-2-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

# 프롬프트 설정
prompt = (
    "You are Romeo, standing under Juliet's balcony on a moonlit night. "
    "With a heart full of love, you confess your feelings to Juliet. "
    "Your words are poetic, passionate, and sincere:\n\n"
    "Romeo: "
)

# 입력 데이터 준비
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 텍스트 생성
outputs = model.generate(
    **inputs,
    max_new_tokens=150,      # 생성할 최대 토큰 수
    temperature=0.8,         # 다양성 조절
    top_p=0.9,               # 상위 확률 기반 샘플링
    no_repeat_ngram_size=3,  # 3-gram 반복 방지
    repetition_penalty=1.2   # 반복 억제
)

# 결과 출력
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
