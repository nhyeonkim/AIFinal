import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def smoothquant_activations(model, data_loader, quant_level=8):
    model.eval()
    scale_factors = {}
    def register_hooks():
        hooks = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):  # Linear 계층만 처리
                def forward_hook(module, inputs, outputs):
                    # RMS (Root Mean Square) 계산
                    rms_activation = torch.sqrt((outputs ** 2).mean(dim=0)).cpu().numpy()
                    activations[name] = rms_activation
                hooks[name] = module.register_forward_hook(forward_hook)
        return hooks

    activations = {}
    hooks = register_hooks()

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch["input_ids"].to("cuda")
            _ = model(inputs)

        for name, activation_values in activations.items():
            activation_mean = np.mean(activation_values)
            scale_factor = 1 / (activation_mean + 1e-6)
            scale_factors[name] = scale_factor

            def forward_hook(module, inputs, outputs):
                return outputs * scale_factor

            layer = dict(model.named_modules())[name]
            layer.register_forward_hook(forward_hook)

    for hook in hooks.values():
        hook.remove()

    return scale_factors

def awq_weights(model, quant_level=8, num_clusters=4):
    quantized_model = model

    for name, param in model.named_parameters():
        if "weight" in name:
            weight = param.data
            weight_flat = weight.view(-1).float().cpu().numpy()

            # K-Means 클러스터링
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(weight_flat.reshape(-1, 1))
            cluster_centers = kmeans.cluster_centers_
            labels = kmeans.labels_

            # 클러스터별 가중치를 재매핑
            scale = 2 ** quant_level - 1
            quantized_weights = np.zeros_like(weight_flat)
            for cluster_id, center in enumerate(cluster_centers):
                cluster_values = weight_flat[labels == cluster_id]
                cluster_min = cluster_values.min()
                cluster_max = cluster_values.max()

                # 가중치 범위를 양자화
                quantized_cluster_values = np.round(
                    (cluster_values - cluster_min) / (cluster_max - cluster_min) * scale
                )
                quantized_weights[labels == cluster_id] = (
                    quantized_cluster_values * (cluster_max - cluster_min) / scale + cluster_min
                )

            param.data.copy_(torch.tensor(quantized_weights).view_as(param))

    return quantized_model

# Combined approach: SmoothQuant + AWQ
def combined_smoothquant_awq(model, data_loader, quant_level=8, num_clusters = 8):
    """
    Combines SmoothQuant and AWQ for activation and weight quantization.

    Args:
        model (nn.Module): The neural network model.
        data_loader (DataLoader): Data loader for calibration data.
        quant_level (int): Target quantization level (e.g., INT8 or INT4).

    Returns:
        quantized_model (nn.Module): Fully quantized model.
    """
    print("Applying SmoothQuant to activations...")
    smoothquant_activations(model, data_loader, quant_level)

    print("Applying AWQ to weights...")
    quantized_model = awq_weights(model, quant_level, num_clusters)

    return quantized_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, DataCollatorForLanguageModeling
    from datasets import load_dataset
    from sklearn.cluster import KMeans

    #토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10%]")

    # 데이터셋에서 텍스트 추출 및 전처리
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # DataLoader로 변환
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM용
    )

    data_loader = DataLoader(
        tokenized_dataset,
        batch_size=2,  # 배치 크기
        shuffle=True,
        collate_fn=data_collator
    )

    # LLaMA2 모델
    model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    device_map="auto",  # 메모리에 맞게 자동 분산
    torch_dtype=torch.float16,
    use_auth_token=True
    )   
    model = model.float()
    model = model.to("cuda")
    # LLaMA2 모델 설정 파일 불러오기
    config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    model.resize_token_embeddings(len(tokenizer))  # 토크나이저 크기 조정

    # 입력 데이터 준비
    #sample_texts = ["Hello, how are you?"]

    # 모델 양자화 실행
    quant_level = 4  # INT4
    num_clusters = 8
    quantized_model = combined_smoothquant_awq(model, data_loader, quant_level, num_clusters)
    #quantized_model = combined_optimized_quantization(model, data_loader, quant_level)

    print("Quantization complete.")
def generate_response(model, tokenizer, input_text, max_length=50):
    """
    Args:
        model (nn.Module): 추론에 사용할 모델.
        tokenizer (AutoTokenizer): 모델의 토크나이저.
        input_text (str): 입력 텍스트.
        max_length (int): 생성할 텍스트의 최대 길이.

    Returns:
        str: 생성된 텍스트.
    """
    # 입력 텍스트를 토크나이저로 인코딩
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048
    ).input_ids.to(model.device)

    # 양자화된 모델로 텍스트 생성
    output = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,  # 확률적 샘플링 활성화
        top_k=50,  # 상위 k개 단어 중 선택
        top_p=0.95,  # 누적 확률 기반 필터링
        temperature=0.7  # 생성의 다양성 제어
    )

    # 생성된 토큰을 텍스트로 변환
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# 양자화된 모델 저장
# 가중치와 활성값 스케일 모두 저장
torch.save(quantized_model.state_dict(), "quantized_model.pth")
scale_factors = smoothquant_activations(model, data_loader, quant_level)
torch.save(scale_factors, "activation_scales.pth")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import evaluate
import numpy as np

# LLaMA 2 모델과 토크나이저 로드
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Wikitext 데이터셋 로드
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
test_data = dataset["test"]

# Perplexity 계산 함수
def calculate_perplexity(model, tokenizer, input_texts):
    model.eval()
    total_loss = 0
    total_length = 0
    with torch.no_grad():
        for text in input_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)
    perplexity = np.exp(total_loss / total_length)
    return perplexity

# BLEU 점수 계산 함수
def calculate_bleu_wikitext(model, tokenizer, test_data):
    bleu_metric = evaluate.load("sacrebleu")
    predictions = []
    references = []

    for text in test_data["text"]:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)
        outputs = model.generate(**inputs, max_length=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append([text])

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    return bleu_score["score"]

# Perplexity 측정
input_texts = test_data["text"]
perplexity = calculate_perplexity(model, tokenizer, input_texts)
print(f"Perplexity: {perplexity}")

# BLEU 점수 측정
bleu_score = calculate_bleu_wikitext(model, tokenizer, test_data)
print(f"BLEU Score: {bleu_score}")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import numpy as np
# Perplexity 계산 함수
def calculate_perplexity(model, tokenizer, input_texts):
    model.eval()
    total_loss = 0
    total_length = 0
    with torch.no_grad():
        for text in input_texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            outputs = model(**inputs, max_length = 50, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)  # 배치 내 전체 토큰 수 곱하기
            total_length += inputs["input_ids"].size(1)
    perplexity = np.exp(total_loss / total_length)
    return perplexity

# BLEU 점수 계산 함수
def calculate_bleu(model, tokenizer, test_data):
    bleu_metric = evaluate.load("sacrebleu")  # evaluate 라이브러리에서 BLEU metric 로드
    predictions = []
    references = []
    for data in test_data:
        # max_length를 토크나이저에서만 설정 (최대 길이 512로 잘라냄)
        inputs = tokenizer(data["input"], return_tensors="pt", truncation=True, padding=True, max_length=512).to(model.device)

        # 생성 시에는 max_length만 설정
        outputs = model.generate(**inputs, max_length=50)  # 생성된 텍스트의 최대 길이 설정
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(generated_text)
        references.append([data["reference"]])

    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    return bleu_score["score"]

# 테스트 데이터 준비
test_data = [
    {"input": "What is AI?", "reference": "AI stands for artificial intelligence, which simulates human intelligence in machines."},
    {"input": "Explain quantum mechanics.", "reference": "Quantum mechanics is a fundamental theory in physics describing the behavior of particles on very small scales."},
    {"input": "What is the capital of France?", "reference": "The capital of France is Paris."},
]

# Perplexity 측정
input_texts = [data["input"] for data in test_data]
perplexity = calculate_perplexity(model, tokenizer, input_texts)
print(f"Perplexity: {perplexity}")

# BLEU 점수 측정
bleu_score = calculate_bleu(model, tokenizer, test_data)
print(f"BLEU Score: {bleu_score}")

# 추론 단계에서 로드
# 가중치 로드
quantized_model.load_state_dict(torch.load("quantized_model.pth"))

# 활성값 스케일 로드 및 적용
scale_factors = torch.load("activation_scales.pth")
for name, scale_factor in scale_factors.items():
    def forward_hook(module, inputs, outputs):
        return outputs * scale_factor
    layer = dict(quantized_model.named_modules())[name]
    layer.register_forward_hook(forward_hook)

# 양자화된 모델로 추론 실행
response = generate_response(quantized_model, tokenizer, "What's the weather like?")
print(response)
