# SOVYN-85M: 처음부터 학습한 한국어 추론 모델

<p align="center">
  <b>85.4M 파라미터 | 86.5% 벤치마크 | 순수 from-scratch</b>
</p>

---

## 1. 프로젝트 개요

SOVYN-85M은 **사전학습된 모델 없이 처음부터(from-scratch)** 학습한 한국어 추론 전문 언어 모델이다.

수학, 코드 트레이싱, 논리, 방정식 등 **119개 카테고리**의 추론 문제를 단계별로 풀이할 수 있다.

- **이름**: SOVYN = **Solve + Synapse** (문제 해결 + 신경 연결)
- **목표**: 한국어 추론 능력을 가진 소형 LLM의 가능성 검증 (PoC)
- **특징**: 외부 사전학습 모델 의존 없이, 합성 데이터만으로 86.5% 달성

## 2. 모델 아키텍처

```
SOVYN-85M (Decoder-only GPT)
├── Token Embedding (16,384 → 768)
├── Positional Embedding (512 → 768)
├── Transformer Block × 12
│   ├── LayerNorm → Causal Self-Attention (12 heads, Flash Attention)
│   └── LayerNorm → FeedForward (768 → 3072 → 768, GELU)
├── LayerNorm (final)
└── LM Head (weight-tied with Token Embedding)
```

| 항목 | 값 |
|------|-----|
| 총 파라미터 | **85,410,816 (85.4M)** |
| 아키텍처 | Decoder-only Transformer (GPT) |
| 레이어 수 | 12 |
| 어텐션 헤드 | 12 |
| 임베딩 차원 | 768 |
| FFN 히든 | 3,072 (4×embed) |
| 컨텍스트 길이 | 512 토큰 |
| 어휘 크기 | 16,384 (ByteLevel BPE) |
| 어텐션 | Flash Attention (scaled_dot_product) |
| 활성화 함수 | GELU |
| 정규화 | LayerNorm (Pre-LN) |
| Weight Tying | ✅ (Embedding ↔ LM Head) |
| 정밀도 | float16 |

## 3. 학습 데이터

| 항목 | 값 |
|------|-----|
| 데이터 유형 | 합성 추론 문제 (Synthetic Reasoning) |
| 총 문제 수 | **591,261개** |
| 카테고리 | **119종** |
| 총 토큰 | **27,970,000 (27.97M)** |
| 형식 | `문제: {질문}\n풀이:\n{단계별 풀이}\n답: {정답}` |

### 카테고리 분류

| 분야 | 포함 내용 |
|------|----------|
| 수학 | 산술, 방정식, 괄호 연산, 연산 우선순위, 수열, 미적분 |
| 코딩 | 코드 트레이싱, 알고리즘, 논리 연산 |
| 과학 | 물리, 화학, 생물, 지구과학 |
| 인문 | 한국사, 논리 추론 |
| 기타 | 리스트 연산, 숫자 성질, 서술형 문제 등 |

## 4. 학습 설정

| 항목 | 값 |
|------|-----|
| 옵티마이저 | AdamW |
| Learning Rate | 3e-4 |
| LR 스케줄 | Cosine Decay |
| Warmup | 500 steps |
| Weight Decay | 0.1 |
| 배치 크기 | 16 × 4 (grad accumulation) = **effective 64** |
| 총 스텝 | 20,000 |
| Gradient Clipping | 1.0 |
| AMP | float16 |
| GPU | NVIDIA GeForce RTX 5080 Laptop (16GB VRAM) |
| 학습 시간 | **~4시간** |

## 5. 벤치마크 결과

자체 벤치마크 **52문제, 10개 카테고리**.

| 카테고리 | 정답 | 총 문제 | 정확도 |
|---------|------|--------|--------|
| 산술 (기본 사칙연산) | 5 | 5 | **100%** |
| 코드 트레이싱 | 5 | 5 | **100%** |
| 숫자 성질 (소수/짝홀) | 5 | 5 | **100%** |
| 서술형 (응용 문제) | 5 | 5 | **100%** |
| 연산 우선순위 | 7 | 8 | **87.5%** |
| 리스트 연산 | 5 | 6 | **83.3%** |
| 괄호 연산 | 4 | 5 | **80.0%** |
| 방정식 | 4 | 5 | **80.0%** |
| 논리 | 4 | 5 | **80.0%** |
| 수열 | 1 | 3 | **33.3%** |
| **전체** | **45** | **52** | **86.5%** |

### 벤치마크 분석

- **완벽 카테고리 (100%)**: 산술, 코드 트레이싱, 숫자 성질, 서술형 — 패턴이 명확한 문제에서 강점
- **우수 카테고리 (80%+)**: 연산 우선순위, 리스트, 괄호, 방정식, 논리 — 복합 추론도 대부분 가능
- **약점 카테고리 (33%)**: 수열 — 등비수열/피보나치 같은 패턴 인식은 아직 부족

## 6. 추론 예시

### 입력
```
문제: 3x + 7 = 22일 때, x의 값을 구하시오.
풀이:
```

### 출력
```
3x + 7 = 22에서
3x = 22 - 7
3x = 15
x = 15 / 3
x = 5
답: 5
```

## 7. 사용법

### 설치
```bash
pip install torch safetensors tokenizers huggingface_hub
```

### 추론
```python
import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

# 모델 다운로드
model_path = hf_hub_download("SOVYN/SOVYN-85M", "model.safetensors")
tok_path = hf_hub_download("SOVYN/SOVYN-85M", "tokenizer.json")
code_path = hf_hub_download("SOVYN/SOVYN-85M", "model.py")

# 아키텍처 로드
import importlib.util
spec = importlib.util.spec_from_file_location("model", code_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 모델 로드
model = mod.SOVYN85M()
state_dict = load_file(model_path)
state_dict = {k: v.float() for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval()

tokenizer = Tokenizer.from_file(tok_path)

# 추론
prompt = "문제: 7 + 8을 계산하시오.\n풀이:\n"
ids = torch.tensor([tokenizer.encode(prompt).ids])
out = model.generate(ids, max_new_tokens=200, temperature=0.3)
print(tokenizer.decode(out[0].tolist()))
```

## 8. 프로젝트 타임라인

| 단계 | 내용 |
|------|------|
| 1단계 | 타이타닉 Kaggle 문제 → 머신러닝 입문 |
| 2단계 | 8M 파라미터 소형 GPT 실험 |
| 3단계 | **85M 파라미터 추론 모델 (이 보고서)** |
| 4단계 | HuggingFace 업로드 (SOVYN/SOVYN-85M) |
| 5단계 | 1.1B 파라미터 LLaMA-style 한국어 모델 (진행 중) |

## 9. 한계점

- **합성 데이터 전용**: 자유 대화, 번역, 요약 등은 불가
- **수열 약점**: 등비수열, 피보나치 등 복잡한 패턴 인식 부족
- **컨텍스트 제한**: 512 토큰 (긴 문제는 잘림)
- **한국어 전용**: 영어 추론 능력은 학습되지 않음
- **85M 크기 한계**: 복잡한 다단계 추론은 어려움

## 10. 후속 프로젝트: SOVYN-1.1B

현재 **SOVYN-1.1B** 모델을 학습 중이다.

| | SOVYN-85M | SOVYN-1.1B |
|---|---|---|
| 파라미터 | 85.4M | 1,082M |
| 아키텍처 | GPT | LLaMA-style |
| Attention | MHA | GQA (4 KV heads) |
| Norm | LayerNorm | RMSNorm |
| Position | Absolute | RoPE |
| FFN | GELU | SwiGLU |
| 데이터 | 28M 토큰 (합성) | 1.74B 토큰 (실제 한국어) |
| 용도 | 추론 PoC | 범용 한국어 모델 |

## 라이선스

Apache-2.0

## 링크

- **HuggingFace**: [SOVYN/SOVYN-85M](https://huggingface.co/SOVYN/SOVYN-85M)
- **GitHub**: [gaedsstudio/SOVYN-85M](https://github.com/gaedsstudio/SOVYN-85M)

---

*SOVYN (Solve + Synapse) — 문제 해결과 신경 연결의 합성어*
