# Artificial Intelligence Course Project

이화여자대학교 인공지능(2026 Spring) 과제 및 최종 프로젝트 정리 저장소.
과제 노트북 골격과 평가 코드는 수업 제공 자료(Ewha AGI Lab, DSAIL@SNU 자료 기반)이며, 각 항목의 **직접 구현** 범위를 아래에 구분해 표기.

| 노트북 | 주제 | 직접 구현 | 결과 |
| --- | --- | --- | --- |
| [`vqa-final-project.ipynb`](notebooks/vqa-final-project.ipynb) | **최종 프로젝트** · VLM visual token 압축 | 학습 없는 token 선별 알고리즘 설계 · 구현 | token 576 → 64, accuracy 80.0% |
| [`neural-network-from-scratch.ipynb`](notebooks/neural-network-from-scratch.ipynb) | NumPy 신경망 | 2-hidden-layer 신경망 forward / backward | test accuracy 88.5% |
| [`cnn-classification.ipynb`](notebooks/cnn-classification.ipynb) | CNN 이미지 분류 | Inception module 및 개선 모델 설계 | test accuracy 80% |
| [`vanilla-rnn.ipynb`](notebooks/vanilla-rnn.ipynb) | Vanilla RNN | step · sequence 단위 forward / backward | gradient check 전 항목 통과 |

<br/>

## 최종 프로젝트 — Visual Token Compression for Efficient VQA

### 문제 정의
- LLaVA-Phi-3-Mini의 이미지 입력: 이미지 1장당 visual token 576개(24 × 24 grid)
- visual token이 입력 시퀀스의 대부분을 차지, 추론 비용 증가 요인
- 목표: **추가 학습 없이** token 64개만 선별해 VQA 정확도 유지 (token 수 64개 고정 조건)

### 접근 — Hybrid Top-k + Spatial CLS-Norm (32 : 32)

```text
576 visual tokens (24 × 24 grid)
        ↓
token별 점수 = z(feature norm) + 1.5 × z(CLS 유사도)
        ↓
 ┌──────────────────────────┬──────────────────────────────┐
 │ Global Top-k  32개        │ Spatial  32개                 │
 │ 전체 점수 상위 token       │ 8 × 4 영역(3 × 6 블록)별 최고점 │
 └──────────────────────────┴──────────────────────────────┘
        ↓
중복 제거 → 부족분은 점수순 보충 → 64개 확정 → 원래 위치 순서로 정렬
```

- **Feature norm**: 정보량이 큰 token 우선
- **CLS 유사도**: 이미지 전체 요약(CLS token)과 가까운 token 우선, CLS token을 projector에 함께 통과시켜 같은 특징 공간에서 비교
- **Spatial 선택**: 점수 상위 token의 특정 영역 쏠림 방지, 이미지 전 영역의 최소 커버리지 확보
- 구현 위치: `select_image_tokens()` 및 CLS feature를 함께 반환하도록 수정한 `get_compressed_image_features()`

### 결과

| 항목 | 값 | 조건 |
| --- | ---: | --- |
| visual token | **576 → 64** | 토큰 수 고정 조건 |
| Token Reduction Efficiency | **88.89%** | 1 − 64 / 576 |
| Accuracy | **80.0%** | 라벨 제공 30문항, 객관식(A/B…) 정답 일치 |

- 측정 범위: 정확도와 토큰 수. 실제 추론 시간 · 메모리 감소는 별도 측정하지 않음
- 평가 표본이 30문항으로 작아 문항 1개당 정확도 약 3.3%p 변동

### 실행 환경
- Google Colab (GPU), `transformers==4.57.2`, 모델 `xtuner/llava-phi-3-mini-hf`
- 수업 제공 `utils/`(모델 · 평가 코드)와 `data/` 필요

<br/>

## 과제

### NumPy 신경망 (`neural-network-from-scratch.ipynb`)
- notMNIST 10-class 분류, 프레임워크 자동 미분 없이 NumPy만 사용
- forward propagation · cross-entropy loss · backpropagation · batch gradient descent 구현
- hidden layer 크기 변화에 따른 과적합 경향 비교, 2-hidden-layer 모델로 확장
- 결과: train 96.5% / validation 88.9% / test 88.5%

### CNN 이미지 분류 (`cnn-classification.ipynb`)
- PyTorch 기반 CIFAR-10 분류
- 기본 CNN(test 56%) → 1×1 · 3×3 · 5×5 · pooling 분기의 Inception module 구현 → batch normalization 포함 개선 모델 설계
- 결과: test accuracy 80% (과제 기준 70% 이상)

### Vanilla RNN (`vanilla-rnn.ipynb`)
- 프레임워크 없이 vanilla RNN 구현: 단일 timestep 및 전체 sequence의 forward / backward
- 수치 gradient와 비교한 검증 4종 모두 기준 이내 통과 (상대 오차 1e-7 이하)

<br/>

## 기술
`Python` `NumPy` `PyTorch` `Hugging Face Transformers` `LLaVA-Phi-3-Mini` `Google Colab`

## 비고
수업 제공 데이터셋, 모델 가중치, `utils/` 코드, 제출 파일은 저장소에 포함하지 않음.
