# Artificial Intelligence Course Project

인공지능 수업에서 수행한 주요 과제와 최종 프로젝트를 정리한 저장소입니다.

신경망 기초 구현부터 CNN, RNN, VQA 프로젝트까지 인공지능 모델의 기본 구조와 학습 과정을 실습했습니다.

---

## 프로젝트 구성

### `neural-network-from-scratch.ipynb`
NumPy를 활용해 기본 신경망 구조를 직접 구현한 과제입니다.  
딥러닝 프레임워크의 자동 미분 기능에 의존하지 않고, forward propagation과 backward propagation의 계산 흐름을 직접 구현하며 신경망 학습 과정의 내부 동작을 학습했습니다.

주요 내용:
- Fully Connected Neural Network 구현
- Forward / Backward propagation 구현
- Loss 계산 및 gradient update 과정 이해
- NumPy 기반 행렬 연산 활용

---

### `cnn-classification.ipynb`
CNN(Convolutional Neural Network)을 활용한 이미지 분류 과제입니다.  
합성곱 계층과 pooling 구조를 통해 이미지의 공간적 특징을 추출하고, 분류 모델을 학습하는 과정을 실습했습니다.

주요 내용:
- CNN 기반 image classification
- Convolution / pooling layer 구조 이해
- PyTorch 기반 모델 학습
- Training / validation 과정 비교

---

### `vanilla-rnn.ipynb`
Vanilla RNN의 forward 및 backward 과정을 구현한 과제입니다.  
순차 데이터가 시간 단계별 hidden state를 통해 처리되는 방식을 학습하고, recurrent 구조에서 gradient가 전파되는 과정을 확인했습니다.

주요 내용:
- Vanilla RNN 구조 이해
- Hidden state 업데이트 과정 구현
- Forward pass / backward pass 구현
- Sequential data 처리 방식 학습

---

### `vqa-final-project.ipynb`
Visual Question Answering(VQA)을 주제로 수행한 최종 프로젝트입니다.  
이미지와 자연어 질문을 함께 입력으로 받아 정답을 예측하는 멀티모달 태스크를 다뤘으며, 시각 정보와 텍스트 정보를 함께 활용하는 모델 구조를 실험했습니다.

주요 내용:
- VQA task 이해
- Image feature와 text feature 결합
- Multimodal model 구조 실험
- 최종 프로젝트 모델 학습 및 결과 분석

---

## 주요 학습 내용

- Neural Network 기본 구조 이해
- Forward / Backward propagation 구현
- CNN 기반 이미지 특징 추출
- RNN 기반 순차 데이터 처리
- VQA 모델 구조 이해
- PyTorch 기반 모델 학습 및 실험

---

## 사용 기술

- Python
- NumPy
- PyTorch
- Jupyter Notebook / Google Colab

---

## 비고

수업에서 제공된 데이터셋, 모델 가중치, 제출 파일은 저장소에 포함하지 않았습니다.
