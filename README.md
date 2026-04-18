# 로봇 조작 강건성 향상을 위한 MAE 이미지 표현 학습 시스템
본 시스템은 MAE(Masked Autoencoder) 기반 사전 학습을 활용해 시각 정보 기반 모방 학습에서의 데이터 부족 문제를 완화하고, 로봇 조작을 위한 안정적이고 풍부한 표현을 학습하도록 구성되어 있다. (1) 시연 영상을 활용해 ViT 백본을 **사전 학습**한 뒤, (2) **다운스트림** 단계에서는 해당 백본을 고정하거나 부분적으로만 미세조정하여 소량의 작업 특화 데이터로도 높은 행동 예측 성능을 달성한다. MAE 기반 복원 학습을 통해 물체 위치, 장면 구조, 시간적 흐름 등을 포괄적으로 모델링하며, 이는 기존 대조 학습 대비 더 효과적인 시퀀스 표현을 제공한다.

---
## 환경 설정
가상 환경 설정 및 파이썬 패키지 설치:
```bash
conda create -n robot_mae python=3.10
conda activate robot_mae
pip install -r requirements.txt
```
- 파이썬 버전 3.10에 대해 테스트함
- 필요에 따라 torch 버전을 파이썬 및 장치에 맞는 버전으로 설치

## 데이터 설정
테스트한 데이터는 다음 모델에서 사용한 데이터 활용:

> [**Visual Imitation Made Easy**](https://arxiv.org/pdf/2008.04899)  
> [CoRL 2020](https://www.robot-learning.org/)

커스텀 데이터를 활용할 경우에는 아래와 같은 데이터 디렉토리 형식으로 설정:
```bash
./
└── data/
    └── test/
        └── TESTGH000000_frames/
            └── images/
                ├── frame_000001.jpg
                ├── frame_000002.jpg
                ├── frame_000003.jpg
                └── ...
    └── train/
        └── ...
    └── val/
        └── ...
```

## 학습 및 평가
### (1) 사전 학습
1. `pretrain_mae.py`의 `params` dictionary의 `root_dir`, `train_dir`, `val_dir` 설정

    1.1 `train_layer_num`: 디폴트 값은 11로 frozen할 레이어를 설정함. (예: 11로 설정할 경우, 10번째 layer까지 학습하고 이후는 frozen)
2. `pretrain_mae.py`로 모델 학습
3. 학습 종료 후에 모델의 가중치는 `params` dictionary의 `save_dir`에 저장됨

    3.1 Wandb를 통해 학습 및 평가 상태 확인 가능

### 세부 설정
`pretrain_mae.py`의 `params` dictionary를 수정하여 각 옵션 조정 가능

(예: `epochs`, `patch_size`,`mask_ratio` 등)

### (2) 다운스트림 학습
1. 데이터 경로 설정: `downstream_train_mae.py`의 인자 값  `train_dir`, `val_dir`, `test_dir` 설정
2. 사전 학습 경로 설정: `downstream_train_mae.py`의 인자 값 `pretrain_path` 설정
3. `downstream_train_mae.py`로 모델 학습

    3.1 Wandb를 통해 학습 및 평가 상태 확인 가능
4. 학습 종료 이후, `./results` 폴더에서 이미지별 평가 결과를 시각적으로 확인 가능

## Acknowledgement
This repository is based on the codebase from [Visual-Imitation-Made-Easy](https://github.com/sarahisyoung/Visual-Imitation-Made-Easy/). Thanks for their excellent work.

## Contributors
Original authors: [홍연주](https://github.com/lightorange0v0), [이성재](https://github.com/ubless607).  
