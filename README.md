# aiv_coding
## Virtual env

```
conda create --name py38_env python=3.8
conda activate py38_env
pip install -r requirements.txt
```

## Data Generation

- 현재 디렉토리에 train 폴더가 존재해야함
- train folder 안에는 20장의 bmp와 json pair가 존재함

```
./data_generate.sh
```

## Train CLIP

```
./clip_train.sh
```

## Train Header
- 학습된 CLIP 모델을 가져와서 Header만 학습하는 코드

```
./train.sh
```

