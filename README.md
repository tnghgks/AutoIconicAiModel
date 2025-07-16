# AutoIconicModel 🎨

텍스트 프롬프트를 아웃 라인SVG 아이콘으로 변환하는 Transformer 기반 AI 모델입니다.
** 주의 ** 아직 완벽히 개발이 이루어져 있지 않습니다.

[✈️ 데모 링크](https://autoiconic-client.netlify.app/)

## 🚀 주요 기능

- **텍스트-투-SVG 변환**: 자연어 설명을 SVG 경로로 변환 (**현재는 간단한 영단어를 통한 다중분류모델입니다.**)
- **Transformer 아키텍처**: 인코더-디코더 구조로 아웃라인 SVG 생성
- **실시간 API 서버**: FastAPI 기반 웹 API 제공
- **인터랙티브 추론**: 명령행 인터페이스로 즉석 생성

## 🛠️ 설치 및 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/tnghgks/AutoIconicAiModel.git
cd AutoIconicAiModel
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. 요구사항
- Python 3.8+
- PyTorch 2.3.0+
- CUDA (선택사항, GPU 가속을 위해)

## 📊 데이터셋 구조

프로젝트는 다음과 같은 데이터셋을 사용합니다:

```
dataset/
├── icon-name.svg     # 해당 SVG 파일
└── ...
```

각 SVG 파일의 제목은 해당하는 벡터 그래픽을 설명합니다.
학습 데이터는 아래 링크를 통해 얻었으며, 비상업적인 용도로만 사용하였습니다.

[Feather](https://feathericons.com/)
[lucide](https://lucide.dev/icons/)



## 🔧 사용법

### 1. 데이터 전처리
```bash
python main.py preprocess
```

이 단계에서는:
- SVG 파일을 파싱하여 경로 토큰화
- 텍스트 데이터를 SentencePiece로 토큰화
- 훈련/검증 데이터셋 생성

### 2. 모델 훈련
```bash
python main.py train
```

훈련 과정:
- Transformer 인코더-디코더 모델 학습
- 배치 크기: 64
- 학습률: 1e-4
- 최대 에포크: 1000
- 모델 체크포인트는 `results/model/` 디렉토리에 저장

### 3. 추론 실행

#### 명령행 인터페이스
```bash
python main.py infer
```

인터랙티브 모드에서 프롬프트를 입력하면 즉시 SVG가 생성됩니다:
```
🔍 Prompt 입력 ('exit' 입력 시 종료): arrow pointing right
🖋 arrow pointing right to Svg:
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" stroke="black" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
<path d="M 5 12 L 19 12 M 12 5 L 19 12 L 12 19"/>
</svg>
```

#### API 서버
```bash
uvicorn apiServer:app --reload
```

API 엔드포인트: `POST /generate`

요청 예시:
```json
{
  "prompt": "house with chimney"
}
```

응답 예시:
```json
{
  "svg": "<svg viewBox=\"0 0 24 24\" xmlns=\"http://www.w3.org/2000/svg\" stroke=\"black\" fill=\"none\" stroke-width=\"2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M 3 9 L 12 2 L 21 9 V 20 A 1 1 0 0 1 20 21 H 4 A 1 1 0 0 1 3 20 V 9 Z\"/></svg>"
}
```

## 🏗️ 모델 아키텍처

### Transformer 기반 구조
- **인코더**: 텍스트 임베딩을 처리
- **디코더**: SVG 토큰을 순차적으로 생성
- **파라미터**:
  - 모델 차원: 256
  - 어텐션 헤드: 4개
  - 인코더/디코더 레이어: 각 3개
  - 드롭아웃: 0.3

### 토큰화 전략
- **텍스트**: SentencePiece BPE 토큰화
- **SVG**: 의미 단위 토큰화 (예: `M_3_3`, `L_4_4`, `Z`)
- **특수 토큰**: `<bos>`, `<eos>`, `<pad>`, `<unk>`

## 📁 프로젝트 구조

```
AutoIconicModel/
├── main.py                 # 메인 실행 스크립트
├── inference.py           # 추론 모듈
├── apiServer.py          # FastAPI 서버
├── config.py             # 설정 파일
├── requirements.txt      # 의존성 목록
├── preprocess/           # 데이터 전처리
│   ├── svg_tokenizer.py
│   ├── text_tokenizer.py
│   └── build_trainset.py
├── train/                # 모델 훈련
│   ├── model.py
│   ├── svg_dataset.py
│   └── to_weight.py
├── dataset/              # 원본 데이터셋
├── results/              # 모델 결과물
│   ├── model/           # 저장된 모델
│   ├── token2id.json    # SVG 토큰 사전
│   └── text_bpe.model   # 텍스트 토크나이저
└── README.md
```

## 🎯 성능 및 특징

- **최대 시퀀스 길이**: 512 토큰
- **지원 SVG 속성**: 경로(path) 기반 벡터 그래픽
- **실시간 생성**: 평균 응답 시간 < 1초
- **확장 가능**: 새로운 아이콘 카테고리 추가 가능

## 🔮 예시 사용 사례

```python
# 직접 사용
from inference import infer

svg_path = infer("circular arrow clockwise")
print(svg_path)  # "M 12 2 A 10 10 0 1 1 2 12 L 5 12 A 7 7 0 1 0 12 5 V 2 Z"
```

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.

## 🙏 감사의 말

- PyTorch 커뮤니티
- SentencePiece 개발팀
- FastAPI 프레임워크

---

📧 문의사항: [GitHub Issues](https://github.com/tnghgks/AutoIconicAiModel/issues)