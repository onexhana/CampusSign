아래는 복사해서 GitHub README에 바로 붙여넣을 수 있는 **Markdown 포맷 최종 버전**입니다. 이미지, 링크 없이 기본 텍스트와 마크업만 사용해 가독성과 복붙 호환성을 최대로 유지했습니다.

---

```markdown
# 📹 CampusSign - 실시간 수어 번역기

> 대학생을 위한 상황별 수어 인식 및 자연어 문장 출력 시스템

---

## 📌 프로젝트 개요

**CampusSign**은 수어 영상을 입력받아  
- 수어 단어를 인식하고  
- 해당 의미에 맞는 자연어 문장을 출력하는  

실시간 수어 번역 웹 애플리케이션입니다.  
대학교 캠퍼스에서 자주 사용하는 수어 표현(출결, 과제, 도서관 등)에 최적화되어 있습니다.

---

## 🎯 주요 기능

- 🎥 수어 영상 업로드(.mp4, .mkv)
- 📸 실시간 웹캠 입력 (로컬 환경 전용)
- 🧠 LSTM 기반 단어 인식
- 📄 자연어 문장 출력
- 🔍 Mediapipe 기반 키포인트 추출 및 자동 전처리 (.json → .npy)

---

## 🧩 프로젝트 구조

```

CampusSign/
├── app/
│   ├── extractor.py
│   ├── preprocess.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── run\_pipeline.py
│   ├── label\_mapping.json
│   └── sentence\_templates.json
├── train\_runner.py
├── streamlit\_app.py
├── npy\_data/
├── assets/sign\_videos/
├── requirements.txt
└── README.md

```

---

## 🚀 실행 방법

### 1. 의존성 설치
```

pip install -r requirements.txt

```

### 2. 모델 학습
```

python train\_runner.py

```

### 3. Streamlit 실행
```

streamlit run streamlit\_app.py

```

---

## 🧠 사용 기술

- Python, PyTorch
- MediaPipe Holistic
- OpenCV
- Streamlit

---

## 예시 시나리오

| 상황        | 예시 단어       | 출력 문장 예시                        |
|-------------|------------------|----------------------------------------|
| 출결        | 지각, 출석       | 지각했습니다. 출석 가능할까요?        |
| 수업 요청   | 질문, 다시       | 이 부분 다시 설명해 주세요.          |
| 과제 안내   | 과제, 제출       | 과제 제출일이 언제인가요?            |
| 캠퍼스 생활 | 식당, 도서관     | 도서관은 어디에 있어요?              |

---

## 🔧 향후 계획

- 실시간 웹캠 수어 → 다단어 문장 조합
- Streamlit Cloud 배포
- 문장 음성 출력 기능 추가
- 시나리오/도메인 확장

---

## 📩 문의

- 개발자: CampusSign 팀
- 이메일: your_email@example.com
- 라이선스: MIT License

---
```

---

📌 `your_email@example.com`은 실제 메일 주소로 바꾸시고, 원하시면 팀원 정보나 배포 링크도 추가 가능합니다.
필요하면 `.gif` 시연 이미지나 사용 예시도 첨부해드릴 수 있어요.
