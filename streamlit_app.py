import streamlit as st
import numpy as np
import tempfile
import os
import cv2
import json
import torch
import shutil

from app.extractor import extract_keypoints_from_video
from app.preprocess import convert_json_to_sequence
from app.model import load_model

# 설정
MODEL_PATH = "trained_model.pt"
LABEL_MAP_PATH = "app/label_mapping.json"
SENTENCE_TEMPLATE_PATH = "app/sentence_templates.json"
INPUT_SIZE = 225
HIDDEN_SIZE = 128

# 모델 로딩
with open(LABEL_MAP_PATH, encoding="utf-8") as f:
    label_mapping = json.load(f)

label_set = set()
for v in label_mapping.values():
    label_set.update(v) if isinstance(v, list) else label_set.add(v)
label_list = sorted(label_set)
label2idx = {label: i for i, label in enumerate(label_list)}
idx2label = {i: label for label, i in label2idx.items()}
num_classes = len(label2idx)

model = load_model(MODEL_PATH, INPUT_SIZE, HIDDEN_SIZE, num_classes)

# Streamlit UI
st.title("📹 CampusSign 실시간 수어 번역기")
st.write("🟢 웹캠/영상 입력 → 수어 단어 인식 → 자연어 문장 출력")

mode = st.radio("입력 방식 선택", ["🎥 영상 업로드", "📸 실시간 웹캠 (데모용)"])

# 템플릿 로딩
with open(SENTENCE_TEMPLATE_PATH, encoding="utf-8") as f:
    templates = json.load(f)

predicted_words = []

def predict_from_video(video_path):
    json_path = video_path.replace(".mp4", ".json").replace(".mkv", ".json")
    try:
        extract_keypoints_from_video(video_path, json_path)
    except Exception as e:
        st.error(f"❌ 키포인트 추출 중 오류: {e}")
        return None

    if not os.path.exists(json_path):
        st.error("❌ .json 파일이 생성되지 않았습니다.")
        return None

    try:
        with open(json_path, encoding='utf-8') as f:
            kp_data = json.load(f)
        sequence_array = convert_json_to_sequence(kp_data)
        if sequence_array is None:
            st.error("❌ 키포인트 시퀀스가 비어있거나 변환 실패.")
            return None
    except Exception as e:
        st.error(f"❌ JSON 처리 중 오류 발생: {e}")
        return None

    try:
        x = torch.tensor(sequence_array[0], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            y_pred = model(x)
            pred_label = idx2label[torch.argmax(y_pred).item()]
            return pred_label
    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {e}")
        return None

# 🎥 영상 업로드 모드
if mode == "🎥 영상 업로드":
    uploaded_file = st.file_uploader("수어 영상 업로드 (.mp4, .mkv)", type=["mp4", "mkv"])
    if uploaded_file:
        suffix = os.path.splitext(uploaded_file.name)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
            shutil.copyfileobj(uploaded_file, temp_video)
            temp_path = temp_video.name

        pred = predict_from_video(temp_path)
        if pred:
            st.success(f"🧠 예측된 단어: **{pred}**")
            sentence = templates.get(pred)
            if sentence:
                st.info(f"📝 출력 문장: {sentence}")
            else:
                st.warning(f"ℹ️ '{pred}'에 대한 문장 템플릿이 없습니다.")
        else:
            st.warning("❗ 예측 결과가 없습니다.")

# 📸 실시간 웹캠 모드
elif mode == "📸 실시간 웹캠 (데모용)":
    st.warning("⚠️ 이 모드는 데스크탑에서만 작동하며, Streamlit Cloud에서는 지원되지 않습니다.")
    run_webcam = st.button("실시간 예측 시작")
    if run_webcam:
        st.info("📸 웹캠을 열고 한 단어 수어를 보여주세요. 예: '출석', '과제' 등")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ 웹캠을 열 수 없습니다. 다른 앱이 사용 중일 수 있습니다.")
        else:
            frame_list = []
            for _ in range(60):  # 약 2초간 프레임 수집
                ret, frame = cap.read()
                if not ret:
                    st.error("웹캠에서 프레임을 가져올 수 없습니다.")
                    break
                frame = cv2.flip(frame, 1)
                frame_list.append(frame)
            cap.release()

            if frame_list:
                temp_path = os.path.join(tempfile.gettempdir(), "webcam_clip.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_path, fourcc, 15.0, (frame.shape[1], frame.shape[0]))
                for frame in frame_list:
                    out.write(frame)
                out.release()

                pred = predict_from_video(temp_path)
                if pred:
                    predicted_words.append(pred)
                    st.success(f"🧠 예측된 단어: **{pred}**")
                    st.write(f"🧩 현재 인식된 단어들: `{predicted_words}`")
                    if pred in templates:
                        st.info(f"📝 출력 문장 예시: {templates[pred]}")
                    else:
                        st.warning(f"ℹ️ '{pred}'에 대한 문장 템플릿이 없습니다.")
                else:
                    st.warning("❗ 예측 결과가 없습니다.")
