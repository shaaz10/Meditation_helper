import cv2
import mediapipe as mp
import numpy as np
import pygame
import streamlit as st
import time

def eye_aspect_ratio(landmarks, left_eye_indices, right_eye_indices):
    def get_eye_ratio(eye_points):
        p1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        p2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        p3 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (p1 + p2) / (2.0 * p3)

    left_eye = [landmarks[i] for i in left_eye_indices]
    right_eye = [landmarks[i] for i in right_eye_indices]

    left_ratio = get_eye_ratio(left_eye)
    right_ratio = get_eye_ratio(right_eye)

    return (left_ratio + right_ratio) / 2.0

st.title("Meditation Timer")
meditation_time = st.slider("Set your meditation time (minutes)", 0.5, 10.0, 5.0, step=0.5)

time_limit = meditation_time * 60
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# Safely try initializing pygame audio
audio_enabled = True
try:
    pygame.mixer.init()
except pygame.error:
    st.warning("Audio playback is not supported in this environment.")
    audio_enabled = False

eye_closed_threshold = 0.2
meditation_score = 0
meditation_threshold = 80
meditation_started = False
mp3_file = "om.mp3"
completion_sound = "su.mp3"
start_time = None
points = 0
session_count = 0

eye_indices = {
    "left": [33, 160, 158, 133, 153, 144],
    "right": [362, 385, 387, 263, 373, 380]
}

frame_placeholder = st.empty()
success_message_placeholder = st.empty()

if not cap.isOpened():
    st.error("Failed to access the webcam. Please check permissions and ensure no other app is using the camera.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame from webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in face_landmarks.landmark]
            ear = eye_aspect_ratio(landmarks, eye_indices["left"], eye_indices["right"])

            if ear < eye_closed_threshold:
                meditation_score += 1
                if start_time is None:
                    start_time = time.time()
            else:
                meditation_score = max(0, meditation_score - 1)
                start_time = None

            if meditation_score >= meditation_threshold and not meditation_started:
                meditation_started = True
                if audio_enabled:
                    pygame.mixer.music.load(mp3_file)
                    pygame.mixer.music.play(-1)

            if meditation_started:
                elapsed_time = time.time() - start_time if start_time else 0
                if elapsed_time >= time_limit:
                    if audio_enabled:
                        pygame.mixer.music.stop()
                    meditation_started = False
                    points = 100

                    if audio_enabled:
                        pygame.mixer.music.load(completion_sound)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            time.sleep(0.1)

                    session_count += 1

                    success_message_placeholder.markdown(
                        "<h1 style='color: green;'>🎉 Congratulations! You've completed today's session! 🎉</h1>",
                        unsafe_allow_html=True
                    )
                    success_message_placeholder.image("dd.gif", use_column_width=True)
                else:
                    points = int((elapsed_time / time_limit) * 100)

            font_scale = 1.5
            cv2.putText(frame, f"Points: {points}%", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 3)
            cv2.putText(frame, f"Meditation Score: {meditation_score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 3)
            cv2.putText(frame, f"Sessions Completed: {session_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 3)

            if meditation_started and ear > eye_closed_threshold:
                if audio_enabled:
                    pygame.mixer.music.stop()
                meditation_started = False
                meditation_score = 0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    time.sleep(0.03)

cap.release()

st.write(f"You earned {points}% of your goal!")
st.write(f"Total Sessions Completed: {session_count}")
