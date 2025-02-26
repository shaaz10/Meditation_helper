import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time

# Function to calculate the Eye Aspect Ratio (EAR)
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

# Streamlit UI setup
st.title("Meditation Timer")
meditation_time = st.slider("Set your meditation time (minutes)", 0.5, 10.0, 5.0, step=0.5)
time_limit = meditation_time * 60

# MediaPipe FaceMesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video capture setup
cap = cv2.VideoCapture(0)

# Thresholds and initial variables
eye_closed_threshold = 0.2
meditation_score = 0
meditation_threshold = 80
meditation_started = False
start_time = None
points = 0
session_count = 0

# Eye landmark indices
eye_indices = {
    "left": [33, 160, 158, 133, 153, 144],
    "right": [362, 385, 387, 263, 373, 380]
}

# Streamlit placeholders
frame_placeholder = st.empty()
success_message_placeholder = st.empty()

# Audio playback function for Streamlit
def play_audio(file_path):
    with open(file_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

# Meditation audio files
mp3_file = "om.mp3"
completion_sound = "su.mp3"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
            
            # Start meditation music when threshold is reached
            if meditation_score >= meditation_threshold and not meditation_started:
                meditation_started = True
                play_audio(mp3_file)
            
            if meditation_started:
                elapsed_time = time.time() - start_time if start_time else 0
                if elapsed_time >= time_limit:
                    meditation_started = False
                    points = 100  # Full meditation time completed

                    # Play completion sound when session is done
                    play_audio(completion_sound)

                    # Increment the session count
                    session_count += 1

                    # Show success message and animation
                    success_message_placeholder.markdown(
                        "<h1 style='color: green;'>Congratulations! You've completed today's session!</h1>",
                        unsafe_allow_html=True
                    )
                    success_message_placeholder.image("dd.gif", use_column_width=True)
                else:
                    points = int((elapsed_time / time_limit) * 100)
                
            # Display stats on the video frame
            font_scale = 1.5
            cv2.putText(frame, f"Points: {points}%", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 3)
            cv2.putText(frame, f"Meditation Score: {meditation_score}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 3)
            cv2.putText(frame, f"Sessions Completed: {session_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 3)
    
    # Convert the frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    time.sleep(0.03)

cap.release()

# Final meditation progress summary
st.write(f"You earned {points}% of your goal!")
st.write(f"Total Sessions Completed: {session_count}")
