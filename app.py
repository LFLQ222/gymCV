import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load the Muscle-Up model
with open('muscleup.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the pose landmarks
landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

# Define the Muscle-Up Detection Transformer
class MuscleUpDetector(VideoTransformerBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.current_stage = ''

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        # Process the frame using Mediapipe
        results = self.mp_pose.process(image)

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks[1:])
            body_language_class = model.predict(X)[0]
            body_language_prob = model.predict_log_proba(X)[0]

            # Perform action based on Muscle-Up detection
            if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                self.current_stage = 'down'
            elif self.current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                self.current_stage = 'up'
                self.counter += 1
                print(self.counter)

            cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
            cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'COUNT', (180, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter), (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        except Exception as e:
            pass

        return image

def main():
    st.title('Muscle-Up Detection App')
    st.sidebar.title('Muscle-Up Detection App')
    st.sidebar.subheader('Parameters')

    # Radio button to select input source
    selected_input = st.sidebar.radio('Select Input', ['Webcam', 'Upload Video'])

    if selected_input == 'Webcam':
        webrtc_streamer(key="example", video_transformer_factory=MuscleUpDetector)
    else:
        st.error("This option is not supported yet.")

if __name__ == '__main__':
    main()
