import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import pickle
import tempfile

# DEMO_VIDEO = 'muscleuptest1.mp4'

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

with open('muscleup.pkl', 'rb') as f:
    model = pickle.load(f)

landmarks = ['class']
for val in range(1, 33+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

def main():
    st.title('Muscle-Up Detection App')
    st.sidebar.title('Muscle-Up Detection App')
    st.sidebar.subheader('Parameters')

    # Radio button to select input source
    selected_input = st.sidebar.radio('Select Input', ['Webcam', 'Upload Video'])

    if selected_input == 'Webcam':
        cap = cv2.VideoCapture(0)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_file.name)
        else:
            st.error("Please upload a video file.")
            return

    stframe = st.empty()

    counter = 0
    current_stage = ''

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR

            try:
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmarks[1:])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_log_proba(X)[0]

                if body_language_class == 'down' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                    current_stage = 'down'
                elif current_stage == 'down' and body_language_class == 'up' and body_language_prob[body_language_prob.argmax()] >= 0.7:
                    current_stage = 'up'
                    counter += 1
                    print(counter)

                cv2.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)
                cv2.putText(image, 'CLASS', (95, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, body_language_class.split(' ')[0], (90, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)], 2)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'COUNT', (180, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(counter), (175, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                pass

            stframe.image(image, channels="BGR", use_column_width=True)  # Specify BGR channels
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
