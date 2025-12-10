import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "./models/model_libras.h5"
LABELS = np.load("./data/dataset_final.npz", allow_pickle=True)["labels"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def extrair_landmarks(hand_landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()

def prever_em_tempo_real():
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(max_num_hands=1) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                lm = extrair_landmarks(hand).reshape(1, -1)

                pred = model.predict(lm)[0]
                idx = np.argmax(pred)
                letra = LABELS[idx]

                cv2.putText(frame, f"Pred: {letra}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            cv2.imshow("Teste LIBRAS", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Pressione Q para sair.")
    prever_em_tempo_real()
