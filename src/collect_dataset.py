import cv2
import mediapipe as mp
import numpy as np
import os

LETRAS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]     # escolha as letras que quer capturar
CAPTURAS_POR_LETRA = 300              # quantidade de frames por letra
DATASET_DIR = "data"
os.makedirs(DATASET_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

for letra in LETRAS:
    letra_dir = os.path.join(DATASET_DIR, letra)
    os.makedirs(letra_dir, exist_ok=True)

    print(f"\n➡ Prepare-se para capturar a letra: {letra}")
    input("Pressione ENTER quando estiver pronta...")

    count = 0

    while count < CAPTURAS_POR_LETRA:
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            # extrai (x, y, z) dos 21 pontos
            lm = np.array([[l.x, l.y, l.z] for l in hand.landmark]).flatten()

            # salva em .npy
            np.save(os.path.join(letra_dir, f"{letra}_{count}.npy"), lm)

            count += 1

            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Letra {letra} | Captura {count}/{CAPTURAS_POR_LETRA}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow("Coleta de Dados - LIBRAS", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

print("\n Dataset criado com sucesso!")
