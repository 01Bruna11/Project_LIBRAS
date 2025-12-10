import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam não abriu!")
    exit()

print("✅ Webcam abriu! Mostrando vídeo...")

while True:
    ok, frame = cap.read()
    if not ok:
        print("❌ Falha ao ler frame!")
        break

    cv2.imshow("Webcam Teste", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
