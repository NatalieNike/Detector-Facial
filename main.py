import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque, Counter
 
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions,
    HandLandmarker, HandLandmarkerOptions,
    RunningMode,
)
 
 #Carregamento das imagens
IMAGES = {
    "sorriso": "imgs/sorriso.jpeg",
    "boca_aberta": "imgs/boca_aberta.jpeg",
    "sobrancelha": "imgs/sobrancelha.jpg",
    "neutro": "imgs/neutro.jpg",
    "mao_aberta": "imgs/mao_aberta.jpeg",
    "punho": "imgs/punho.jpeg",
    "joia": "imgs/joia.jpeg",
    "paz": "imgs/paz.jpg",
}
 
#Arquivos .task são modelos de rostos e mãos que as redes neurais do MediaPipe usa para detectar os landmarks (pontos da face e mão).
FACE_MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
 
THRESHOLDS = {
    "sorriso_ratio": 0.42,
    "boca_aberta_ratio": 0.055,
    "sobrancelha_ratio": 0.145,
}
 
DEBOUNCE_FRAMES = 6
 
 #Criação dos detectores uma única vez fora do loop, para melhorar a performance o modelo não pode ser carregado a cada frame.
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(face_options)
 
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=1,
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)
 
 
def dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

#classificação de expressão facial
 
def classify_expression(lm):
    """lm: lista de landmarks do rosto (result.face_landmarks[0])."""
    left_face = lm[234]
    right_face = lm[454]
    top_face = lm[10]
    bottom_face = lm[152]
    face_width = dist(left_face, right_face)
    face_height = dist(top_face, bottom_face)
 
    mouth_left = lm[61]
    mouth_right = lm[291]
    mouth_top = lm[13]
    mouth_bottom = lm[14]
 
    mouth_width = dist(mouth_left, mouth_right)
    mouth_open = dist(mouth_top, mouth_bottom)
 
    left_brow = lm[105]
    left_eye = lm[159]
    brow_eye_dist = dist(left_brow, left_eye)
 
    smile_ratio = mouth_width / face_width
    open_ratio = mouth_open / face_height
    brow_ratio = brow_eye_dist / face_height
 
    if open_ratio > THRESHOLDS["boca_aberta_ratio"]:
        return "boca_aberta"
    if brow_ratio > THRESHOLDS["sobrancelha_ratio"]:
        return "sobrancelha"
    if smile_ratio > THRESHOLDS["sorriso_ratio"]:
        return "sorriso"
    return "neutro"
 
#classificação de gesto de mão
 
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
 
 
def fingers_up(lm, handedness_label):
    """lm: lista de landmarks da mão (result.hand_landmarks[0])."""
    fingers = []
 
    if handedness_label == "Right":
        fingers.append(1 if lm[FINGER_TIPS[0]].x < lm[FINGER_PIPS[0]].x else 0)
    else:
        fingers.append(1 if lm[FINGER_TIPS[0]].x > lm[FINGER_PIPS[0]].x else 0)
 
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)
 
    return fingers
 
 
def classify_hand_gesture(lm, handedness_label):
    f = fingers_up(lm, handedness_label)
 
    if f == [0, 0, 0, 0, 0]:
        return "punho"
    if f == [1, 1, 1, 1, 1]:
        return "mao_aberta"
    if f == [1, 0, 0, 0, 0]:
        return "joia"
    if f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
        return "paz"
    return None
 
 
#loop principal
 
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível abrir a webcam.")
        return
 
    loaded_images = {}
    for nome, caminho in IMAGES.items():
        img = cv2.imread(caminho)
        if img is not None:
            loaded_images[nome] = img
 
    history = deque(maxlen=DEBOUNCE_FRAMES)
    estado_atual = "neutro"
 
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)
 
    start_time = time.time()
 
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int((time.time() - start_time) * 1000)
 
        estado_detectado = None
 
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks:
            hand_lm = hand_result.hand_landmarks[0]
            handedness = hand_result.handedness[0][0].category_name
            gesto = classify_hand_gesture(hand_lm, handedness)
            if gesto:
                estado_detectado = gesto
 
            h, w, _ = frame.shape
            for ponto in hand_lm:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
 
        if estado_detectado is None:
            face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
            if face_result.face_landmarks:
                face_lm = face_result.face_landmarks[0]
                estado_detectado = classify_expression(face_lm)
 
        if estado_detectado is None:
            estado_detectado = "neutro"
 
        history.append(estado_detectado)
        mais_comum, contagem = Counter(history).most_common(1)[0]
        if contagem == len(history):
            estado_atual = mais_comum
 
        cv2.putText(
            frame, f"Estado: {estado_atual}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA
        )
        cv2.imshow("Webcam", frame)
 
        if estado_atual in loaded_images:
            cv2.imshow("Resultado", loaded_images[estado_atual])
 
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()
    hand_landmarker.close()
 
 
if __name__ == "__main__":
    main()
 
