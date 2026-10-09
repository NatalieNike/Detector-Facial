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
 
 #carregamento das imagens
IMAGES = {
    "sorriso": "imgs/sorriso.jpeg",
    "boca_aberta": "imgs/boca_aberta.jpeg",
    "sobrancelha": "imgs/sobrancelha.jpg",
    "neutro": "imgs/neutro.jpg",
    "mao_aberta": "imgs/mao_aberta.jpeg",
    "joia": "imgs/joia.jpeg",
    "paz": "imgs/paz.jpg",
    "olho_arregalado": "imgs/olho_arregalado.jpeg",
    "shiu": "imgs/shiu.jpeg",
    "indicador": "imgs/indicador.jpeg",
    "maos_cabeca": "imgs/maos_cabeca.jpeg",
    "triste": "imgs/triste.jpeg",
}
 
#arquivos .task são modelos de rostos e mãos que as redes neurais do MediaPipe usa para detectar os landmarks (pontos da face e mão).
FACE_MODEL_PATH = "models/face_landmarker.task"
HAND_MODEL_PATH = "models/hand_landmarker.task"
 
THRESHOLDS = {
    "sorriso_ratio": 0.42,
    "boca_aberta_ratio": 0.055,
    "sobrancelha_ratio": 0.170,
    "olho_arregalado_ratio": 0.070,
    "shiu_dist_ratio": 0.12,
    "cabeca_y_ratio": 0.45,
    "cabeca_x_ratio": 1.0,
    "triste_ratio": 0.015,
}
 
DEBOUNCE_FRAMES = 6
 
 #criação dos detectores uma única vez fora do loop, para não ser necessário carregar o modelo a cada frame.
face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=FACE_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
face_landmarker = FaceLandmarker.create_from_options(face_options)
 
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
)
hand_landmarker = HandLandmarker.create_from_options(hand_options)
 
 #visão computacional
def dist(p1, p2):
    return np.hypot(p1.x - p2.x, p1.y - p2.y)

#classificação de expressão facial, atribuição de valores para boca, sorriso, sobrancelha e etc.
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
    left_eye_top = lm[159]
    left_eye_bottom = lm[145]
    brow_eye_dist = dist(left_brow, left_eye_top)
    eye_open = dist(left_eye_top, left_eye_bottom)
 
    smile_ratio = mouth_width / face_width
    open_ratio = mouth_open / face_height
    brow_ratio = brow_eye_dist / face_height
    eye_open_ratio = eye_open / face_height
    
    # altura média dos cantos da boca vs. altura do centro dos lábios
    mouth_center_y = (mouth_top.y + mouth_bottom.y) / 2
    corners_y = (mouth_left.y + mouth_right.y) / 2
    frown_ratio = (corners_y - mouth_center_y) / face_height
 
    if eye_open_ratio > THRESHOLDS["olho_arregalado_ratio"]:
        return "olho_arregalado"
    if open_ratio > THRESHOLDS["boca_aberta_ratio"]:
        return "boca_aberta"
    if frown_ratio > THRESHOLDS["triste_ratio"]:
        return "triste"
    if brow_ratio > THRESHOLDS["sobrancelha_ratio"]:
        return "sobrancelha"
    if smile_ratio > THRESHOLDS["sorriso_ratio"]:
        return "sorriso"
    return "neutro"
 
#classificação de gesto de mão, tip representa a ponta do dedo e pip a articulação intermediária.
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
 
 #verificação da posição dos dedos, esticados ou encolhidos, determina o gesto da mão.
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
 
 
def classify_hand_gesture(lm, handedness_label, face_lm=None):
    f = fingers_up(lm, handedness_label)
 
    # shiu: só o indicador esticado (polegar ignorado) + ponta do dedo perto da boca
    if f[1:] == [1, 0, 0, 0] and face_lm is not None:
        index_tip = lm[8]
        mouth_x = (face_lm[13].x + face_lm[14].x) / 2
        mouth_y = (face_lm[13].y + face_lm[14].y) / 2
        face_height = dist(face_lm[10], face_lm[152])

        d = np.hypot(index_tip.x - mouth_x, index_tip.y - mouth_y) / face_height
        if d < THRESHOLDS["shiu_dist_ratio"]:
            return "shiu"
        
    if f[1:] == [1, 0, 0, 0]:
        return "indicador"
    
    if f == [1, 1, 1, 1, 1]:
        return "mao_aberta"
    if f == [1, 0, 0, 0, 0]:
        return "joia"
    if f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0:
        return "paz"
    return None

def maos_cabeca(hands_lm, face_lm):
    if len(hands_lm) < 2 or face_lm is None:
        return False

    face_height = dist(face_lm[10], face_lm[152])
    face_width = dist(face_lm[234], face_lm[454])
    center_x = (face_lm[234].x + face_lm[454].x) / 2
    limite_y = face_lm[10].y + THRESHOLDS["cabeca_y_ratio"] * face_height

    lados = []
    for lm in hands_lm[:2]:
        # centro aproximado da palma: média entre o pulso (0) e a base do dedo médio (9)
        px = (lm[0].x + lm[9].x) / 2
        py = (lm[0].y + lm[9].y) / 2

        if py > limite_y:   # mão abaixo da altura da cabeça
            return False
        if abs(px - center_x) > face_width * THRESHOLDS["cabeca_x_ratio"]:   # mão longe demais dos lados
            return False
        lados.append(px < center_x)

    return lados[0] != lados[1]   # uma mão de cada lado do rosto
 
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

        face_lm = None
        face_result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        if face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]

        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        hands_lm = hand_result.hand_landmarks or []

        # 1) gesto de duas mãos tem prioridade máxima
        if maos_cabeca(hands_lm, face_lm):
            estado_detectado = "maos_cabeca"
        else:
            # 2) gestos de uma mão: vale o primeiro reconhecido
            for hand_lm, handedness_list in zip(hands_lm, hand_result.handedness):
                gesto = classify_hand_gesture(hand_lm, handedness_list[0].category_name, face_lm)
                if gesto:
                    estado_detectado = gesto
                    break

        # desenha os pontos de todas as mãos detectadas
        h, w, _ = frame.shape
        for hand_lm in hands_lm:
            for ponto in hand_lm:
                cx, cy = int(ponto.x * w), int(ponto.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        # 3) sem gesto, cai para a expressão facial
        if estado_detectado is None and face_lm is not None:
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