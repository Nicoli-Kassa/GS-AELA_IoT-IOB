import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from modules.ear_fatigue import EARFatigueDetector
from modules.head_pose import HeadPoseDetector
from modules.rppg_heartrate import RPPGHeartRate

# ---------- Setup do FaceLandmarker (API tasks) ----------
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)

face = vision.FaceLandmarker.create_from_options(options)

# ---------- Detectores ----------
ear_det = EARFatigueDetector(thresh=0.22, consec_frames=20)
head_det = HeadPoseDetector(queda=0.15, normal=0.10)
rppg_det = RPPGHeartRate(tamanho_janela=300)

# ---------- Parametros ----------
LIMITE_ALERTA = 3          # acima/igual a isto -> recomenda descanso
LARGURA, ALTURA = 900, 600  # tamanho da janela

# ---------- Cores (BGR) ----------
FONTE = cv2.FONT_HERSHEY_SIMPLEX
BRANCO = (255, 255, 255)
CINZA = (180, 180, 180)
VERDE = (0, 255, 0)
VERDE_STATUS = (100, 230, 100)
VERMELHO_STATUS = (80, 80, 240)
BORDA = (200, 180, 120)
FUNDO_PAINEL = (40, 30, 20)
FUNDO_ALERTA = (20, 20, 90)


def painel(frame, x, y, titulo, subtitulo, label_valor, valor, status_ok):
    """Desenha um painel HUD semitransparente com titulo, valor e status."""
    largura, altura = 290, 92

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + largura, y + altura), FUNDO_PAINEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x, y), (x + largura, y + altura), BORDA, 1)

    cv2.putText(frame, titulo, (x + 12, y + 24), FONTE, 0.55, BRANCO, 1)
    cv2.putText(frame, subtitulo, (x + 12, y + 42), FONTE, 0.40, CINZA, 1)
    cv2.putText(frame, f"{label_valor}: {valor}", (x + 12, y + 74),
                FONTE, 0.55, BRANCO, 1)

    cor_status = VERDE_STATUS if status_ok else VERMELHO_STATUS
    texto_status = "Normal" if status_ok else "ALERTA"
    cv2.putText(frame, texto_status, (x + largura - 95, y + 74),
                FONTE, 0.55, cor_status, 2)


def alerta_descanso(frame, w, nods, episodios, limite):
    """Faixa de alerta no topo, indicando o que disparou o aviso."""
    # monta a lista de causas que ultrapassaram o limite
    causas = []
    if nods >= limite:
        causas.append(f"Postura - Quedas ({nods})")
    if episodios >= limite:
        causas.append(f"EAR - Fadiga ocular ({episodios})")
    motivo = " + ".join(causas)

    overlay = frame.copy()
    cv2.rectangle(overlay, (310, 10), (w - 10, 88), FUNDO_ALERTA, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.rectangle(frame, (310, 10), (w - 10, 88), VERMELHO_STATUS, 2)

    cv2.putText(frame, "FADIGA ELEVADA - Descansar", (325, 36),
                FONTE, 0.6, VERMELHO_STATUS, 2)
    cv2.putText(frame, f"Causa: {motivo}", (325, 64),
                FONTE, 0.5, BRANCO, 1)


# ---------- Loop principal ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (LARGURA, ALTURA))   # resize ANTES de processar

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = face.detect(mp_image)

    if result.face_landmarks:
        lm = result.face_landmarks[0]
        h, w, _ = frame.shape

        r_ear = ear_det.update(lm, w, h)
        r_head = head_det.update(lm, w, h)
        r_rppg = rppg_det.update(lm, frame, w, h)

        # --- Painel 1: EAR (fadiga ocular / cognitivo) ---
        painel(frame, 10, 10,
               "EAR - fadiga ocular", "modulo cognitivo",
               "EAR", f"{r_ear['ear']:.2f}",
               status_ok=not r_ear["fadiga"])

        # --- Painel 2: rPPG (frequencia cardiaca / cardiovascular) ---
        if not r_rppg["pronto"]:
            valor_fc = f"calib {r_rppg['progresso']}/{r_rppg['total']}"
            fc_ok = True
        elif r_rppg["bpm"] is not None:
            valor_fc = f"~{int(r_rppg['bpm'])} bpm"
            fc_ok = True
        else:
            valor_fc = "instavel"
            fc_ok = False
        painel(frame, 10, 112,
               "rPPG - freq. cardiaca", "modulo cardiovascular",
               "FC", valor_fc,
               status_ok=fc_ok)

        # --- Painel 3: Postura (head nod / fadiga fisica) ---
        painel(frame, 10, 214,
               "Postura - cabeca", "fadiga fisica",
               "NODS", f"{r_head['nods']}",
               status_ok=(r_head["estado"] != "caida"))

        # --- Alerta de fadiga acumulada ---
        if r_head["nods"] >= LIMITE_ALERTA or r_ear["episodios"] >= LIMITE_ALERTA:
            alerta_descanso(frame, w, r_head["nods"], r_ear["episodios"], LIMITE_ALERTA)

        # ROI da testa usada pelo rPPG (comente para tela limpa no video)
        if r_rppg["caixa"] is not None:
            x1, y1, x2, y2 = r_rppg["caixa"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), VERDE, 1)

    cv2.imshow("AELA Vision", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('r'):
        # reseta os contadores para repetir a demonstracao
        ear_det.episodios_fadiga = 0
        ear_det._em_fadiga = False
        head_det.eventos_nod = 0
        head_det.estado = "erguida"

cap.release()
cv2.destroyAllWindows()