import time
from collections import deque 
import numpy as np 

class RPPGHeartRate:
    """
    Estimativa OPTICA de frequencia cardiaca via variacao de cor da pele (rPPG).
    Modulo experimental: exige luz estavel e rosto parado. O resultado e uma
    estimativa, nao uma medida clinica.
    """

    FREQ_MIN = 0.7   # Hz  (~42 bpm)
    FREQ_MAX = 4.0   # Hz  (~240 bpm)

    def __init__(self, tamanho_janela=300):
        self.tamanho_janela = tamanho_janela
        self.sinal_verde = deque(maxlen=tamanho_janela)
        self.tempos = deque(maxlen=tamanho_janela)
        self.bpm = None

    def _roi_testa(self, lm, frame, w, h):
        """Recorta um retangulo pequeno na testa (pele estavel, pouca sombra)."""
        cx = int(lm[10].x * w)
        cy = int(lm[10].y * h)

        topo = np.array([lm[10].x * w, lm[10].y * h])
        base = np.array([lm[152].x * w, lm[152].y * h])
        altura_rosto = np.linalg.norm(base - topo)

        meia_largura = int(altura_rosto * 0.15)
        meia_altura = int(altura_rosto * 0.08)
        deslocamento = int(altura_rosto * 0.10)

        y1 = max(cy + deslocamento - meia_altura, 0)
        y2 = min(cy + deslocamento + meia_altura, h)
        x1 = max(cx - meia_largura, 0)
        x2 = min(cx + meia_largura, w)

        if y2 <= y1 or x2 <= x1:
            return None, None
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _estimar_bpm(self):
        """FFT do sinal verde para achar a frequencia dominante na banda valida."""
        sinal = np.array(self.sinal_verde, dtype=float)
        sinal = sinal - np.mean(sinal)  # Remove componente DC

        duracao = self.tempos[-1] - self.tempos[0]
        if duracao <= 0:
            return None
        fps_real = len(self.tempos) / duracao

        fft = np.abs(np.fft.rfft(sinal))
        freqs = np.fft.rfftfreq(len(sinal), d=1.0 / fps_real)

        mascara = (freqs >= self.FREQ_MIN) & (freqs <= self.FREQ_MAX)
        if not np.any(mascara):
            return None

        fft_banda = fft[mascara]
        freqs_banda = freqs[mascara]
        freq_dominante = freqs_banda[np.argmax(fft_banda)]
        return freq_dominante * 60.0  # Hz -> bpm

    def update(self, lm, frame, w, h):
        roi, caixa = self._roi_testa(lm, frame, w, h)
        if roi is not None and roi.size > 0:
            media_verde = float(np.mean(roi[:, :, 1]))  # Canal G no BGR
            self.sinal_verde.append(media_verde)
            self.tempos.append(time.time())

        pronto = len(self.sinal_verde) == self.tamanho_janela
        if pronto:
            self.bpm = self._estimar_bpm()

        return {
            "bpm": self.bpm,
            "pronto": pronto,
            "progresso": len(self.sinal_verde),
            "total": self.tamanho_janela,
            "caixa": caixa,
        }