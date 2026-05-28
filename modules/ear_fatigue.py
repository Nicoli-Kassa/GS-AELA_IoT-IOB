import numpy as np
 
class EARFatigueDetector:
    """Detecta fadiga ocular via Eye Aspect Ratio (EAR)."""

    IDX_OLHO_DIR = [33, 160, 158, 133, 153, 144]
    IDX_OLHO_ESQ = [362, 385, 387, 263, 373, 380]

    def __init__(self, thresh=0.22, consec_frames=20):
        self.thresh = thresh
        self.consec_frames = consec_frames
        self.contador_olho = 0
        self.episodios_fadiga = 0  # Conta episodios completos de fadiga
        self._em_fadiga = False    # Estado interno para detectar a borda

    def _pega_pontos(self, lm, indices, w, h):
        return [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]

    def _ear(self, pontos):
        A = np.linalg.norm(pontos[1] - pontos[5])
        B = np.linalg.norm(pontos[2] - pontos[4])
        C = np.linalg.norm(pontos[0] - pontos[3])
        return (A + B) / (2.0 * C)

    def update(self, lm, w, h):
        olho_d = self._pega_pontos(lm, self.IDX_OLHO_DIR, w, h)
        olho_e = self._pega_pontos(lm, self.IDX_OLHO_ESQ, w, h)
        ear = (self._ear(olho_d) + self._ear(olho_e)) / 2.0

        fadiga = False
        if ear < self.thresh:
            self.contador_olho += 1
            if self.contador_olho >= self.consec_frames:
                fadiga = True
        else:
            self.contador_olho = 0

        # Conta um episodio apenas na TRANSICAO de normal -> fadiga
        if fadiga and not self._em_fadiga:
            self.episodios_fadiga += 1
        self._em_fadiga = fadiga

        return {
            "ear": ear,
            "fadiga": fadiga,
            "episodios": self.episodios_fadiga,
        }