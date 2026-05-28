import numpy as np

class HeadPoseDetector:
    """Detecta head nodding (cabeceio de fadiga) via deslocamento vertical do nariz."""

    IDX_NARIZ = 1          # Ponta do nariz
    IDX_TEMPORA_DIR = 234  # Lateral direita do rosto
    IDX_TEMPORA_ESQ = 454  # Lateral esquerda do rosto

    def __init__(self, queda=0.15, normal=0.05):
        # metrica = (y_nariz - y_medio_temporas) / largura_rosto
        # Cabeca reta  -> metrica BAIXA
        # Queixo caindo -> metrica ALTA
        # Histerese: 'queda' e 'normal' afastados evitam contagem dupla.
        self.queda = queda
        self.normal = normal
        self.estado = "erguida"
        self.eventos_nod = 0

    def update(self, lm, w, h):
        nariz = np.array([lm[self.IDX_NARIZ].x * w, lm[self.IDX_NARIZ].y * h])
        temp_d = np.array([lm[self.IDX_TEMPORA_DIR].x * w, lm[self.IDX_TEMPORA_DIR].y * h])
        temp_e = np.array([lm[self.IDX_TEMPORA_ESQ].x * w, lm[self.IDX_TEMPORA_ESQ].y * h])

        largura = np.linalg.norm(temp_e - temp_d)
        y_medio_temporas = (temp_d[1] + temp_e[1]) / 2.0
        metrica = (nariz[1] - y_medio_temporas) / largura if largura > 0 else 0.0

        # Estado
        if metrica > self.queda:
            self.estado = "caida"

        if metrica < self.normal and self.estado == "caida":
            self.estado = "erguida"
            self.eventos_nod += 1

        return {
            "metrica": metrica,
            "nods": self.eventos_nod,
            "estado": self.estado,
        }