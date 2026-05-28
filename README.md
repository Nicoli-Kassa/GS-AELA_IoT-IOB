# AELA Vision — Detector de Prontidão Biométrica por Câmera

> Módulo de **IoT & IOB** do projeto **AELA — Adaptive Extremes Life Analytics**  
> FIAP · Global Solution 2026 · _Space Connect_

Sistema de visão computacional em Python que estima, em tempo real e **sem nenhum sensor de contato**, três sinais de prontidão fisiológica a partir do rosto capturado por uma câmera comum: fadiga ocular, frequência cardíaca e fadiga física postural.

### Integrantes

| Nome                          | RM        |
| ----------------------------- | --------- |
| Camila Pedroza da Cunha       | RM 558768 |
| Nicolli Amy Kassa             | RM 559104 |
| Isabelle Dallabeneta Carlesso | RM 554592 |

## 1. O que é o AELA

O **AELA (Adaptive Extremes Life Analytics)** é uma plataforma de inteligência biológica para operadores em ambientes extremos. A ideia central cabe numa frase:

> Todo sistema de saúde compara você com outras pessoas. O AELA compara você com **você mesmo** — antes, durante e depois do ambiente extremo.

Sistemas de monitoramento de saúde tradicionais fazem uma pergunta: _o operador está dentro dos parâmetros normais da população?_ O AELA faz uma pergunta diferente e mais útil: _este operador específico está se afastando do seu próprio estado ideal — e quão rápido?_

Antes de uma missão, cada operador cria um **baseline individual** — o seu "zero" pessoal: frequência cardíaca de repouso, padrão de sono, tempo de reação, acuidade visual, e outras métricas. Durante a operação, o sistema monitora o **desvio** em relação a esse baseline em tempo real. O resultado não é um diagnóstico genérico, e sim uma **decisão operacional**: quem está mais apto para a tarefa crítica de amanhã? quem precisa descansar agora?

### O diferencial

O que torna o AELA difícil de copiar não é a tecnologia, e sim o conceito de **baseline individual + desvio contextual por tarefa**. Um alarme tradicional diz "pressão ocular elevada". O AELA diz "o operador B está mais apto que o A para a tarefa de precisão de amanhã — use o B". Essa tradução de dado fisiológico em decisão é o coração do produto.

E embora tenha sido projetado para o espaço — o ambiente mais extremo que existe — a biologia humana sob estresse não muda de regra conforme a altitude. O mesmo sistema que protege um astronauta em órbita protege um bombeiro num incêndio florestal, um alpinista no Everest ou uma equipe de resgate nas primeiras 72 horas após um terremoto. Em todos esses casos o problema é idêntico: **um corpo operando no limite, sem um sistema que conheça o seu normal.**

## 2. Onde o IoT entra

O AELA coleta dados fisiológicos de várias fontes — principalmente wearables e roupas inteligentes. Mas há um problema prático em campo:

> **Wearables nem sempre estão disponíveis.** A bateria acaba, o sensor descola com o suor, o traje não comporta o dispositivo, ou simplesmente não há tempo de equipar o operador numa emergência.

Uma câmera, por outro lado, quase sempre está presente — no capacete, no painel da nave, no equipamento de campo. Este módulo de IoT responde à pergunta:

> _Quanto da prontidão de um operador dá para inferir usando apenas uma câmera, sem nenhum sensor de contato?_

Ele é a **camada de coleta biométrica por visão computacional** do AELA: captura sinais de fadiga e esforço diretamente do rosto e alimenta os módulos de saúde da plataforma quando o wearable falha. Na arquitetura do AELA, este componente entrega dados que seriam consumidos pelo motor de cálculo de prontidão (o _ReadinessScore_) — mas funciona de forma autônoma, fechando seu próprio ciclo de captura → inferência → recomendação.

### Posição na arquitetura do AELA

```
  [ Wearables / roupas inteligentes ]  --+
                                         +-->  Baseline individual + desvio  -->  ReadinessScore  -->  Decisao de missao
  [ ESTE MODULO: camera + visao CV ]  ---+        (camada de dados do AELA)
        (backup quando nao ha wearable)
```

## 3. A solução técnica

Um único pipeline de visão computacional roda o **MediaPipe Face Landmarker** uma vez por frame e distribui os 478 pontos faciais para três detectores independentes:

| Módulo        | Mede                | Técnica                                               | Módulo AELA alimentado          |
| ------------- | ------------------- | ----------------------------------------------------- | ------------------------------- |
| **EAR**       | Fadiga ocular       | Eye Aspect Ratio — proporção de abertura dos olhos    | Cognitivo                       |
| **rPPG**      | Frequência cardíaca | Remote photoplethysmography — variação de cor da pele | Cardiovascular                  |
| **Head Pose** | Fadiga física       | Inclinação da cabeça (_head nodding_)                 | Ósseo-muscular / comportamental |

Quando os indicadores de fadiga acumulam acima de um limite, o sistema emite uma **recomendação acionável de descanso**, indicando qual sinal a disparou — coerente com a filosofia do AELA: não "você está cansado", mas "qual sinal, quão acumulado, e o que fazer".

## 4. Como cada módulo funciona

### 4.1 EAR — fadiga ocular _(sólido)_

O _Eye Aspect Ratio_ compara a abertura vertical do olho com sua largura horizontal. Com o olho aberto a razão é alta; ao fechar, despenca. Quando o EAR fica abaixo de um limiar por um número mínimo de frames consecutivos, registra-se um episódio de fadiga.

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 · ||p1 - p4||)
```

É um cálculo puramente geométrico, estável e determinístico — a âncora do sistema.

### 4.2 Head Pose — fadiga física _(sólido)_

Detecta o _head nodding_ (a cabeça caindo de sono e voltando). Mede o deslocamento vertical do nariz em relação às têmporas, **normalizado pela largura do rosto** — o que torna a métrica imune à distância da câmera. Uma máquina de estados conta um cabeceio completo a cada ciclo de queda e retorno.

### 4.3 rPPG — frequência cardíaca _(experimental)_

Estima a frequência cardíaca pela variação sutil de cor da pele na testa causada pelo fluxo sanguíneo. A cada frame mede-se a média do canal verde numa região de interesse (ROI) da testa; após acumular uma janela de ~10 segundos, uma **FFT** identifica a frequência dominante na banda fisiológica plausível (0,7–4 Hz, ou seja 42–240 bpm).

> **Nota técnica:** rPPG capta variações de brilho da ordem de menos de 1%, exigindo luz estável e o rosto imóvel. Por isso o sistema rotula o resultado explicitamente como **estimativa óptica** e o trataria como sinal _complementar_, nunca diagnóstico. É um módulo experimental por natureza, e seu valor está em demonstrar inferência cardiovascular sem contato.

## 5. Estrutura do projeto

```
GS_IoT&IOB/
├── aela_vision.py            # Loop principal: captura, orquestração e HUD
├── face_landmarker.task      # Modelo MediaPipe (baixar — ver instalação)
├── requirements.txt
├── README.md
└── modules/
    ├── __init__.py
    ├── ear_fatigue.py        # Detector de fadiga ocular (EAR)
    ├── head_pose.py          # Detector de head nodding
    └── rppg_heartrate.py     # Estimador de frequência cardíaca (rPPG)
```

Cada detector é uma **classe com estado próprio** que expõe um método `update()`. O `aela_vision.py` apenas orquestra: roda o Face Landmarker, chama os três `update()` e desenha o painel. Essa separação de responsabilidades mantém cada técnica isolada e testável.

## 6. Bibliotecas utilizadas

| Biblioteca                   | Uso                                                  |
| ---------------------------- | ---------------------------------------------------- |
| **OpenCV** (`opencv-python`) | Captura de vídeo, desenho do HUD e exibição          |
| **MediaPipe**                | Detecção dos 478 landmarks faciais (Face Landmarker) |
| **NumPy**                    | Cálculos vetoriais, distâncias e FFT do sinal rPPG   |

## 7. Instalação e execução

### Pré-requisitos

- Python 3.9 ou superior
- Uma webcam

### Passo a passo

**1. Clone o repositório e entre na pasta:**

```bash
git clone [PREENCHER: URL DO REPOSITÓRIO]
cd GS_IoT&IOB
```

**2. (Recomendado) crie um ambiente virtual:**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. Baixe o modelo do MediaPipe** (coloque na mesma pasta do `aela_vision.py`):

```bash
curl -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**5. Execute:**

```bash
python aela_vision.py
```

### Controles

| Tecla | Ação                                                  |
| ----- | ----------------------------------------------------- |
| `q`   | Encerra o programa                                    |
| `r`   | Zera os contadores (útil para repetir a demonstração) |

## 8. Dica de uso para o rPPG

O módulo rPPG é sensível às condições de captura. Para a melhor estimativa:

- **Iluminação de frente** ao rosto (luz natural ou luminária) — é o fator mais importante.
- **Permaneça imóvel** durante os ~10 segundos de calibração (a barra `calib N/300` mostra o progresso).
- Evite luzes com cintilação no fundo.

Os módulos EAR e Head Pose funcionam de forma robusta em qualquer condição de luz razoável.

## 9. Conexão com o tema espacial (Space Connect) e ODS

Este módulo demonstra que sinais de prontidão de um operador em ambiente extremo podem ser inferidos sem hardware de contato — relevante para cenários onde wearables falham: uma EVA, um incêndio florestal, uma missão de resgate. Está conectado aos Objetivos de Desenvolvimento Sustentável:

- **ODS 3 — Saúde e bem-estar:** monitoramento de prontidão para proteger operadores de alto risco.
- **ODS 9 — Indústria, inovação e infraestrutura:** nova infraestrutura de inferência biológica baseada em tecnologia espacial aplicada.

## 9. Vídeo de demonstração

[PREENCHER: link do YouTube não listado]
