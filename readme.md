# Detecção Facial e Gestos com MediaPipe

Projeto de estudos em visão computacional que utiliza a webcam para detectar **expressões faciais** e **gestos de mão** em tempo real, exibindo uma imagem correspondente ao estado identificado.

> **Status do projeto:** este é um protótipo experimental, criado de forma descontraída como exercício de aprendizado em Machine Learning e Visão Computacional. A lógica de classificação é baseada em geometria simples (razões entre distâncias de landmarks) e não em um modelo treinado — funciona bem como prova de conceito, mas tem limitações conhecidas (ver seção [Limitações](#limitações-atuais)).
>
> O projeto está em evolução para se tornar um **sistema de detecção e reconhecimento facial mais robusto**, com persistência em banco de dados e identificação de pessoas. Veja o [roadmap](#roadmap-evolução-planejada) abaixo.

## Sumário

- [Como funciona](#como-funciona)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Uso](#uso)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Limitações atuais](#limitações-atuais)
- [Roadmap (evolução planejada)](#roadmap-evolução-planejada)
- [Aviso sobre privacidade](#aviso-sobre-privacidade)

## Como funciona

O programa usa a [Tasks API do MediaPipe](https://ai.google.dev/edge/mediapipe) para extrair landmarks (pontos de referência) do rosto e da mão a partir do vídeo da webcam, processando um frame por vez:

1. **Detecção de mão** (`HandLandmarker`): identifica 21 pontos por mão e classifica gestos (`punho`, `mão aberta`, `joia`, `paz`) com base em quais dedos estão esticados.
2. **Detecção facial** (`FaceLandmarker`): identifica 468 pontos do rosto e classifica a expressão (`sorriso`, `boca aberta`, `sobrancelha levantada`, `neutro`) com base em razões geométricas entre pontos-chave (ex: largura da boca em relação à largura do rosto).
3. **Debounce**: o estado só muda na tela depois de se repetir por alguns frames consecutivos, evitando oscilação por ruído de detecção.
4. **Exibição**: o estado atual é mostrado como texto sobre o vídeo da webcam, e uma imagem correspondente é exibida em uma janela separada.

A lógica de classificação é baseada em **thresholds (limiares) calibrados manualmente**, não em um modelo de machine learning treinado para esse fim — o MediaPipe fornece os landmarks, mas a interpretação deles ("isso é um sorriso") é feita por regras geométricas simples escritas no código.

## Requisitos

- Python 3.9+
- Webcam
- Dependências (ver `requirements.txt` ou instalar manualmente):
  - `opencv-python`
  - `mediapipe`
  - `numpy`

## Instalação

```bash
git clone <url-do-repositorio>
cd <nome-do-projeto>
pip install opencv-python mediapipe numpy
```

Baixe os modelos do MediaPipe e coloque na pasta `models/`:
- `face_landmarker.task`
- `hand_landmarker.task`

Disponíveis em: https://ai.google.dev/edge/mediapipe/solutions/vision

Adicione as imagens correspondentes a cada estado na pasta `imgs/` (ver dicionário `IMAGES` no código para os nomes de arquivo esperados).

## Uso

```bash
python main.py
```

- Pressione `q` para encerrar o programa.
- Duas janelas serão abertas: `Webcam` (vídeo com overlay do estado detectado) e `Resultado` (imagem correspondente ao estado atual).

## Estrutura do projeto

```
.
├── main.py              # script principal
├── models/               # modelos .task do MediaPipe (não versionados)
│   ├── face_landmarker.task
│   └── hand_landmarker.task
├── imgs/                 # imagens exibidas para cada estado
└── README.md
```

## Limitações atuais

Documentar isso é importante tanto para transparência quanto para orientar a evolução do projeto:

- **Apenas uma pessoa e uma mão por vez** (`num_faces=1`, `num_hands=1`).
- **Thresholds calibrados manualmente**, sensíveis a iluminação, ângulo de câmera e características individuais do rosto — não generalizam bem entre pessoas diferentes sem reajuste.
- **Sem reconhecimento de identidade**: o sistema detecta *que há* um rosto e *qual expressão* ele tem, mas não sabe *de quem* é o rosto.
- **Sem persistência de dados**: nada é salvo entre execuções; cada sessão começa do zero.
- **Prioridade fixa entre expressões**: quando múltiplas condições são satisfeitas simultaneamente (ex: sorriso + sobrancelha levantada), a ordem dos `if` no código decide qual prevalece, o que pode não refletir a expressão dominante percebida por um humano.
- **Sem tratamento de erros** para modelos ausentes ou falha de inicialização da webcam.

## Roadmap (evolução planejada)

Objetivo: evoluir de protótipo de estudos para um **sistema funcional de detecção e reconhecimento facial**, com as seguintes frentes:

### 1. Reconhecimento de pessoas (identificação, não só detecção)
- [ ] Gerar *embeddings* faciais (vetor numérico que representa unicamente um rosto) usando um modelo apropriado para reconhecimento (ex: FaceNet, ArcFace, ou a própria solução de embeddings do MediaPipe).
- [ ] Fluxo de cadastro: capturar e associar um rosto a um nome/identificador.
- [ ] Fluxo de identificação: comparar o embedding capturado em tempo real com os cadastrados (por similaridade de vetores) e reconhecer a pessoa.

### 2. Persistência em banco de dados
- [ ] Definir modelo de dados (pessoas, embeddings, histórico de detecções/expressões, timestamps).
- [ ] Escolher banco adequado (relacional como PostgreSQL para dados estruturados, ou vetorial como FAISS/Pinecone/Chroma para busca eficiente de embeddings por similaridade).
- [ ] Camada de acesso a dados (ORM ou queries diretas) desacoplada da lógica de detecção.

### 3. Robustez da detecção de expressões
- [ ] Substituir thresholds fixos por calibração automática por usuário (ex: modo de calibração que registra faixa de valores neutros antes de iniciar).
- [ ] Avaliar substituir regras geométricas por um classificador treinado (ex: um modelo simples treinado sobre os próprios ratios como features).
- [ ] Tratar múltiplas pessoas e múltiplas mãos simultaneamente.

### 4. Engenharia e qualidade
- [ ] Separar responsabilidades em módulos (captura, detecção, classificação, persistência, interface) em vez de um único script.
- [ ] Testes automatizados para as funções de classificação geométrica.
- [ ] Tratamento de erros (modelos ausentes, webcam indisponível, falha de leitura de imagem).
- [ ] Arquivo `requirements.txt` com versões fixadas.
- [ ] Logging estruturado em vez de `print`.

### 5. Interface e usabilidade
- [ ] Interface gráfica (ex: Streamlit, PyQt) em vez de apenas janelas do OpenCV.
- [ ] Dashboard de histórico (quem foi detectado, quando, com qual expressão).

## Aviso sobre privacidade

Este projeto processa imagens de rosto capturadas pela webcam. Na versão atual (apenas detecção, sem persistência), nada é salvo ou enviado — todo o processamento é local e descartado ao fechar o programa.

Ao evoluir para reconhecimento facial com banco de dados, dados biométricos passarão a ser armazenados, o que exige atenção a:
- Consentimento explícito de quem é cadastrado.
- Armazenamento seguro dos embeddings (que são dados sensíveis, mesmo não sendo a imagem em si).
- Conformidade com legislação aplicável (no Brasil, a **LGPD** classifica dados biométricos como dado pessoal sensível).

Essas questões devem ser endereçadas antes de qualquer uso do sistema fora de um ambiente de estudo/teste controlado.
