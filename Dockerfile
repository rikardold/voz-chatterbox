# Worker de voz (Chatterbox Multilingual) — imagem base com torch e CUDA já instalados.
#
# ## A versão da imagem NÃO é livre (duas lições medidas)
#
# **1ª — as bibliotecas têm de nascer alinhadas.** O `chatterbox-tts` fixa o torch, e o primeiro build
# falhou porque a base trazia torch 2.5.1 com o torchvision casado em 2.5.1: o pip trocava o torch,
# deixava o torchvision quebrado, e o `transformers` estourava ao importar o Llama — com uma mensagem
# que apontava para o lugar errado:
#     torchvision 0.20.1+cu124 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible
#     ModuleNotFoundError: Could not import module 'LlamaModel'
#
# **2ª — a GPU do Runpod mudou, e o torch 2.6 não a conhece (19/10/2026).** O worker passou a morrer
# no boot com:
#     NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb with CUDA capability sm_120
#     is not compatible with the current PyTorch installation.
#     The current PyTorch install supports: sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
#     RuntimeError: CUDA initialization failed: no kernel image is available for execution
#     Worker is unhealthy, exiting.
#
# O sintoma que chegava até nós era o pior possível: `/health` dizia "3 workers prontos", a fila não
# drenava, e o contador de `completed` ficava congelado — nada executava, nem um job inválido.
#
# `sm_120` é a arquitetura Blackwell (RTX 50xx / RTX PRO 6000), e exige **torch 2.7+ com CUDA 12.8+**.
# O torch 2.6 compila até sm_90. Subir a base resolve, e mantém sm_50..sm_90 — as GPUs antigas
# (A5000, L4, 3090) continuam funcionando, então a troca não fecha porta nenhuma.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/modelos \
    DISPOSITIVO=cuda \
    IDIOMA_PADRAO=pt \
    T3_MODEL=""

# O ffmpeg é para a AMOSTRA DE VOZ. Ela pode chegar em webm/mp4 (gravação de navegador) e o modelo só
# lê wav; sem ele, clonar a voz do apresentador falharia com a amostra real.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
# `--no-deps` no chatterbox NÃO serve: ele precisa das próprias dependências (transformers, etc.). O que
# protege o torch é a constraint abaixo, e por um motivo concreto:
#
# o `chatterbox-tts` declara `torch==2.6.0` no pyproject. Com a base em 2.8.0, o pip veria o conflito e
# REBAIXARIA o torch para 2.6.0 — voltando exatamente ao `sm_120` incompatível que derrubou o worker.
# A constraint fixa o torch da BASE como intocável: se o resolver não conseguir honrar as duas coisas,
# o build fica VERMELHO no CI (onde é barato) em vez de o worker morrer na GPU (onde é caro).
COPY restricoes.txt ./
RUN pip install --no-cache-dir -c restricoes.txt -r requirements.txt

# Conferência barata da pilha, ANTES do download pesado dos pesos: uma versão trocada pelo pip não
# pode custar 20 minutos de build para aparecer. As versões impressas aqui são também a evidência de
# qual combinação foi construída — sem elas o log não conta o que foi testado.
RUN python -c "import torch, torchaudio, torchvision; from transformers import LlamaModel; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| torchvision', torchvision.__version__, '| llama ok')"

# Qual versão da biblioteca veio, e QUAIS parâmetros ela aceita. Sem isto, um argumento a mais
# derruba o build com uma mensagem que não diz de que versão se trata — foi a segunda falha:
# `t3_model` existe no repositório oficial, mas não na versão publicada no PyPI que o pip instala.
RUN python -c "import inspect, importlib.metadata as md; from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; print('chatterbox-tts', md.version('chatterbox-tts')); print('from_pretrained', inspect.signature(M.from_pretrained))"

COPY handler.py teste.py ./

# Os pesos ficam DENTRO da imagem de propósito: baixá-los no primeiro trabalho faria cada worker novo
# gastar minutos de GPU (e de dinheiro) antes de falar a primeira palavra. O carregamento passa pelo
# MESMO caminho de código do worker (`handler.modelo()`) e não por uma chamada paralela aqui: assim o
# build não pode testar uma coisa e o endpoint rodar outra.
RUN DISPOSITIVO=cpu python -c "import handler; handler.modelo(); print('pesos prontos')"

CMD ["python", "-u", "handler.py"]
