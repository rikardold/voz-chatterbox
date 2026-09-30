# Worker de voz (Chatterbox Multilingual) — imagem base com torch e CUDA já instalados.
#
# A versão da imagem NÃO é livre. O `chatterbox-tts` fixa **torch 2.6.0**, e o primeiro build falhou
# porque a base trazia torch 2.5.1 com o torchvision casado em 2.5.1: o pip trocava o torch, deixava o
# torchvision quebrado, e o `transformers` estourava ao importar o Llama — com uma mensagem que
# apontava para o lugar errado:
#     torchvision 0.20.1+cu124 requires torch==2.5.1, but you have torch 2.6.0 which is incompatible
#     ModuleNotFoundError: Could not import module 'LlamaModel'
# Com a base já em 2.6.0, torch/torchvision/torchaudio nascem alinhados e o pip não troca nada.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/modelos \
    DISPOSITIVO=cuda \
    IDIOMA_PADRAO=pt \
    T3_MODEL=v3

# O ffmpeg é para a AMOSTRA DE VOZ. Ela pode chegar em webm/mp4 (gravação de navegador) e o modelo só
# lê wav; sem ele, clonar a voz do apresentador falharia com a amostra real.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Conferência barata da pilha, ANTES do download pesado dos pesos: uma versão trocada pelo pip não
# pode custar 20 minutos de build para aparecer. As versões impressas aqui são também a evidência de
# qual combinação foi construída — sem elas o log não conta o que foi testado.
RUN python -c "import torch, torchaudio, torchvision; from transformers import LlamaModel; print('torch', torch.__version__, '| torchaudio', torchaudio.__version__, '| torchvision', torchvision.__version__, '| llama ok')"

# Os pesos ficam DENTRO da imagem de propósito: baixá-los no primeiro trabalho faria cada worker novo
# gastar minutos de GPU (e de dinheiro) antes de falar a primeira palavra. Se este comando estiver
# errado, o build falha aqui — que é o lugar certo para descobrir.
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; M.from_pretrained(device='cpu', t3_model='v3')"

COPY handler.py teste.py ./

CMD ["python", "-u", "handler.py"]
