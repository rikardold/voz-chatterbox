# Worker de voz (Chatterbox Multilingual) — imagem base com torch e CUDA já instalados.
#
# Começar de `python:3.11-slim` obrigaria o pip a baixar vários GB de torch CUDA no build; a imagem
# oficial do PyTorch já traz torch, torchvision e torchaudio alinhados com a CUDA.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/modelos \
    DISPOSITIVO=cuda \
    IDIOMA_PADRAO=pt \
    T3_MODEL=v3

# O ffmpeg é para a AMOSTRA DE VOZ. Ela pode chegar em webm/mp4 (gravação de navegador) e o modelo só
# lê wav; sem ele, clonar a voz do apresentador falharia com a amostra real.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Os pesos ficam DENTRO da imagem de propósito: baixá-los no primeiro trabalho faria cada worker novo
# gastar minutos de GPU (e de dinheiro) antes de falar a primeira palavra. Se este comando estiver
# errado, o build falha aqui — que é o lugar certo para descobrir.
RUN python -c "from chatterbox.mtl_tts import ChatterboxMultilingualTTS as M; M.from_pretrained(device='cpu', t3_model='v3')"

COPY handler.py teste.py ./

CMD ["python", "-u", "handler.py"]
