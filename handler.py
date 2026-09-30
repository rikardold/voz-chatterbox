"""
Worker de voz para os Reels: texto + idioma (+ amostra de voz) -> áudio.

Por que este worker existe
--------------------------
O card do Hub (`aydinmyilmaz/chatterbox-multilingual-tts`, v1.0.4) morre na SEGUNDA linha do próprio
handler:

    from runpod.serverless.utils import upload_output
    ImportError: cannot import name 'upload_output' from 'runpod.serverless.utils'

O container sai com código 1, nunca chega a pedir trabalho na fila — e o sintoma que aparece para
quem chama é o pior possível: "3 workers prontos, 0 rodando" e o trabalho parado em IN_QUEUE para
sempre. Imagem de outra pessoa não dá para consertar. Este worker é nosso.

Duas decisões vieram de erro real, não de gosto:

1. **A saída é base64, e NÃO um upload pelos utilitários do SDK.** Foi exatamente um utilitário do
   SDK que quebrou o outro worker. Devolvendo os bytes na resposta, o worker não depende de nenhum
   helper que muda de versão para versão.
2. **`language_id` é explícito.** O modelo Multilingual usa o idioma para escolher o tokenizador;
   sem ele, o português sai com fonema de outro idioma — o defeito que já foi para produção com o
   `chatterbox-turbo` (que é English-only). A ficha do modelo ainda avisa: se a amostra de voz
   estiver em outro idioma, passe `cfg_weight: 0`, senão a saída herda o sotaque da amostra.

Entrada (`input`)
-----------------
    prompt         texto a falar (obrigatório)
    language_id    'pt' por padrão — o modelo cobre 23 idiomas:
                   ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh
    voice_url      amostra de voz (6-15 s) para clonar — opcional
    exaggeration   emoção (padrão do modelo: 0.5)
    cfg_weight     aderência à amostra (padrão do modelo: 0.5; use 0 quando a amostra for de outro idioma)

Saída
-----
    audio_base64, sample_rate, segundos, bytes, language_id, clonou
"""

import base64
import io
import os
import subprocess
import tempfile
import urllib.request

import torch  # noqa: F401  (importado para o torchaudio achar o backend certo)
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

DISPOSITIVO = os.environ.get("DISPOSITIVO", "cuda")
IDIOMA_PADRAO = os.environ.get("IDIOMA_PADRAO", "pt")
T3_MODEL = os.environ.get("T3_MODEL", "v3")

_modelo = None


def modelo():
    """
    Carrega o modelo UMA vez por worker.

    `from_pretrained` é caro (0,5B de parâmetros) e o worker vive várias requisições: carregar por
    trabalho jogaria fora justamente o ganho de ter um endpoint próprio.
    """
    global _modelo

    if _modelo is None:
        _modelo = ChatterboxMultilingualTTS.from_pretrained(device=DISPOSITIVO, t3_model=T3_MODEL)

    return _modelo


def preparar_referencia(url):
    """
    Baixa a amostra de voz e devolve um wav que o modelo consegue ler.

    O ffmpeg no meio não é enfeite: a amostra real vem do gravador do navegador (webm/mp4) e o
    Chatterbox lê wav. Sem a conversão, clonar a voz do apresentador falharia justamente com a
    amostra de verdade — e passaria no teste feito com um wav de laboratório.
    """
    if not url:
        return None

    pasta = tempfile.mkdtemp()
    origem = os.path.join(pasta, "amostra")
    destino = origem + ".wav"

    urllib.request.urlretrieve(url, origem)

    subprocess.run(
        ["ffmpeg", "-y", "-i", origem, "-ar", "24000", "-ac", "1", destino],
        check=True,
        capture_output=True,
    )

    return destino


def falar(entrada):
    """
    A parte PURA: recebe o `input` do trabalho e devolve o dicionário de saída.

    Estar separada da cola do SDK é de propósito — é o que permite testar a voz **no build**, sem
    GPU e sem fila. O card que a gente tentou morria no import e ninguém via, porque nada testava a
    imagem antes de ela entrar no ar.
    """
    texto = (entrada.get("prompt") or "").strip()

    if not texto:
        return {"erro": "informe `prompt` com o texto a falar."}

    idioma = entrada.get("language_id") or IDIOMA_PADRAO
    voz = modelo()

    # Os ajustes finos só entram quando alguém pede: assim a chamada padrão usa apenas o que a ficha
    # do modelo documenta, e um nome de parâmetro errado aparece no teste do build, não na produção.
    ajustes = {
        chave: float(entrada[chave])
        for chave in ("exaggeration", "cfg_weight")
        if entrada.get(chave) is not None
    }

    wav = voz.generate(
        texto,
        language_id=idioma,
        audio_prompt_path=preparar_referencia(entrada.get("voice_url")),
        **ajustes,
    )

    buffer = io.BytesIO()
    ta.save(buffer, wav, voz.sr, format="wav")
    bytes_do_audio = buffer.getvalue()

    return {
        "audio_base64": base64.b64encode(bytes_do_audio).decode("ascii"),
        "sample_rate": voz.sr,
        "segundos": round(wav.shape[-1] / voz.sr, 2),
        "bytes": len(bytes_do_audio),
        "language_id": idioma,
        "clonou": bool(entrada.get("voice_url")),
    }


def atender(trabalho):
    """A cola com o RunPod: só repassa o `input` para a parte pura."""
    return falar(trabalho.get("input") or {})


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": atender})
