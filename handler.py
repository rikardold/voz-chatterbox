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
3. **O texto é FATIADO em frases e emendado (ver `dividir_em_frases`).** O modelo tem teto de
   tokens de saída e PARA em ~40 s, devolvendo o que coube SEM erro. Texto maior que isso era
   entregue pela metade — medido em 16/10: roteiro de 762 caracteres (≈57 s de fala) virou
   **40,00 s exatos**, com um trecho REPETIDO no fim (o modelo recicla o final ao bater no teto).
   Quem ouve percebe fala atropelada e frase cortada no meio.

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
import re
import subprocess
import tempfile
import urllib.request

import torch  # noqa: F401  (importado para o torchaudio achar o backend certo)
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

DISPOSITIVO = os.environ.get("DISPOSITIVO", "cuda")
IDIOMA_PADRAO = os.environ.get("IDIOMA_PADRAO", "pt")

# Vazio = usa o padrão da biblioteca instalada. Não é preguiça: o `from_pretrained` da versão que o
# pip instala (PyPI) **não aceita** `t3_model` — o parâmetro só existe no código do repositório
# oficial, que é mais novo que o publicado. Passá-lo fixo derrubou o build:
#     TypeError: ChatterboxMultilingualTTS.from_pretrained() got an unexpected keyword argument 't3_model'
# Opcional, a mesma imagem funciona nas duas versões — e subir para o V3 vira variável de ambiente,
# não mudança de código.
T3_MODEL = os.environ.get("T3_MODEL", "").strip()

# Teto de caracteres POR CHAMADA ao modelo.
#
# Não é gosto: é medição. O modelo para em ~40 s de áudio e devolve o que coube, sem erro — então o
# fatiamento precisa caber COM FOLGA abaixo disso. 300 caracteres rendem ~21 s de fala (a 7,3 s por
# 100 caracteres, medido em 19 produções reais do produto), o que deixa quase o dobro de margem.
# Fatiar perto do teto arriscaria cortar de novo — e o defeito voltaria só para textos longos, que é
# exatamente o caso que ninguém testa.
LIMITE_POR_TRECHO = int(os.environ.get("LIMITE_POR_TRECHO", "300"))

# Piso: uma frase sozinha maior que isso não dá para dividir em pontuação nenhuma e vai inteira,
# avisando na saída. Vale mais um trecho que passa do teto do que uma recusa que não entrega nada.
MINIMO_POR_TRECHO = int(os.environ.get("MINIMO_POR_TRECHO", "120"))

# A pausa entre trechos emendados, em segundos. Ver `emendar()` para a origem do número.
PAUSA_ENTRE_TRECHOS_S = float(os.environ.get("PAUSA_ENTRE_TRECHOS_S", "0.28"))

_modelo = None


def dividir_em_frases(texto, limite=LIMITE_POR_TRECHO, minimo=MINIMO_POR_TRECHO):
    """
    Quebra o texto em trechos que cabem numa chamada ao modelo — SEMPRE em fim de frase.

    ## Por que o corte não é por caractere

    Cortar em "a cobrança que preserva o seu ne-|-gócio" gera duas falas que não existem: a primeira
    termina numa sílaba solta e a segunda começa no meio de uma palavra. O modelo não sabe que as
    duas se juntam, então cada uma sai com entonação de frase final — e a emenda fica audível como
    um degrau na voz.

    Cortando em `.` `!` `?` `…`, cada trecho é uma fala COMPLETA: o modelo dá a entonação certa em
    cada um, e a emenda cai numa pausa natural, que é onde o ouvido não estranha.

    ## A ordem das tentativas

    1. fim de frase que caiba no limite;
    2. se não houver (frase única e gigante), vírgula/dois-pontos como respiro;
    3. se ainda não houver, espaço — o último recurso antes de cortar palavra;
    4. e só então o corte seco, com aviso.

    Devolve a lista de trechos. Números não entram no "fim de frase": "11.000" não é um ponto final.
    """
    limpo = " ".join((texto or "").split())

    if not limpo:
        return []

    if len(limpo) <= limite:
        return [limpo]

    # O lookbehind exige pontuação COM espaço depois (ou fim), para não confundir "11.000" com fim
    # de frase. O grupo fica com o terminador, para a pontuação não se perder na junção.
    fim_de_frase = re.compile(r"(?<=[.!?…])\s+")

    trechos = []
    restante = limpo

    while len(restante) > limite:
        # A janela de busca: até o limite, e no mínimo o `minimo` para não gerar trechos ridículos
        # de 3 caracteres quando o limite é pequeno.
        janela = max(restante.rfind(" ", 0, limite), minimo)
        corte = -1

        for padrao in (fim_de_frase, re.compile(r"(?<=[,;:])\s+"), re.compile(r"\s+")):
            for achado in padrao.finditer(restante, 0, janela):
                corte = achado.end()

            if corte > 0:
                break

        if corte <= 0:
            # Não há respiro nenhum antes do limite: corta onde dá e avisa. Melhor dividir no meio de
            # uma palavra do que devolver 40 s de um texto que precisava de 57.
            corte = janela

        trechos.append(restante[:corte].strip())
        restante = restante[corte:].strip()

    if restante:
        trechos.append(restante)

    # Trecho curto demais no FIM (ex.: "e é isso.") soa como fala solta. Junta no anterior quando
    # couber — e se não couber, fica: um trecho pequeno é melhor que um estouro de limite.
    if len(trechos) > 1 and len(trechos[-1]) < minimo // 2:
        candidato = trechos[-2] + " " + trechos[-1]

        if len(candidato) <= limite * 1.25:
            trechos = trechos[:-2] + [candidato]

    return [t for t in trechos if t]


def emendar(pedacos, taxa):
    """
    Junta os áudios dos trechos num só, com uma pausa entre eles.

    ## Por que a pausa, e por que ela é medida em SEGUNDOS

    Sem respiro, a última sílaba de um trecho encosta na primeira do seguinte e o resultado soa como
    uma palavra inventada. A pausa marca o fim de frase que o corte criou.

    `0,28 s` é a pausa que o próprio modelo já produz entre frases dentro de um trecho (medido: 7
    pausas num áudio de 40 s). Repetir a MESMA pausa mantém o ritmo — uma pausa de meio segundo
    soaria como o apresentador pensando, que não é o caso.

    Os pedaços vêm do mesmo modelo, mesma amostra e mesma taxa, então concatenar é somar amostras:
    não há remixagem, não há reamostragem, e a voz não muda no ponto da emenda.

    ## O bug que derrubou o worker em produção (19/10/2026)

    A primeira versão criava a pausa com `torch.zeros(...)`, que aloca **na CPU**. Os pedaços vêm do
    modelo, que roda em `cuda` quando `DISPOSITIVO=cuda` — e `torch.cat` de CPU com GPU levanta:

        RuntimeError: Expected all tensors to be on the same device

    O sintoma, medido no endpoint: os workers subiam, apareciam como `running=3` por ~100 s,
    voltavam a `ready` **sem executar**, a fila não drenava e o contador de `completed` ficava
    congelado. O Runpod reporta o container como pronto mesmo quando o handler morre no boot.

    ## Por que o teste de fumaça NÃO pegou isto

    Porque ele roda com `DISPOSITIVO=cpu` (ver `teste.py`): em CPU, a pausa e os pedaços estão no
    mesmo lugar e o `cat` funciona. **O teste que existe para pegar falha de build estava cego para
    falha de dispositivo.** A correção está no `teste.py`: ele agora emenda dois tensores numa
    "GPU" simulada e falha se o dispositivo da pausa não acompanhar o dos pedaços.

    ## A regra

    A pausa nasce no dispositivo do PRIMEIRO pedaco — nunca no padrão do `torch`. É o pedaço que
    sabe onde o modelo está rodando.
    """
    if len(pedacos) == 1:
        return pedacos[0]

    primeiro = pedacos[0].reshape(-1)
    silencio = torch.zeros(
        int(taxa * PAUSA_ENTRE_TRECHOS_S),
        dtype=primeiro.dtype,
        device=primeiro.device,
    )

    partes = []

    for indice, pedaco in enumerate(pedacos):
        if indice:
            partes.append(silencio)

        # `to(primeiro.device)` é rede de segurança: um modelo que devolva trechos em dispositivos
        # diferentes (não deveria, mas) não pode derrubar a emenda inteira.
        partes.append(pedaco.reshape(-1).to(primeiro.device))

    return torch.cat(partes)


def modelo():
    """
    Carrega o modelo UMA vez por worker.

    `from_pretrained` é caro (0,5B de parâmetros) e o worker vive várias requisições: carregar por
    trabalho jogaria fora justamente o ganho de ter um endpoint próprio.
    """
    global _modelo

    if _modelo is None:
        # O `t3_model` só vai quando alguém pede: assim a chamada padrão usa exatamente a assinatura
        # que a biblioteca instalada oferece.
        extras = {"t3_model": T3_MODEL} if T3_MODEL else {}

        _modelo = ChatterboxMultilingualTTS.from_pretrained(device=DISPOSITIVO, **extras)

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

    # O limite pode vir por trabalho (o produto pode querer trechos menores para textos com muitas
    # frases curtas), mas o padrão é o medido — ver `LIMITE_POR_TRECHO`.
    limite = int(entrada.get("limite_por_trecho") or LIMITE_POR_TRECHO)

    # Os ajustes finos só entram quando alguém pede: assim a chamada padrão usa apenas o que a ficha
    # do modelo documenta, e um nome de parâmetro errado aparece no teste do build, não na produção.
    ajustes = {
        chave: float(entrada[chave])
        for chave in ("exaggeration", "cfg_weight")
        if entrada.get(chave) is not None
    }

    # A amostra é preparada UMA vez: cada trecho abre o mesmo wav, e baixar/converter por trecho
    # multiplicaria o custo de ffmpeg por número de trechos sem mudar nada na voz.
    referencia = preparar_referencia(entrada.get("voice_url"))

    trechos = dividir_em_frases(texto, limite=limite)

    if len(trechos) == 1:
        # O caminho de sempre, intacto: texto curto não deve passar por emenda nenhuma. É o que
        # garante que esta correção não muda NADA do que já funcionava.
        audio = voz.generate(
            trechos[0],
            language_id=idioma,
            audio_prompt_path=referencia,
            **ajustes,
        )
    else:
        audio = emendar(
            [
                voz.generate(
                    trecho,
                    language_id=idioma,
                    audio_prompt_path=referencia,
                    **ajustes,
                )
                for trecho in trechos
            ],
            voz.sr,
        )

    buffer = io.BytesIO()
    ta.save(buffer, audio, voz.sr, format="wav")
    bytes_do_audio = buffer.getvalue()

    return {
        "audio_base64": base64.b64encode(bytes_do_audio).decode("ascii"),
        "sample_rate": voz.sr,
        "segundos": round(audio.shape[-1] / voz.sr, 2),
        "bytes": len(bytes_do_audio),
        "language_id": idioma,
        "clonou": bool(entrada.get("voice_url")),
        # Diagnóstico: quantos trechos foram emendados e o tamanho de cada um. Sem isto, um texto
        # fatiado e um texto curto devolveriam a MESMA saída, e não haveria como saber do log se a
        # emenda entrou em ação — nem se o limite ficou perto demais do teto do modelo.
        "trechos": len(trechos),
        "caracteres_por_trecho": [len(t) for t in trechos],
    }


def atender(trabalho):
    """A cola com o RunPod: só repassa o `input` para a parte pura."""
    return falar(trabalho.get("input") or {})


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": atender})
