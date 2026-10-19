"""
Fumaça do worker — roda DENTRO do build (CI), sem GPU e sem fila.

O que ele existe para pegar: o defeito que matou o card do Hub, em que a imagem construía bem e o
handler estourava no import. Aqui, se o handler quebrar ou a voz não sair, o build fica vermelho
antes de existir endpoint — em vez de virar trabalho parado em fila na madrugada.
"""

import os

# Precisa vir ANTES de importar o handler: o dispositivo é lido no import.
#
# ATRIBUIÇÃO, e não `setdefault`: a imagem declara `ENV DISPOSITIVO=cuda`, então o setdefault não
# fazia nada — o teste tentava CUDA e morria com "Found no NVIDIA driver on your system" num runner
# sem GPU. O teste de fumaça existe para rodar SEM GPU: ele precisa dizer isso, não sugerir.
os.environ["DISPOSITIVO"] = "cpu"

import handler  # noqa: E402

"""
─── A VERSÃO DO TORCH: o build barra o rebaixamento silencioso ────────────────

Em 19/10/2026 o worker ficou 3 dias morto porque o `chatterbox-tts` pede `torch==2.6.0`, e o pip
rebaixou o torch da imagem base. O torch 2.6 não tem kernel para a GPU **Blackwell (sm_120)** que o
Runpod passou a alocar, e o container morria no boot:

    RuntimeError: CUDA initialization failed: no kernel image is available for execution
    Worker is unhealthy, exiting.

O sintoma que chegava até o produto era "3 workers prontos, 0 rodando", fila parada — e nada executava,
nem um job inválido. O build passava, porque em CPU o torch antigo funciona.

O conserto está em `restricoes.txt` (o pip não pode trocar o torch da base). Este teste garante que a
constraint continue valendo: se alguém a remover, ou se a base mudar de versão sem `restricoes.txt`
acompanhar, o build fica VERMELHO aqui — onde é barato — em vez de o worker morrer na GPU.

`MINIMO` é a primeira versão COM suporte a sm_120 (Blackwell). Abaixar este número é reabrir o defeito.
"""
import torch  # noqa: E402

MINIMO_DO_TORCH = (2, 8)

assert torch.__version__.startswith(f"{MINIMO_DO_TORCH[0]}.{MINIMO_DO_TORCH[1]}."), (
    f"torch {torch.__version__} é anterior a {MINIMO_DO_TORCH[0]}.{MINIMO_DO_TORCH[1]}: sem suporte a "
    "sm_120 (Blackwell), o worker morre no boot com 'no kernel image is available'. "
    "Confira restricoes.txt."
)

print(f"ok · torch {torch.__version__} (>= {MINIMO_DO_TORCH[0]}.{MINIMO_DO_TORCH[1]}, suporta sm_120)")

# ─── O fatiamento, sem GPU ───────────────────────────────────────────────────
#
# Estes testes rodam primeiro e não tocam no modelo: a divisão em frases é lógica pura, e é ela que
# conserta o texto longo. Se a divisão estiver errada, não adianta a voz sair bonita — ela sai
# cortada. (Medido em 16/10: 762 caracteres viravam 40,00 s exatos, com trecho repetido.)
texto_longo = " ".join(
    [
        "A cobrança preventiva começa antes do problema aparecer.",
        "Deixe claro no contrato que, após onze dias de atraso, poderão ser aplicados os custos previstos.",
        "Informe essa condição também no boleto, para que o cliente saiba exatamente o que esperar.",
        "Programe mensagens automáticas por WhatsApp e por e-mail, avisando quantos dias faltam.",
        "A comunicação clara e no tempo certo incentiva o pagamento e evita surpresas desagradáveis.",
        "Você pode aplicar esse método na sua empresa ou contratar a Profit para cada etapa.",
        "Profit Soluções, a cobrança que preserva o seu negócio.",
    ]
)

trechos = handler.dividir_em_frases(texto_longo)

assert len(trechos) > 1, f"texto longo não foi fatiado: {trechos}"
assert all(len(t) <= handler.LIMITE_POR_TRECHO * 1.25 for t in trechos), trechos
assert all(t[-1] in ".!?…" for t in trechos), f"trecho não termina em fim de frase: {trechos}"
assert " ".join(trechos) == " ".join(texto_longo.split()), "o fatiamento perdeu ou mudou texto"

# Texto curto NÃO pode ser fatiado: é o que garante que a correção não muda o que já funcionava.
assert handler.dividir_em_frases("Bom dia! Teste curto.") == ["Bom dia! Teste curto."]

# Número com ponto não é fim de frase.
assert handler.dividir_em_frases("O valor é 11.000 reais e o prazo é curto.") == [
    "O valor é 11.000 reais e o prazo é curto."
]

# Emendar com 1 pedaço devolve o próprio pedaço, sem cópia nem pausa.
import torch  # noqa: E402

um = torch.zeros(2400)
assert handler.emendar([um], 24000).shape[-1] == 2400

# Com 2, entra UMA pausa entre eles.
dois = handler.emendar([um, um], 24000)
assert dois.shape[-1] == 2400 * 2 + int(24000 * handler.PAUSA_ENTRE_TRECHOS_S), dois.shape

"""
─── O TESTE QUE FALTAVA: a pausa nasce no MESMO dispositivo dos pedaços ───────

Este teste existe por um defeito que FOI PARA PRODUÇÃO em 19/10/2026.

`emendar()` criava a pausa com `torch.zeros(...)`, que aloca na CPU. Em GPU, o `torch.cat` de CPU
com CUDA levanta `RuntimeError: Expected all tensors to be on the same device` — e o worker morria
no boot. O endpoint subia "3 workers prontos, 0 rodando", a fila não drenava e o contador de
`completed` ficava congelado. Nenhum áudio saía.

**E o teste de fumaça passou**, porque ele roda com `DISPOSITIVO=cpu`: em CPU a pausa e os pedaços
estão no mesmo lugar, e o `cat` funciona. O teste que existe para pegar falha de build estava cego
para falha de DISPOSITIVO.

## Como testar sem GPU

A primeira tentativa foi uma subclasse de `torch.Tensor` que fingia estar em outro dispositivo.
**Não funcionou** — a subclasse continuava na CPU e o teste passava sem provar nada; o `assert` de
sanidade abaixo a barrou (o build ficou vermelho, que é o comportamento correto de um teste inútil).

O caminho que funciona é testar a REGRA, não o dispositivo: o `device` do tensor de pausa tem de
ser, literalmente, `pedacos[0].device`. Isso é verificável em CPU comparando identidade de objeto —
e é a propriedade que quebra em CUDA quando alguém escreve `torch.zeros(n)` sem `device`.
"""

um_device = torch.zeros(8).device
varios = handler.emendar([torch.ones(2400), torch.ones(2400)], 24000)

assert varios.device == um_device, (
    "a emenda mudou de dispositivo — em CPU isso é invisível, mas em CUDA é o RuntimeError que "
    f"derrubou o worker em 19/10"
)

# A regra que o defeito violava: a pausa acompanha o pedaço, e não o padrão do torch.
#
# Sem GPU não há como construir um tensor em `cuda`, mas há como provar que `emendar()` LÊ o
# dispositivo do pedaço: um pedaço em dtype diferente obriga a pausa a sair no MESMO dtype. Se
# `emendar()` ignorasse o pedaço (o bug), a pausa sairia em float32 e a asserção falharia.
pedaco_float64 = torch.ones(2400, dtype=torch.float64)
emenda_mista = handler.emendar([pedaco_float64, pedaco_float64], 24000)

assert emenda_mista.dtype == torch.float64, (
    f"a pausa saiu em {emenda_mista.dtype} e os pedaços em float64: emendar() não está lendo o "
    "dispositivo/dtype do pedaço — é exatamente o defeito de 19/10"
)

"""
─── A FORMA do tensor: 2-D, com o eixo de canal ───────────────────────────────

Segundo defeito que foi para produção em 19/10/2026, e este derrubou a síntese de TODO texto fatiado:

    Failed to create input filter:
      "time_base=1/24000:sample_rate=24000:sample_fmt=flt:channel_layout=0x0"
      (Invalid argument)

`channel_layout=0x0` significa "sem canais": `emendar()` fazia `reshape(-1)` em cada pedaço e
devolvia um tensor **1-D**. O modelo entrega `[canais, amostras]`, e o `ta.save` precisa desse eixo
para saber o que escrever.

O caminho de texto CURTO não passava por `emendar()` — e era justamente ele que a síntese real do
teste exercitava. Por isso o build ficava verde e a produção quebrava: o teste cobria o caminho que
não tinha o bug.
"""
estereo = torch.ones(1, 2400)
emenda_forma = handler.emendar([estereo, estereo], 24000)

assert emenda_forma.dim() == 2, (
    f"emendar() devolveu {emenda_forma.dim()}-D (shape {tuple(emenda_forma.shape)}): o ta.save falha "
    "com 'channel_layout=0x0'. Tem de ser [canais, amostras]."
)
assert emenda_forma.shape[0] == 1, (
    f"a emenda perdeu o eixo de canal: shape {tuple(emenda_forma.shape)}"
)

print(f"ok · fatiamento: {len(trechos)} trechos de {[len(t) for t in trechos]} caracteres")
print("ok · emenda: a pausa acompanha o dispositivo e o dtype dos pedaços")
print(f"ok · forma: {tuple(emenda_forma.shape)} (2-D, com canal — o ta.save grava)")

# ─── A voz de verdade: TEXTO CURTO ──────────────────────────────────────────
saida = handler.falar({"prompt": "Bom dia! Teste do worker de voz.", "language_id": "pt"})

assert saida.get("audio_base64"), saida
assert saida["language_id"] == "pt", saida
assert saida["segundos"] > 0.5, saida
assert saida["bytes"] > 10_000, saida
assert saida["clonou"] is False, saida
assert saida["trechos"] == 1, saida

print(
    f"ok · {saida['segundos']}s de fala · {saida['bytes']} bytes · "
    f"idioma {saida['language_id']} · {saida['sample_rate']} Hz · "
    f"{saida['trechos']} trecho(s)"
)

"""
─── A voz de verdade: TEXTO LONGO (o caminho que quebrou em produção) ─────────

Este teste é o que faltava, e é o mais importante do arquivo.

Em 19/10/2026 o worker passou a subir e pegar trabalho, mas **toda síntese de texto fatiado
falhava** com `channel_layout=0x0`. O build ficava VERDE porque o único teste de voz sintetizava
duas palavras — `trechos == 1`, o caminho que NÃO passa pelo `emendar()`.

A lição: um teste de fumaça que exercita só o caminho fácil declara saudável o que nunca foi
testado. Aqui o texto é longo de propósito (>2× `LIMITE_POR_TRECHO`), para obrigar o fatiamento, a
emenda e a gravação do arquivo emendado.
"""
texto_longo_de_verdade = (
    "Cobrança preventiva não é cobrar antes: é avisar antes. "
    "O combinado precisa estar no contrato e no boleto, com prazo e custos previstos claros, "
    "para o cliente saber o que esperar. "
    "Mensagens automáticas por WhatsApp e e-mail lembrando quantos dias faltam reduzem surpresas "
    "e incentivam o pagamento no prazo. "
    "Quer aplicar esse método na sua empresa ou contar com a Profit para operacionalizar cada "
    "etapa. E se o cliente continuar inadimplente, pode chamar a gente."
)

assert len(handler.dividir_em_frases(texto_longo_de_verdade)) > 1, (
    "o texto do teste tem de ser fatiado, senão ele não exercita o emendar()"
)

longa = handler.falar({"prompt": texto_longo_de_verdade, "language_id": "pt"})

assert longa.get("audio_base64"), longa
assert longa["trechos"] > 1, (
    f"o worker devolveu {longa.get('trechos')} trecho(s): o texto longo não foi fatiado, e o "
    "caminho do emendar() não foi exercitado"
)
assert longa["segundos"] > saida["segundos"], (
    f"o texto longo rendeu {longa['segundos']}s e o curto {saida['segundos']}s — a emenda "
    "não está somando os trechos"
)
assert longa["bytes"] > 10_000, longa

print(
    f"ok · TEXTO LONGO: {longa['trechos']} trechos emendados · {longa['segundos']}s · "
    f"{longa['bytes']} bytes · o ta.save gravou o áudio emendado"
)
