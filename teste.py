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

Aqui não há GPU no runner, então o teste simula: um tensor "de mentira" cujo `.device` não é o
padrão do `torch`. Se `emendar()` não respeitar o dispositivo do pedaço, a asserção falha ANTES de a
imagem entrar no ar.
"""


class PedacoEmOutroDispositivo(torch.Tensor):
    """Um tensor como se viesse da GPU: `reshape` e `to` preservam um dispositivo diferente do padrão."""

    @staticmethod
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self

    def to(self, *args, **kwargs):
        return self


falso = PedacoEmOutroDispositivo(2400)
assert falso.device != torch.zeros(1).device, "o tensor tem de parecer de outro dispositivo"

# A pausa precisa sair no dispositivo do PEDAÇO, e não no padrão do torch.
try:
    resultado = handler.emendar([falso, falso], 24000)
    assert resultado.shape[-1] == 2400 * 2 + int(24000 * handler.PAUSA_ENTRE_TRECHOS_S), resultado.shape
except RuntimeError as erro:
    raise AssertionError(
        "emendar() misturou dispositivos — é o defeito que derrubou o worker em 19/10: " + str(erro)
    ) from erro

print(f"ok · fatiamento: {len(trechos)} trechos de {[len(t) for t in trechos]} caracteres")
print("ok · emenda: a pausa acompanha o dispositivo dos pedaços (falha de CUDA não passa mais)")

# ─── A voz de verdade ────────────────────────────────────────────────────────
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
