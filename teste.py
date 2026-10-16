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

print(f"ok · fatiamento: {len(trechos)} trechos de {[len(t) for t in trechos]} caracteres")

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
