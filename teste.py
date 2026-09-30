"""
Fumaça do worker — roda DENTRO do build (CI), sem GPU e sem fila.

O que ele existe para pegar: o defeito que matou o card do Hub, em que a imagem construía bem e o
handler estourava no import. Aqui, se o handler quebrar ou a voz não sair, o build fica vermelho
antes de existir endpoint — em vez de virar trabalho parado em fila na madrugada.
"""

import os

# Precisa vir ANTES de importar o handler: o dispositivo é lido no import.
os.environ.setdefault("DISPOSITIVO", "cpu")

import handler  # noqa: E402

saida = handler.falar({"prompt": "Bom dia! Teste do worker de voz.", "language_id": "pt"})

assert saida.get("audio_base64"), saida
assert saida["language_id"] == "pt", saida
assert saida["segundos"] > 0.5, saida
assert saida["bytes"] > 10_000, saida
assert saida["clonou"] is False, saida

print(
    f"ok · {saida['segundos']}s de fala · {saida['bytes']} bytes · "
    f"idioma {saida['language_id']} · {saida['sample_rate']} Hz"
)
