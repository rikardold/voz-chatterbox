# Worker de voz em português (Chatterbox Multilingual)

Endpoint próprio para transformar **texto em fala** nos Reels, com opção de **clonar a voz** pela
amostra do apresentador.

## Por que não usar um card do Hub

O primeiro card que tentamos (`aydinmyilmaz/chatterbox-multilingual-tts`) construía a imagem e
morria na segunda linha do próprio handler:

```
from runpod.serverless.utils import upload_output
ImportError: cannot import name 'upload_output' from 'runpod.serverless.utils'
```

O container saía com código 1 e **nunca pedia trabalho na fila**. O sintoma que chega para quem
chama é enganoso: os workers aparecem como "prontos", o trabalho fica eternamente em `IN_QUEUE`, e
nada indica que o problema é o handler de outra pessoa — que não dá para editar.

## Entrada

| campo | padrão | o que é |
| --- | --- | --- |
| `prompt` | — | texto a falar (obrigatório) |
| `language_id` | `pt` | idioma. O modelo cobre 23: `ar da de el en es fi fr he hi it ja ko ms nl no pl pt ru sv sw tr zh` |
| `voice_url` | — | amostra de voz (6–15 s) para clonar. Pode ser webm/mp4: o worker converte |
| `exaggeration` | do modelo | emoção |
| `cfg_weight` | do modelo | aderência à amostra. **Use `0` quando a amostra estiver em outro idioma** |

## Saída

`audio_base64`, `sample_rate`, `segundos`, `bytes`, `language_id`, `clonou`.

**Por que base64 e não um link:** foi um utilitário do SDK (`upload_output`) que quebrou o outro
worker. Devolvendo os bytes na resposta, este worker não depende de nenhum helper que muda de versão
para versão. O custo é uma resposta maior; a troca vale.

## Deploy

A imagem é construída aqui no GitHub Actions e publicada em `ghcr.io/<usuario>/<repo>:latest`. O
endpoint no RunPod aponta para essa imagem (24 GB de GPU basta: o modelo tem 0,5B).

O build **testa o worker**: o passo de fumaça roda a imagem recém-construída e exige áudio de
verdade na saída. Se o handler quebrar no import — o defeito que motivou este repositório — o build
fica vermelho aqui, e não em produção.

## Limites conhecidos

- **Os pesos vêm dentro da imagem.** O build é lento e a imagem é grande; em troca, o worker novo já
  nasce sabendo falar, sem baixar 2–3 GB em tempo de GPU pago.
- **A fumaça roda em CPU**, então ela prova que o caminho funciona, não que ele é rápido.
- **Ainda não há o finetune dedicado de pt-BR.** Existe `ResembleAI/Chatterbox-Multilingual-pt-br`
  (Single Language Pack, sotaque e qualidade mais controlados), mas a ficha oficial não documenta
  como carregá-lo pela biblioteca. Aqui usamos o Multilingual V3 com `language_id="pt"`, que é o
  caminho documentado e verificável.
