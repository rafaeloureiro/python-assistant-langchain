# 🐍 Python Assistant

Assistente de programação Python construído com **LangChain**, **Groq** e **Streamlit**.

## Destaques técnicos

| Conceito | Implementação |
|---|---|
| **LCEL** | Chain montada com pipe operator: `prompt \| llm \| parser` |
| **Streaming** | Respostas transmitidas token a token via `.stream()` + `st.write_stream()` |
| **Chat History** | Histórico gerenciado com `HumanMessage` / `AIMessage` injetado via `MessagesPlaceholder` |
| **Separação de responsabilidades** | Lógica da chain isolada em `chain.py` |

## Estrutura do projeto

```
python-ai-assistant/
├── app.py           # Interface Streamlit
├── chain.py         # Chain LangChain (LCEL)
├── requirements.txt
└── README.md
```

## Como rodar

```bash
# 1. Instale as dependências
pip install -r requirements.txt

# 2. Rode o app
streamlit run app.py
```

Obtenha sua API Key gratuita em https://console.groq.com/keys

## Stack

-  [LangChain](https://python.langchain.com/) — LCEL + Streaming
-  [Groq](https://groq.com/) — LLaMA 3.3 70B
-  [Streamlit](https://streamlit.io/) — Interface web
