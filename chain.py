# chain.py — Lógica da chain LangChain (LCEL)
# Separar a chain do app.py é uma boa prática: mantém o código organizado
# e facilita testes e reutilização.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------------------------
# System Prompt
# Define a personalidade e as regras do assistente.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
Você é o "Python Assistant", um assistente especialista em programação Python.
Sua missão é ajudar desenvolvedores com dúvidas de forma clara, precisa e didática.

REGRAS:
1. Responda apenas perguntas relacionadas a programação e tecnologia.
   Para outros assuntos, redirecione educadamente ao seu foco.

2. Estruture sempre suas respostas assim:
   - **Explicação**: conceito direto e didático.
   - **Código**: bloco Python bem comentado.
   - **Detalhes**: explique a lógica e as funções usadas.
   - **📚 Referência**: link para documentação oficial relevante.

3. Use linguagem clara. Evite jargões desnecessários.
"""

# ---------------------------------------------------------------------------
# build_chain(api_key)
# Recebe a API key do Groq e retorna uma chain LCEL pronta para uso.
#
# A chain segue o padrão LCEL (LangChain Expression Language):
#
#   prompt | llm | parser
#
# Cada "|" é um pipe que passa a saída de um componente para o próximo,
# similar ao pipe do Unix. Isso é o coração do LCEL.
# ---------------------------------------------------------------------------
def build_chain(api_key: str):
    """
    Constrói e retorna a chain LCEL do assistente.

    Componentes:
      - ChatPromptTemplate : monta as mensagens (system + histórico + input)
      - ChatGroq           : chama o modelo via API Groq
      - StrOutputParser    : extrai o texto puro da resposta do modelo
    """

    # 1. Prompt Template
    # MessagesPlaceholder injeta o histórico de conversa dinamicamente.
    # Isso é o que dá "memória" ao assistente sem precisar do LangChain Memory.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),  # histórico injetado aqui
        ("human", "{input}"),                          # pergunta atual do usuário
    ])

    # 2. Modelo LLM via Groq
    # streaming=True é necessário para transmitir tokens em tempo real.
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.7,
        max_tokens=2048,
        streaming=True,
    )

    # 3. Parser de saída
    # Transforma o objeto AIMessage retornado pelo LLM em uma string simples.
    parser = StrOutputParser()

    # 4. Montagem da chain com LCEL (pipe operator)
    chain = prompt | llm | parser

    return chain
