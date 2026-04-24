# app.py — Interface Streamlit do Python Assistant
#
# Destaques LangChain usados neste projeto:
#   ✅ LCEL  : chain montada com o pipe operator (prompt | llm | parser)
#   ✅ Stream: resposta transmitida token a token com .stream()
#   ✅ Prompt: ChatPromptTemplate com MessagesPlaceholder para histórico

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Importa a função que constrói a chain (definida em chain.py)
from chain import build_chain

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Python Assistant",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Python Assistant")
    st.markdown("Assistente de programação Python construído com **LangChain + Groq**.")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Obtenha sua chave gratuita em https://console.groq.com/keys",
    )

    st.markdown("---")

    # Botão para limpar o histórico de conversa
    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Project Stack:**")
    st.markdown("- LangChain (LCEL + Streaming)")
    st.markdown("- Groq — LLaMA 3.3 70B")
    st.markdown("- Streamlit")

# ---------------------------------------------------------------------------
# Cabeçalho principal
# ---------------------------------------------------------------------------
st.title("Python Assistant 🐍")
st.caption("Powered by LangChain · Groq · LLaMA 3.3 70B · LCEL + Streaming")

# ---------------------------------------------------------------------------
# Estado da sessão
# Armazena o histórico como objetos LangChain (HumanMessage / AIMessage).
# Isso permite injetar o histórico diretamente no MessagesPlaceholder da chain.
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # lista de HumanMessage / AIMessage

# ---------------------------------------------------------------------------
# Renderiza o histórico de mensagens na tela
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ---------------------------------------------------------------------------
# Input do usuário
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Qual sua dúvida sobre Python?"):

    # Valida se a API key foi fornecida
    if not groq_api_key:
        st.warning("⚠️ Insira sua Groq API Key na barra lateral para começar.")
        st.stop()

    # Exibe a mensagem do usuário imediatamente
    with st.chat_message("user"):
        st.markdown(user_input)

    # Salva a mensagem do usuário no histórico como HumanMessage
    st.session_state.messages.append(HumanMessage(content=user_input))

    # -----------------------------------------------------------------------
    # Geração da resposta com streaming
    # -----------------------------------------------------------------------
    with st.chat_message("assistant"):
        try:
            # Constrói a chain com a API key atual
            chain = build_chain(groq_api_key)

            # chain.stream() retorna um generator que emite tokens um a um.
            # st.write_stream() consome esse generator e renderiza o texto
            # na tela em tempo real — sem precisar de nenhum código extra.
            response_stream = chain.stream({
                "history": st.session_state.messages[:-1],  # histórico sem a msg atual
                "input": user_input,
            })

            # Renderiza o stream e captura o texto completo ao final
            full_response = st.write_stream(response_stream)

        except Exception as e:
            st.error(f"Erro ao chamar a API da Groq: {e}")
            st.stop()

    # Salva a resposta completa no histórico como AIMessage
    st.session_state.messages.append(AIMessage(content=full_response))

# ---------------------------------------------------------------------------
# Rodapé
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="text-align: center; color: gray; margin-top: 2rem;">
        <hr>
        <p>Python Assistant · LangChain LCEL + Streaming · Groq</p>
    </div>
    """,
    unsafe_allow_html=True,
)
