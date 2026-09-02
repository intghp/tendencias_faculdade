import streamlit as st
import math
import re
import string
import pandas as pd
from pypdf import PdfReader

st.set_page_config(page_title="Laboratório BM25", layout="wide")
st.title("⚙️ Laboratório de Rank BM25")

# --- Funções Auxiliares ---
def extrair_texto_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    texto = ""
    for page in reader.pages:
        conteudo = page.extract_text()
        if conteudo:
            texto += conteudo + " "
    return texto.strip()

def tokenizar(texto):
    texto = texto.lower()
    texto = re.sub(f"[{re.escape(string.punctuation)}]", " ", texto)
    return [palavra for palavra in texto.split() if len(palavra) > 0]

# --- Sidebar: Entrada de Documentos ---
st.sidebar.header("📁 Upload de PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Envie seus documentos (ex: docs1.pdf, docs2.pdf...)", 
    type=["pdf"], 
    accept_multiple_files=True
)

docs = {}
if uploaded_files:
    for arquivo in uploaded_files:
        docs[arquivo.name] = extrair_texto_pdf(arquivo)

# --- Controle de Exibição Principal ---
if docs:
    
    col_query, col_k1, col_b = st.columns([2, 1, 1])
    with col_query:
        query_input = st.text_input("Termo(s) de busca (Query):", value="inteligência")
    with col_k1:
        k1 = st.slider("Parâmetro k1 (Saturação)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
    with col_b:
        b = st.slider("Parâmetro b (Tamanho do doc)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

    # --- Processamento e Cálculo ---
    tokens_docs = {nome: tokenizar(texto) for nome, texto in docs.items()}
    query_tokens = tokenizar(query_input)

    dl = {nome: len(tokens) for nome, tokens in tokens_docs.items()}
    N = len(docs)
    avgdl = sum(dl.values()) / N if N > 0 else 1

    scores = {nome: 0.0 for nome in docs.keys()}
    detalhes_termos = {nome: {} for nome in docs.keys()}

    if N > 0 and len(query_tokens) > 0:
        for termo in query_tokens:
            df_t = sum(1 for tokens in tokens_docs.values() if termo in tokens)
            
            if df_t > 0:
                idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
            else:
                idf = 0.0

            for nome, tokens in tokens_docs.items():
                f = tokens.count(termo)
                detalhes_termos[nome][termo] = f
                
                numerador = f * (k1 + 1)
                denominador = f + k1 * (1 - b + b * (dl[nome] / avgdl))
                score_termo = idf * (numerador / denominador) if denominador > 0 else 0.0
                scores[nome] += score_termo

    # --- Montagem da Tabela Original ---
    resultados = []
    for nome in docs.keys():
        resultados.append({
            "Documento": nome,
            "Tamanho (|D|)": dl[nome],
            "Frequência dos Termos": ", ".join([f"{k}: {v}" for k, v in detalhes_termos[nome].items()]),
            "Score BM25": round(scores[nome], 4)
        })

    df_bm25 = pd.DataFrame(resultados).sort_values(by="Score BM25", ascending=False)
    
    st.subheader("🏆 Ranking Resultante")
    st.dataframe(df_bm25, use_container_width=True)
    st.markdown(f"**Média de palavras dos documentos ($avgdl$):** `{round(avgdl, 2)}`")
    
    # --- Amostragem Simples do Conteúdo ---
    st.divider()
    st.subheader("📖 Amostragem do Conteúdo dos Documentos")
    
    abas = st.tabs(list(docs.keys()))
    
    for i, (nome, texto) in enumerate(docs.items()):
        with abas[i]:
            # Mostra a quantidade de palavras naquele doc
            st.markdown(f"**Tamanho:** `{dl[nome]}` palavras lidas com sucesso.")
            
            st.text_area("Texto bruto extraído (use a rolagem para ver tudo):", 
                         value=texto, 
                         height=300, 
                         disabled=True, 
                         key=f"preview_{i}")

else:
    # Tela mostrada quando não há nenhum arquivo carregado
    st.info("👋 Bem-vindo ao Laboratório BM25! Para começar, faça o upload dos seus arquivos PDF no menu lateral à esquerda.")