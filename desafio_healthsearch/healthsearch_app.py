# =============================================================================
# SISTEMA AVANÇADO DE RECUPERAÇÃO HÍBRIDA (BM25 + FAISS + RRF + CROSS-ENCODER)
# Arquitetura Profissional para RAG e Search Engines (Revisado)
# =============================================================================

import os
import re
import time
import io
import logging
from typing import List, Dict, Tuple, Any

import streamlit as st
import pandas as pd
import numpy as np

# Ingestão de Documentos
import pypdf

# Processamento de Linguagem Natural e Tokenização
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Busca Léxica
from rank_bm25 import BM25Okapi

# Busca Semântica
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

# Configuração de Logging para depuração silenciosa
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Download preventivo de recursos do NLTK (Compatível com NLTK >= 3.9)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    # Fallback para versões antigas do NLTK
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)


# =============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Enterprise Hybrid Search & RAG Lab",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS para refinamento estético da interface
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; color: #0f2b48; font-weight: 700; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #475569; margin-bottom: 1.5rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0px 0px; padding: 8px 16px; background-color: #f1f5f9; }
    .stTabs [aria-selected="true"] { background-color: #0f2b48 !important; color: white !important; }
    .result-card { background-color: #f8fafc; border-left: 4px solid #0f2b48; padding: 12px; margin-bottom: 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MÓDULOS DE PROCESSAMENTO DE TEXTO E CHUNKING
# =============================================================================
class TextProcessor:
    """Classe responsável pelo pré-processamento avançado e tokenização do corpus."""
    
    def __init__(self, language: str = "portuguese"):
        self.language = language
        try:
            self.stop_words = set(stopwords.words(language))
        except Exception:
            self.stop_words = set()
        self.stemmer = SnowballStemmer(language)

    def clean_text(self, text: str) -> str:
        """Remove caracteres especiais, padroniza espaçamento, mas preserva hífens (útil para códigos)."""
        text = re.sub(r'\s+', ' ', text)  # Substitui qualquer espaço em branco (incluindo \n, \t) por espaço único
        text = re.sub(r'[^\w\s\-]', '', text) # Preserva letras, números, espaços e hífens
        return text.strip().lower()

    def tokenize(self, text: str, apply_stemming: bool = False) -> List[str]:
        """Tokeniza, remove stopwords e opcionalmente aplica stemming."""
        cleaned = self.clean_text(text)
        tokens = cleaned.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        if apply_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens


class DocumentChunker:
    """Divide textos longos em chunks semânticos com sobreposição ajustável."""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50, doc_name: str = "doc") -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []
            
        # Salvaguarda contra loop infinito
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size - 1)
            
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "chunk_id": f"{doc_name}_chunk_{chunk_id}",
                "doc_name": doc_name,
                "text": chunk_text,
                "start_word": start,
                "end_word": min(end, len(words)),
                "char_length": len(chunk_text)
            })
            
            chunk_id += 1
            start += (chunk_size - chunk_overlap)
            if start >= len(words):
                break
                
        return chunks


# =============================================================================
# CARREGAMENTO DE MODELOS E CACHING
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Carrega e armazena em cache o modelo de embedding denso."""
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def load_cross_encoder_model(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Carrega e armazena em cache o modelo de re-ranking (Cross-Encoder)."""
    return CrossEncoder(model_name)


# =============================================================================
# MOTOR DE BUSCA HÍBRIDA (ENGINE)
# =============================================================================
class HybridSearchEngine:
    """Engine unificada que integra BM25Okapi, FAISS, RRF e Cross-Encoder Re-Ranking."""
    
    def __init__(self, chunks: List[Dict[str, Any]], embedding_model_name: str):
        self.chunks = chunks
        self.processor = TextProcessor(language="portuguese")
        self.embedding_model = load_embedding_model(embedding_model_name)
        
        # Preparação do Corpus Léxico
        self.corpus_tokenized = [
            self.processor.tokenize(c["text"], apply_stemming=False) 
            for c in self.chunks
        ]
        
        # Inicialização Padrão do BM25
        self.bm25 = BM25Okapi(self.corpus_tokenized)
        
        # Construção do Índice Vetorial FAISS
        self._build_vector_index()

    def update_bm25_params(self, k1: float, b: float):
        """Atualiza dinamicamente os parâmetros k1 e b do BM25."""
        self.bm25 = BM25Okapi(self.corpus_tokenized, k1=k1, b=b)

    def _build_vector_index(self):
        """Gera os embeddings e constrói o índice FAISS (L2 Normalizado para Cosseno)."""
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalização L2 para calcular similaridade de cosseno via Produto Escalar
        faiss.normalize_L2(self.embeddings)
        dimension = self.embeddings.shape[1]
        
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)

    def search_lexical(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executa a busca léxica via BM25."""
        tokenized_query = self.processor.tokenize(query, apply_stemming=False)
        if not tokenized_query:
            return []
            
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "doc_name": self.chunks[idx]["doc_name"],
                "text": self.chunks[idx]["text"],
                "lexical_score": float(scores[idx]),
                "original_index": int(idx)
            })
        return results

    def search_semantic(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Executa a busca semântica vetorial via FAISS."""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk_id": self.chunks[idx]["chunk_id"],
                "doc_name": self.chunks[idx]["doc_name"],
                "text": self.chunks[idx]["text"],
                "semantic_score": float(score),
                "original_index": int(idx)
            })
        return results

    def hybrid_search_rrf(
        self, 
        query: str, 
        top_k: int = 10, 
        rrf_k: int = 60, 
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combina os rankings Léxico e Semântico utilizando Reciprocal Rank Fusion (RRF).
        Otimizado para não varrer o corpus inteiro, mas sim um pool de candidatos expandido.
        """
        # Otimização: Buscar um pool maior, mas limitado, em vez de len(self.chunks)
        retrieval_k = min(len(self.chunks), top_k * 5)
        
        lexical_results = self.search_lexical(query, top_k=retrieval_k)
        semantic_results = self.search_semantic(query, top_k=retrieval_k)
        
        rrf_scores: Dict[int, float] = {}
        item_data: Dict[int, Dict[str, Any]] = {}

        # Processar Ranks Léxicos
        for rank, res in enumerate(lexical_results):
            idx = res["original_index"]
            item_data[idx] = res
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + alpha * (1.0 / (rrf_k + (rank + 1)))

        # Processar Ranks Semânticos
        for rank, res in enumerate(semantic_results):
            idx = res["original_index"]
            if idx not in item_data:
                item_data[idx] = res
            else:
                item_data[idx]["semantic_score"] = res.get("semantic_score", 0.0)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 - alpha) * (1.0 / (rrf_k + (rank + 1)))

        # Ordenar pelo Score RRF acumulado
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for idx in sorted_indices:
            item = item_data[idx]
            results.append({
                "chunk_id": item["chunk_id"],
                "doc_name": item["doc_name"],
                "text": item["text"],
                "rrf_score": float(rrf_scores[idx]),
                "lexical_score": item.get("lexical_score", 0.0),
                "semantic_score": item.get("semantic_score", 0.0),
                "original_index": idx
            })
        return results

    def rerank_cross_encoder(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        cross_encoder_name: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Re-ordena os candidatos mais promissores via Cross-Encoder de alta precisão."""
        if not candidates:
            return []
            
        ce_model = load_cross_encoder_model(cross_encoder_name)
        pairs = [[query, c["text"]] for c in candidates]
        
        ce_scores = ce_model.predict(pairs)
        
        for i, score in enumerate(ce_scores):
            candidates[i]["cross_score"] = float(score)

        sorted_candidates = sorted(candidates, key=lambda x: x["cross_score"], reverse=True)[:top_k]
        return sorted_candidates


# =============================================================================
# PAINEL LATERAL (SIDEBAR) & CONTROLES
# =============================================================================
st.sidebar.title("⚙️ Painel de Engenharia")
st.sidebar.markdown("---")

st.sidebar.subheader("1. Parâmetros BM25 (Léxico)")
k1 = st.sidebar.slider("Term Saturation (k1)", min_value=0.0, max_value=3.0, value=1.2, step=0.1, 
                       help="Controla a saturação da frequência de termos.")
b = st.sidebar.slider("Length Normalization (b)", min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                      help="Controla a penalização pelo tamanho do documento.")

st.sidebar.markdown("---")
st.sidebar.subheader("2. Fusão Híbrida (RRF)")
alpha = st.sidebar.slider("Peso BM25 vs Semântico (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                          help="α = 1.0 (Apenas BM25) | α = 0.0 (Apenas Semântico)")
rrf_k = st.sidebar.number_input("Constante RRF (k)", min_value=1, max_value=100, value=60, step=5,
                                help="Suaviza o peso de posições mais baixas no rank.")

st.sidebar.markdown("---")
st.sidebar.subheader("3. Pipeline & Re-Ranking")
use_reranker = st.sidebar.checkbox("Ativar Cross-Encoder Re-Ranking", value=True)
top_k_retrieval = st.sidebar.slider("Candidatos Híbridos (Top-K)", min_value=3, max_value=30, value=10)
top_k_rerank = st.sidebar.slider("Resultado Final Exibido", min_value=1, max_value=10, value=5)


# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================
st.markdown('<div class="main-header">🔍 Enterprise Hybrid Search & RAG Laboratory</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Plataforma avançada para calibração, análise e benchmark de recuperação léxica, semântica e fusão RRF.</div>', unsafe_allow_html=True)

# Ingestão de Documentos
st.subheader("📁 Ingestão de Corpus e Configuração")

upload_option = st.radio("Selecione a Origem do Corpus:", ["Usar Corpus Exemplo Avançado (Multidomínio)", "Upload de Documentos (PDF / TXT)"], horizontal=True)

raw_chunks = []

if upload_option == "Usar Corpus Exemplo Avançado (Multidomínio)":
    default_docs = [
        {"doc_name": "MANUAL_TECNICO_X100", "text": "Erro CÓD-404-SYS em servidores Linux ocorre por falha de alocação de memória buffer na rotina C++. Ajuste a flag --max-mem 8GB no arquivo /etc/sysconfig/app.conf."},
        {"doc_name": "DIRETRIZ_CLINICA_CARDIOLOGIA", "text": "O tratamento de insuficiência cardíaca congestiva exige acompanhamento da fração de ejeção do ventrículo esquerdo. Medicamentos vasodilatadores e estatinas são recomendados."},
        {"doc_name": "LEGISLAÇÃO_TRABALHISTA_CLT", "text": "Artigo 59 da CLT regulamenta que a duração diária do trabalho poderá ser acrescida de horas suplementares, em número não excedente de duas, mediante acordo individual."},
        {"doc_name": "CONTRATO_MERCADO_LIVRE_ENERGIA", "text": "A contratação do suprimento elétrico no ambiente de contratação livre (ACL) permite a negociação de tarifas PPA flexíveis baseadas na curva PLD da CCEE."},
        {"doc_name": "ARQUITETURA_RAG_PIPELINE", "text": "Em arquiteturas de IA Generativa, a busca híbrida combina a precisão do BM25 para termos raros e códigos com embeddings densos para captura de contexto semântico."},
        {"doc_name": "GLOSSARIO_MEDICO_SINONIMOS", "text": "Pacientes apresentando infarto agudo do miocárdio ou dor precordial intensa devem ser submetidos à angioplastia de emergência no pronto-socorro."},
        {"doc_name": "MANUAL_SISTEMAS_CONFIG", "text": "Para resolver a falha de sistema CÓD-404-SYS em contêineres Docker, certifique-se de reiniciar o daemon e limpar o cache de swap do nó master."}
    ]
    for d in default_docs:
        chunks = DocumentChunker.chunk_text(d["text"], chunk_size=150, chunk_overlap=30, doc_name=d["doc_name"])
        raw_chunks.extend(chunks)
else:
    uploaded_files = st.file_uploader("Faça upload de arquivos PDF ou TXT:", type=["pdf", "txt"], accept_multiple_files=True)
    if uploaded_files:
        for file in uploaded_files:
            try:
                if file.name.lower().endswith(".pdf"):
                    reader = pypdf.PdfReader(io.BytesIO(file.read()))
                    full_text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                else:
                    # errors="replace" evita falhas em arquivos TXT com codificação inesperada (ex: Latin-1)
                    full_text = file.read().decode("utf-8", errors="replace")
                
                chunks = DocumentChunker.chunk_text(full_text, chunk_size=200, chunk_overlap=40, doc_name=file.name)
                raw_chunks.extend(chunks)
            except Exception as e:
                st.error(f"⚠️ Erro ao processar o arquivo **{file.name}**: {str(e)}")

if not raw_chunks:
    st.warning("⚠️ Nenhum documento carregado. Por favor, selecione o corpus padrão ou faça o upload de arquivos válidos.")
    st.stop()

st.success(f"✅ Corpus processado com sucesso: **{len(raw_chunks)} chunks** prontos para indexação.")

# Inicialização da Engine de Busca na Sessão
@st.cache_resource(show_spinner="Indexando corpus em FAISS e preparando BM25...")
def get_engine(chunks_data: List[Dict[str, Any]]):
    return HybridSearchEngine(chunks_data, embedding_model_name="sentence-transformers/all-MiniLM-L6-v2")

engine = get_engine(raw_chunks)
# Atualiza os parâmetros dinamicamente com base nos controles do sidebar
engine.update_bm25_params(k1=k1, b=b)

# =============================================================================
# CONSULTA E EXPERIMENTAÇÃO
# =============================================================================
st.markdown("---")
st.subheader("🔎 Painel de Consulta e Diagnóstico")

query = st.text_input("Digite sua consulta (ex: 'CÓD-404-SYS', 'tratamento de ataque cardíaco', 'horas extras CLT'):", 
                      value="CÓD-404-SYS em servidores Linux")

if query:
    start_time = time.time()
    
    # Execução das 3 Modalidades de Busca
    results_lexical = engine.search_lexical(query, top_k=top_k_retrieval)
    results_semantic = engine.search_semantic(query, top_k=top_k_retrieval)
    results_hybrid = engine.hybrid_search_rrf(query, top_k=top_k_retrieval, rrf_k=rrf_k, alpha=alpha)
    
    # Re-Ranking (opcional)
    if use_reranker and results_hybrid:
        results_final = engine.rerank_cross_encoder(
            query=query, 
            candidates=results_hybrid, 
            cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=top_k_rerank
        )
    else:
        results_final = results_hybrid[:top_k_rerank]
        
    execution_time = (time.time() - start_time) * 1000

    st.caption(f"⚡ Tempo de execução da consulta: **{execution_time:.2f} ms**")

    # Visualização em Abas Comparativas
    tab_overview, tab_lexical, tab_semantic, tab_hybrid, tab_rerank = st.tabs([
        "📊 Matriz Comparativa", 
        "🔤 Busca Léxica (BM25)", 
        "🧠 Busca Semântica (FAISS)", 
        "🔀 Fusão Híbrida (RRF)", 
        "🎯 Resultado Final (Cross-Encoder)"
    ])

    # ABA 1: MATRIZ COMPARATIVA
    with tab_overview:
        st.markdown("### Análise da Posição dos Resultados nas Diferentes Abordagens")
        
        all_chunk_ids = list({r["chunk_id"] for r in (results_lexical + results_semantic + results_hybrid)})
        comparison_data = []

        lex_rank_map = {r["chunk_id"]: i+1 for i, r in enumerate(results_lexical)}
        sem_rank_map = {r["chunk_id"]: i+1 for i, r in enumerate(results_semantic)}
        hyb_rank_map = {r["chunk_id"]: i+1 for i, r in enumerate(results_hybrid)}

        for cid in all_chunk_ids:
            chunk_info = next((c for c in raw_chunks if c["chunk_id"] == cid), None)
            if chunk_info:
                comparison_data.append({
                    "Doc ID": chunk_info["doc_name"],
                    "Texto (Trecho)": chunk_info["text"][:80] + "...",
                    "Rank BM25": lex_rank_map.get(cid, "N/A"),
                    "Rank Semântico": sem_rank_map.get(cid, "N/A"),
                    "Rank RRF Híbrido": hyb_rank_map.get(cid, "N/A")
                })

        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        st.info("💡 **Insights:** Note como códigos de erro ou siglas específicas obtêm destaque no **Rank BM25**, enquanto consultas com sinônimos semânticos se saem melhor no **Rank Semântico**. O **RRF Híbrido** equilibra ambas as forças.")

    # ABA 2: BUSCA LÉXICA
    with tab_lexical:
        st.markdown(f"### Ranking BM25 (k1={k1}, b={b})")
        if results_lexical:
            df_lex = pd.DataFrame(results_lexical)[["doc_name", "lexical_score", "text"]]
            st.dataframe(df_lex, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum match exato encontrado para os termos da consulta.")

    # ABA 3: BUSCA SEMÂNTICA
    with tab_semantic:
        st.markdown("### Ranking Vetorial de Densidade (Cosine Similarity via FAISS)")
        if results_semantic:
            df_sem = pd.DataFrame(results_semantic)[["doc_name", "semantic_score", "text"]]
            st.dataframe(df_sem, use_container_width=True, hide_index=True)
        else:
            st.info("Nenhum vetor próximo encontrado.")

    # ABA 4: FUSÃO HÍBRIDA (RRF)
    with tab_hybrid:
        st.markdown(f"### Reciprocal Rank Fusion (α={alpha}, k={rrf_k})")
        if results_hybrid:
            df_hyb = pd.DataFrame(results_hybrid)[["doc_name", "rrf_score", "lexical_score", "semantic_score", "text"]]
            st.dataframe(df_hyb, use_container_width=True, hide_index=True)

    # ABA 5: CROSS-ENCODER RE-RANKING
    with tab_rerank:
        st.markdown("### Ranking Final com Cross-Encoder Re-Ranker")
        st.markdown("Os top resultados do RRF foram submetidos a um modelo **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) que avalia o par `(Consulta, Documento)` de forma profunda para atribuir uma nota de relevância real.")

        for i, res in enumerate(results_final, 1):
            st.markdown(f"""
            <div class="result-card">
                <strong>#{i} - Documento:</strong> <code>{res['doc_name']}</code><br>
                <strong>Trecho:</strong> {res['text']}
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b, col_c = st.columns(3)
            if "cross_score" in res:
                col_a.metric("Score Cross-Encoder", f"{res['cross_score']:.4f}")
            col_b.metric("Score RRF Híbrido", f"{res['rrf_score']:.4f}")
            col_c.metric("BM25 / Semântico", f"{res['lexical_score']:.2f} / {res['semantic_score']:.2f}")
            st.markdown("---")