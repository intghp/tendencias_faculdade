# Trabalhos Práticos: Tendências em Ciência da Computação

**Aluno:** Gustavo Paiva

Este repositório contém as entregas das atividades práticas da disciplina de Tendências em Ciência da Computação, focadas em Recuperação de Informação (RI) e Processamento de Linguagem Natural (PLN).

---

## 📁 1. Laboratório Prático 04: Motor Léxico (BM25)
**Diretório:** `/aula04`

Laboratório interativo desenvolvido em Streamlit para demonstrar a mecânica do algoritmo Okapi BM25. A aplicação permite o upload de múltiplos documentos PDF e o ajuste dinâmico dos parâmetros matemáticos para observar as mudanças no ranqueamento em tempo real.
* **k1 (Saturação de Frequência):** Controla o limite de peso dado à repetição de termos.
* **b (Normalização de Tamanho):** Penaliza documentos mais longos para equilibrar a relevância.

**Como executar:**
```bash
cd aula04
streamlit run main.py
```

## 📁 2. Desafio Integrador: HealthSearch (Busca Híbrida)
**Diretório:** `/desafio_healthsearch`

Projeto de um Motor de Busca Híbrido projetado para o contexto de saúde (triagem médica e diretrizes clínicas). O sistema resolve as falhas individuais de motores puramente léxicos e semânticos.

Destaques Técnicos:

- **Busca Léxica (BM25):** Funciona como a pesquisa tradicional. Ele caça as palavras exatas ou códigos técnicos que o usuário digitou, garantindo que termos específicos nunca sejam ignorados.

- **Busca Semântica (IA):** Vai além das palavras exatas e foca no contexto e significado. Se o usuário buscar por "ataque cardíaco", a inteligência artificial entende que deve trazer resultados sobre "infarto", encontrando sinônimos de forma inteligente.

- **Fusão Híbrida (RRF):** Pega as melhores respostas da busca exata e as melhores da busca por contexto, criando um ranking único e equilibrado com o melhor dos dois mundos.

- **Cross-Encoder:** Uma camada extra de inteligência artificial que analisa minuciosamente os top-resultados da fusão e reorganiza a lista para garantir a máxima precisão possível na resposta final.

**Como executar:**
```bash
cd desafio_healthsearch
pip install -r requirements.txt
streamlit run healthsearch_app.py
```