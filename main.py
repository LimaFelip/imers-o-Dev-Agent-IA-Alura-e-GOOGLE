import time
from agent_state import AgentState
from TRIAGEM_PROMPT import TRIAGEM_PROMPT
from formatters import formatar_citacoes
from config_key import GOOGLE_API_KEY
from screening import triagem, llm_triagem

import google.generativeai as genai
from typing import Dict
from pathlib import Path

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END


from google.api_core.exceptions import ResourceExhausted, InvalidArgument 



# langchain_community - para fazer as conexções
# faiss-cpu para criar a similaridade de texto
# langchain-text-splitters - para quebrar o texto em pequenos pedaços
# pymupdf - para ler os PDF
docs =[]

for n in Path("./docs/").glob("*.pdf"):
  try:
    loader = PyMuPDFLoader(str(n))
    docs.extend(loader.load())
    print(f"Carregado com sucesso: {n.name}")
  except Exception as e:
    print(f"Erro ao carregar: {n.name}: {e}")
# --------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

def node_triagem(state: AgentState ) -> AgentState:
  print("Executando nó de triagem...")
  return {"triagem": triagem(state["pergunta"])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("Executando nó de auto_resolver...")
    resposta_rag = perguntar_politica_RAG(state["pergunta"])

    update: AgentState = {
        "resposta": resposta_rag["answer"],
        "citacoes": resposta_rag.get("citacoes", []),
        "rag_sucesso": resposta_rag["contexto_encontrado"],
    }

    if resposta_rag["contexto_encontrado"]:
        update["acao_final"] = "AUTO_RESOLVER"

    return update

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY
)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.3, "k": 4}
)

prompt_rag = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um Assistente de Políticas Internas (RH/IT) da empresa Carraro Desenvolvimento. "
     "Responda SOMENTE com base no contexto fornecido. "
     "Se não houver base suficiente, responda apenas 'Não sei'."),

    ("human", "Pergunta: {input}\n\nContexto:\n{context}")
])

document_chain = create_stuff_documents_chain(llm_triagem, prompt_rag)


def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    answer = document_chain.invoke({"input": pergunta,
                                    "context": docs_relacionados})

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {"answer": "Não sei.",
                "citacoes": [],
                "contexto_encontrado": False}

    return {"answer": txt,
            "citacoes": formatar_citacoes(docs_relacionados, pergunta),
            "contexto_encontrado": True}

def node_pedir_info(state: AgentState) -> AgentState:
    print("Executando nó de pedir_info...")
    faltantes = state["triagem"].get("campos_faltantes", [])
    detalhe = ",".join(faltantes)  if faltantes else "Tema e contexto específico" 

    return {
        "resposta": f"Para avançar, preciso que detalhe: {detalhe}",
        "citacoes": [],
        "acao_final": "PEDIR_INFO"
    }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("Executando nó de abrir_chamado...")
    triagem =state["triagem"]

    return {
      "resposta": f"Abrindo chamdado com urgencia {triagem ['urgencia']}. Descrição: {state['pergunta'][:140]}", 
      "citacoes": [],
      "acao_final": "ABRIR_CHAMADO"
    }


KEYWORDS_ABRIR_TICKET = ["aprovação", "exceção", "liberação", "abrir ticket", "abrir chamado", "acesso especial"]

def decidir_pos_triagem_principal(state: AgentState) -> str:
    print("Decidindo após a triagem...")
    decisao = state["triagem"]["decisao"]

    if decisao == "AUTO_RESOLVER": return "auto"
    if decisao == "PEDIR_INFO": return "info"
    if decisao == "ABRIR_CHAMADO": return "chamado"


def decidir_pos_auto_resolver(state: AgentState) -> str:
  print("Decidindo após o auto_resolver....")

  if state.get("rag_sucesso"):
    print("Rag com sucesso, finalizando o atendimento")
    return "ok"
  
  state_da_pergunta = (state["pergunta"] or " ").lower()

  if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
    print("Rag falhou, mas foram encontradas Keywords de ticket. Abrindo...")
    return "abrir_chamado"
  
  print("Rag falhou, sem Keywords, vou pedir mais informações...")
  return "pedir_info"

workflow = StateGraph(AgentState)

workflow.add_node("triagem", node_triagem)  
workflow.add_node("auto_resolver", node_auto_resolver)  
workflow.add_node("pedir_info", node_pedir_info)  
workflow.add_node("abrir_chamado", node_abrir_chamado)  


workflow.add_edge(START, "triagem")
workflow.add_conditional_edges("triagem", decidir_pos_triagem_principal,{
    "auto": "auto_resolver",
    "info": "pedir_info",
    "chamado": "abrir_chamado"                                                 
} )

workflow.add_conditional_edges("auto_resolver", decidir_pos_auto_resolver, {
    "info": "pedir_info",
    "chamado": "abrir_chamado",
    "ok": END
})

workflow.add_edge("pedir_info", END)
workflow.add_edge("abrir_chamado", END)

grafo = workflow.compile()

testes = input()


for msg_test in testes:

    try: 
        resposta_final = grafo.invoke({"pergunta": msg_test})

        triag = resposta_final.get("triagem", {})
        print(f"PERGUNTA: {msg_test}")
        print(f"DECISÃO: {triag.get('decisao')} | URGÊNCIA: {triag.get('urgencia')} | AÇÃO FINAL: {resposta_final.get('acao_final')}")
        print(f"RESPOSTA: {resposta_final.get('resposta')}")

        if resposta_final.get("citacoes"):
            print("CITAÇÕES:")
            for citacao in resposta_final.get("citacoes"):
                print(f" - Documento: {citacao['documento']}, Página: {citacao['pagina']}")
                print(f"   Trecho: {citacao['trecho']}")
    

        # NOVO BLOCO PARA CAPTURAR O ERRO DE CHAVE INVÁLIDA/EXPIRADA
    except InvalidArgument as e:
        # Verificamos a mensagem de erro para ter certeza que é sobre a API Key
        if "API key expired" in str(e) or "API_KEY_INVALID" in str(e):
            print("\n❌ ERRO: Sua chave de API do Google expirou ou é inválida.")
            print("Por favor, gere uma nova chave e atualize sua variável de ambiente.")
            resposta_final = {"erro": "Chave de API expirada ou inválida (400)"}
            break # Interrompe o loop se a chave é inválida
        else:
            # Se for outro tipo de InvalidArgument, tratamos como um erro inesperado
            print(f"\n❌ ERRO: Ocorreu um problema com os dados enviados para a API (Bad Request).")
            print(f"Detalhe do erro: {e}")
            resposta_final = {"erro": "Ocorreu um erro de 'Bad Request' (400) na execução."}
            

    except ResourceExhausted as e:
        # BLOCO QUE VOCÊ JÁ TINHA: Captura o erro específico de limite de uso (429)
        print("\n❌ ERRO: Limite de uso da API do Google foi excedido.")
        print("Isso acontece quando muitas chamadas são feitas em um curto período ou o limite diário foi atingido.")
        print("Por favor, aguarde e tente novamente mais tarde.")
        resposta_final = {"erro": "Limite de cota da API atingido (429)"}
        time.sleep(60) # Pausa por 1 minuto e tenta continuar
        continue # Pula para a próxima pergunta

    except Exception as e:
        # BLOCO DE SEGURANÇA: Captura qualquer outro erro inesperado
        print(f"\n❌ ERRO: Ocorreu um problema inesperado durante a execução do agente.")
        print(f"Detalhe do erro: {e}")
        resposta_final = {"erro": "Ocorreu um erro genérico na execução."}
        continue # Pula para a próxima pergunta

    print("------------------------------------")
