from state import AgentState
from screening import triagem
from formatters import formatar_citacoes
from typing import Dict

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