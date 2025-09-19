from agent.state import AgentState
from agent.node import node_triagem, node_auto_resolver,node_pedir_info,node_abrir_chamado, decidir_pos_auto_resolver, decidir_pos_triagem_principal
from langgraph.graph import StateGraph, START, END

# Cria e copila grafo do agente
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