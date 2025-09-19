"""
Módulo responsável pela triagem inicial de mensagens de usuários.

Este script utiliza um modelo de linguagem do Google (Gemini) via LangChain
para classificar uma mensagem de entrada, decidindo a próxima ação,
a urgência e quais informações podem estar faltando.
"""

from config_key import GOOGLE_API_KEY
from prompts import TRIAGEM_PROMPT

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from typing import Literal, List, Dict
from pydantic import BaseModel, Field


"""
    A class Triagem Out Define a estrutura de dados para a saída da função de triagem.
    
    Este modelo Pydantic força o LLM a retornar uma resposta JSON.
"""
class TriagemOut(BaseModel):
    decisao: Literal["AUTO_RESOLVER", "PEDIR_INFO", "ABRIR_CHAMADO"]
    urgencia: Literal["BAIXA", "MEDIA", "ALTA"]
    campos_faltantes: List[str] = Field(default_factory=list)


# --- Configuração do Agente de IA ---

# Inicializa o modelo de chat do Google (Gemini) com as configurações desejadas.
# - model: Especifica a versão do modelo Gemini a ser usada.
# - temperature: Controla a criatividade da resposta (valores maiores = mais criativo).
# - api_key: Fornece a chave de autenticação para a API.
llm_triagem = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.8,
    api_key=GOOGLE_API_KEY)

# Cria uma "chain" (cadeia) que vincula o LLM à estrutura de saída `TriagemOut`.
# O método `.with_structured_output()` é a mágica que instrui o modelo a sempre
# formatar sua resposta de acordo com a classe Pydantic fornecida.    
triagem_chain = llm_triagem.with_structured_output(TriagemOut)


    # Executa a chain, enviando uma lista de mensagens para o modelo.
    # - SystemMessage: Contém as instruções de alto nível (o prompt de triagem).
    # - HumanMessage: Contém a entrada específica do usuário.
def triagem(mensagem: str) -> Dict:
    saida: TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])

    # Converte o objeto Pydantic de saída para um dicionário Python padrão e o retorna.
    # .model_dump() é o método moderno para fazer essa conversão.
    return saida.model_dump()


# para testar
# testes = ["Posso reembosar a internet?",
#                 "Quero mais 5 dias de trabalho remoto. Como faço?",
#                 "Quem descobriu o Brasil?" ]
# for msg_teste in testes:
#    print(f"Pergunta: {msg_teste}\n -> Resposta: {triagem(msg_teste)}\n")
#