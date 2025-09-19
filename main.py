
from agent.graph import grafo
import time
from google.api_core.exceptions import ResourceExhausted, InvalidArgument 


testes = ["Posso reembolsar a internet?"]


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
