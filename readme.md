<h1>Imersão Dev Agentes de IA - Alura e Google de 08 á 15 de setembro de 2025  </h1>

![Alura](https://github.com/LimaFelip/imers-o-Dev-Agent-IA-Alura-e-GOOGLE/blob/main/docs/alura%201.jpg?raw=true)

# Instalação de Pacotes
    pip install -U langchain
    pip install -q --upgrade langchain-google-genai google-generativeai
    pip install -q --upgrade langchain_community faiss-cpu langchain-text-splitters pymupdf
    pip install -q --upgrade langgraph


<h1>Aula 1</h1>

• Configurar o modelo Gemini 2.5 Flash no LangChain.  
• Escrever prompts de sistema claros para triagem de mensagens.   
• Estruturar saídas em JSON com Pydantic para garantir previsibilidade.   
• Criar uma chain de triagem para classificar mensagens em três categorias.   
• Desenvolver um classificador funcional como base do agente.   

•	O que é o Google Gemini e o que esse modelo de IA é capaz de fazer — com exemplo prático | Alura      
      https://www.alura.com.br/artigos/google-gemini

•	Google Colab: o que é, tutorial de como usar e criar códigos.     
    https://www.alura.com.br/artigos/google-colab-o-que-e-e-como-usar

<h1>Aula 2</h1>

• Carregar e processar documentos PDF.    
• Dividir textos longos em chunks para otimizar a busca de informações.   
• Criar embeddings e armazenar em uma Vector Store com FAISS.   
• Construir uma chain RAG que busca contexto e gera respostas baseadas em documentos.   
• Formatar respostas com citações exatas das fontes consultadas.    

•	O que é RAG e como essa técnica funciona | Alura    
    https://www.alura.com.br/artigos/o-que-e-rag

•	LlamaIndex: onde é aplicado | Alura    
    https://www.alura.com.br/artigos/llamaindex


<h1>Aula 3</h1>

• Definir um estado do agente para armazenar informações do fluxo.    
• Transformar as principais funções (triagem, auto-resolver, pedir informação, abrir chamado) em nós de grafo.    
• Implementar a lógica de roteamento condicional entre os nós.     
• Montar e compilar o grafo no LangGraph para execução do fluxo completo.    
• Visualizar o fluxo em diagrama para entender as decisões do agente.   

![Diagrama](https://github.com/LimaFelip/imers-o-Dev-Agent-IA-Alura-e-GOOGLE/blob/main/docs/diagrama_completo.jpg?raw=true)

•	Agentes de IA com LangGraph     
    https://www.alura.com.br/conteudo/flash-skills-agentes-ia-langgraph    

•	LangChain: criando chatbots inteligentes com RAG     
    https://www.alura.com.br/conteudo/langchain-criando-chatbots-inteligentes-rag    

•	Como agentes potencializam a performance das LLMs    
    https://www.alura.com.br/artigos/como-agentes-podem-ajudar-llms


# Além da imersão da Alura

# Clean CODE     

Foi implementado arquitetura de pastas e codigo com base nos principios de "Clean Code"     
    Princípio da Responsabilidade Única (Single Responsibility Principle)     
    
        AgentAPI/     
        ├── agent/    
        │   ├── __init__.py             # Faz 'agent' ser um pacote Python    
        │   ├── graph.py                # Lógica para construir o grafo    
        │   ├── nodes.py                # Todas as funções dos nós    
        │   ├── prompts.py              # Todos os textos de prompt    
        │   ├── state.py                # Definição do AgentState e outros modelos   
        |   └── screening.py            # Módulo responsável pela triagem inicial de mensagens de usuários   
        ├── docs/                       # Armazena os arquivos .pdfs que foram utilizados    
        |   ├── Política de Reembolsos (Viagens e Despesas).pdf      
        |   ├── Política de Uso de E-mail e Segurança da Informação.pdf    
        |   ├── Políticas de Home Office.pdf   
        ├── utils/     
        │   ├── __init__.py    
        │   └── document_loader.py   # Função para carregar e dividir os PDFs   
        ├── .env                     # Arquivo para guardar a GOOGLE_API_KEY   
        ├── .gitignore   
        ├── config_key.py                   # Configurações da api principal   
        ├── main.py                     # Nosso novo e limpo arquivo principal   
        └── venv/     

*obs.  Com base na segurança alguns arquivos foram ocultos.


# Tratamento de ERROS
Foi acrecentado tratamento erros  usando o pacote from google.api_core.exceptions import ResourceExhausted, InvalidArgument 
        
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
