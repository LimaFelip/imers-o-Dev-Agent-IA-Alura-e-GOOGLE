<h1>Imersão Dev Agentes de IA - Alura e Google</h1>

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

