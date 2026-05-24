import os
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_tavily import TavilySearch


class AgenticRAG:

    def __init__(
        self,
        pdf_files: list[str],
        openai_api_key: str  = "API_KEY",
        tavily_api_key: str  = "API_KEY",
        model: str           = "gpt5-mini",
        verbose: bool        = True,
    ):


        self.pdf_files = pdf_files
        self.verbose   = verbose

        os.environ["OPENAI_API_KEY"]  = openai_api_key
        os.environ["TAVILY_API_KEY"]  = tavily_api_key

        self.llm = ChatOpenAI(model=model, temperature=0)

        pages          = self._load_pdfs()
        chunks         = self._adaptive_split(pages)
        vector_store   = self._build_vector_store(chunks)
        tools          = self._create_tools(vector_store)
        self.memory    = self._create_memory()
        self.agent     = self._build_agent(tools)

        print("\n🚀 AgenticRAG is ready! Call .chat('your question') to start.\n")



    def _load_pdfs(self) -> list:
        
        all_pages = []

        for pdf_path in self.pdf_files:
            if not Path(pdf_path).exists():
                print(f"⚠️'{pdf_path}'– file not found")
                continue

            print(f"📄 Loading: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages  = loader.load()
            all_pages.extend(pages)
            print("pages loaded")

        if not all_pages:
            raise ValueError("❌ No pages loaded.")

        print(f"✅ Total pages: {len(all_pages)}")
        return all_pages


    def _get_chunk_size(self, text: str) -> int:

        length = len(text)

        if length < 2_000:
            return 300
        elif length < 10_000:
            return 600
        else:
            return 1_000


    def _adaptive_split(self, pages: list) -> list:

        all_chunks = []

        for page in pages:
            chunk_size = self._get_chunk_size(page.page_content)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size    = chunk_size,
                chunk_overlap = int(chunk_size * 0.15),   
                separators    = ["\n\n", "\n", ". ", " ", ""],
            )

            chunks = splitter.split_documents([page])
            all_chunks.extend(chunks)

        print(f"✅ Adaptive chunking → {len(all_chunks)} chunks total")
        return all_chunks


    def _build_vector_store(self, chunks: list) -> FAISS:

        embeddings   = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("✅ Vector store ready")
        return vector_store


    def _create_tools(self, vector_store: FAISS) -> list:

        # Tool 1: PDF search
        retriever      = vector_store.as_retriever(search_kwargs={"k": 4})
        retriever_tool = create_retriever_tool(
            retriever   = retriever,
            name        = "search_pdf_documents",
            description = (
                "Search through the uploaded PDF documents. "
                "Always try this tool first before going to the web."
            ),
        )

        # Tool 2: Web search
        web_search_tool = TavilySearch(
            max_results = 3,
            name        = "web_search",
            description = (
                "Search the internet for up-to-date information. "
                "Use this only when the PDFs don't have the answer."
            ),
        )

        tools = [retriever_tool, web_search_tool]
        print(f"✅ Tools ready: {[t.name for t in tools]}")
        return tools


    def _create_memory(self) -> ConversationBufferMemory:

        memory = ConversationBufferMemory(
            memory_key    = "chat_history",   
            return_messages = True,           
        )
        print("✅ Memory ready")
        return memory


    def _build_agent(self, tools: list) -> AgentExecutor:

        # agents prompt
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful research assistant.\n"
                "You have two tools:\n"
                "  1. search_pdf_documents – search the uploaded PDFs\n"
                "  2. web_search           – search the internet\n\n"
                "Rules:\n"
                "  - Always try the PDF tool first.\n"
                "  - Use web search only if the PDFs don't have the answer.\n"
                "  - If you used both tools, say where each piece came from.\n"
                "  - Keep answers clear and simple."
            ),
            MessagesPlaceholder(variable_name="chat_history"),   
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_tools_agent(
            llm    = self.llm,
            tools  = tools,
            prompt = prompt,
        )

        agent_executor = AgentExecutor(
            agent          = agent,
            tools          = tools,
            memory         = self.memory,    
            verbose        = self.verbose,
            max_iterations = 6,
        )

        print("✅ Agent ready")
        return agent_executor


    def chat(self, user_message: str) -> str:

        print(f"\n{'='*55}")
        print(f"You: {user_message}")
        print(f"{'='*55}")

        result = self.agent.invoke({"input": user_message})
        answer = result["output"]

        print(f"\n🤖 Agent: {answer}\n")
        return answer


    def clear_memory(self):
        self.memory.clear()
        print("🧹 Memory cleared!")


    def show_memory(self):
        messages = self.memory.load_memory_variables({})["chat_history"]

        if not messages:
            print("🧠 Memory is empty.")
            return

        print("\n🧠 Conversation history:")
        for msg in messages:
            role = "You" if msg.type == "human" else "Agent"
            print(f"  [{role}]: {msg.content}")


# ============================
# RUN  –  example usage
# ============================
if __name__ == "__main__":

    rag = AgenticRAG(
        pdf_files = [
            "./document1.pdf",
            "./document2.pdf",
        ],
        openai_api_key = "API_KEY",
        tavily_api_key = "API_KEY",
        verbose        = True,
    )

    print("💬 Type your question (or 'quit' to exit, 'memory' to see history)\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        elif user_input.lower() in ("quit", "exit"):
            print("👋 Goodbye!")
            break
        elif user_input.lower() == "memory":
            rag.show_memory()
        elif user_input.lower() == "clear":
            rag.clear_memory()
        else:
            rag.chat(user_input)