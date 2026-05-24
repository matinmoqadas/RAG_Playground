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


OPENAI_API_KEY  = "API_KEY"
TAVILY_API_KEY  = "API_KEY"

PDF_FILES = [
    "./doc1.pdf",
    "./doc2.pdf",
]

os.environ["OPENAI_API_KEY"]  = OPENAI_API_KEY
os.environ["TAVILY_API_KEY"]  = TAVILY_API_KEY


llm = ChatOpenAI(
    model="gp5-nano",
    temperature=0,
    base_url= ""
)


def load_pdfs(pdf_paths: list[str]) -> list:
    all_pages = []

    for pdf_path in pdf_paths:

        print(f"📄 Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages  = loader.load()     
        all_pages.extend(pages)
        print(f"   → loaded {len(pages)} pages")

    print(f"\n✅ Total pages loaded: {len(all_pages)}")
    return all_pages


def get_chunk_size(text: str) -> int:

    length = len(text)

    if length < 2_000:
        return 300
    elif length < 10_000:
        return 600
    else:
        return 1_000


def adaptive_split(pages: list) -> list:

    all_chunks = []

    for page in pages:
        page_text  = page.page_content
        chunk_size = get_chunk_size(page_text)   

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15), 
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents([page])
        all_chunks.extend(chunks)

    print(f"✅ Adaptive chunking done → {len(all_chunks)} chunks total")
    return all_chunks


def build_vector_store(chunks: list) -> FAISS:

    embeddings    = OpenAIEmbeddings()
    vector_store  = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store built!")
    return vector_store


def create_tools(vector_store: FAISS) -> list:

    # --- Tool 1: PDF Retriever ---
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4} 
    )

    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="search_pdf_documents",
        description=(
            "Search through the uploaded PDF documents. "
            "Use this when the user asks about topics covered in the PDFs. "
            "Always try this tool first before searching the web."
        ),
    )

    # --- Tool 2: Tavily Web Search ---
    web_search_tool = TavilySearch(
        max_results=3,
        name="web_search",
        description=(
            "Search the internet for current information. "
            "Use this when the PDFs don't have enough information, "
            "or when the user asks about recent events or news."
        ),
    )

    tools = [retriever_tool, web_search_tool]
    print(f"✅ Tools ready: {[t.name for t in tools]}")
    return tools


# Memory
memory = ConversationBufferMemory(
    memory_key="chat_history",    
    return_messages=True,         
)


# Agent
def build_agent(tools: list) -> AgentExecutor:

    # Agent prompt
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful research assistant. \n"
            "You have two tools:\n"
            "  1. search_pdf_documents – search uploaded PDF files\n"
            "  2. web_search – search the internet\n\n"
            "Rules:\n"
            "  - Always try the PDF search tool first.\n"
            "  - Use web search only if PDFs don't have the answer.\n"
            "  - If you used both tools, mention where each piece of info came from.\n"
            "  - Keep answers clear and simple."
        ),
        MessagesPlaceholder(variable_name="chat_history"),  
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,        
        verbose=False,        # agents thinking
        max_iterations=6,
    )

    print("✅ Agent is ready!")
    return agent_executor

def chat(agent_executor: AgentExecutor, user_message: str) -> str:

    print(f"\n{'='*55}")
    print(f"You: {user_message}")
    print(f"{'='*55}")

    result = agent_executor.invoke({"input": user_message})
    answer = result["output"]

    print(f"\n🤖 Agent: {answer}")
    return answer


if __name__ == "__main__":

    pages = load_pdfs(PDF_FILES)

    if not pages:
        print("❌ No pages loaded.")
        exit()

    chunks = adaptive_split(pages)

    vector_store = build_vector_store(chunks)

    tools = create_tools(vector_store)

    agent_executor = build_agent(tools)

    print("\n💬 Start chatting! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("👋 Goodbye!")
            break

        chat(agent_executor, user_input)
