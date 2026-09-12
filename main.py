"""
Research Agent API — LangGraph + RAG (Pinecone) + ArXiv + SerpAPI
Wraps the working agent from the notebook behind a FastAPI endpoint.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --reload
Then POST to http://localhost:8000/ask with {"question": "..."}
"""

import os
import re
import logging
from typing import List, TypedDict, Annotated

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.getLogger("langsmith.client").setLevel(logging.ERROR)
os.environ["LANGCHAIN_TRACING_V2"] = "false"

# ── Load API keys from .env ──────────────────────────────────────────
load_dotenv()
# Expects OPENAI_API_KEY, PINECONE_API_KEY, SERPAPI_KEY in your .env file

# ── Embeddings + Pinecone (connect to EXISTING index, don't recreate) ─
from semantic_router.encoders import OpenAIEncoder
from pinecone import Pinecone

encoder = OpenAIEncoder(name="text-embedding-3-small")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "langgraph-research-agent"
index = pc.Index(index_name)  # assumes this index already exists & is populated

# ── Tools ─────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from serpapi import GoogleSearch

serpapi_params = {
    "engine": "google",
    "api_key": os.environ["SERPAPI_KEY"],
}

abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL,
)


@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str) -> str:
    """Fetches the abstract from an ArXiv paper given its ArXiv ID."""
    res = requests.get(f"https://arxiv.org/abs/{arxiv_id}")
    re_match = abstract_pattern.search(res.text)
    return re_match.group(1) if re_match else "Abstract not found."


@tool("web_search")
def web_search(query: str) -> str:
    """Finds general knowledge information using a Google search."""
    search = GoogleSearch({**serpapi_params, "q": query, "num": 5})
    results = search.get_dict().get("organic_results", [])
    formatted_results = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return formatted_results if results else "No results found."


@tool("image_search")
def image_search(query: str) -> str:
    """Finds a relevant image on the web illustrating a concept, when no diagram exists in the source paper."""
    search = GoogleSearch(
        {**serpapi_params, "engine": "google_images", "q": query, "num": 3}
    )
    results = search.get_dict().get("images_results", [])
    return "\n".join([f"{r.get('title')}: {r.get('original')}" for r in results[:3]])


def format_rag_contexts(matches: list) -> str:
    formatted_results = []
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Chunk: {x['metadata']['chunk']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
        )
        formatted_results.append(text)
    return "\n---\n".join(formatted_results)


@tool
def rag_search_filter(query: str, arxiv_id: str) -> str:
    """Finds information from the ArXiv database using a natural language query and a specific ArXiv ID."""
    xq = encoder([query])
    xc = index.query(
        vector=xq, top_k=6, include_metadata=True, filter={"arxiv_id": arxiv_id}
    )
    return format_rag_contexts(xc["matches"])


@tool("rag_search")
def rag_search(query: str) -> str:
    """Finds specialist information on AI using a natural language query."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    return format_rag_contexts(xc["matches"])


@tool
def final_answer(
    introduction: str,
    research_steps: List[str],
    main_body: str,
    conclusion: str,
    sources: str,
) -> str:
    """Returns a natural language response in the form of a research report."""
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])

    return (
        f"{introduction}\n\nResearch Steps:\n{research_steps}\n\n"
        f"Main Body:\n{main_body}\n\nConclusion:\n{conclusion}\n\nSources:\n{sources}"
    )


# ── Oracle (decision-making LLM) ───────────────────────────────────────
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import ToolCall
from langchain_core.agents import AgentAction
from langchain_openai import ChatOpenAI
import operator

system_prompt = """You are the oracle, the great AI decision-maker.
Given the user's query, you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a particular
query, do NOT use that same tool with the same query again. Also, do NOT use
any tool more than twice (i.e., if the tool appears in the scratchpad twice, do
not use it again).

You should aim to collect information from a diverse range of sources before
providing the answer to the user. Once you have collected plenty of information
to answer the user's question (stored in the scratchpad), use the final_answer tool."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ]
)

llm = ChatOpenAI(
    model="gpt-5-mini",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0,
)

tools = [rag_search_filter, rag_search, fetch_arxiv, web_search, image_search, final_answer]


def create_scratchpad(intermediate_steps: list[ToolCall]) -> str:
    research_steps = []
    for action in intermediate_steps:
        if action.log != "TBD":
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\nOutput: {action.log}"
            )
    return "\n---\n".join(research_steps)


oracle = (
    {
        "input": lambda x: x["input"],
        "chat_history": lambda x: x["chat_history"],
        "scratchpad": lambda x: create_scratchpad(intermediate_steps=x["intermediate_steps"]),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)

MAX_TURNS = 6


def run_oracle(state: dict) -> dict:
    if len(state["intermediate_steps"]) >= MAX_TURNS:
        out = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "scratchpad": lambda x: create_scratchpad(intermediate_steps=x["intermediate_steps"]),
            }
            | prompt
            | llm.bind_tools(tools, tool_choice="final_answer")
        ).invoke(state)
    else:
        out = oracle.invoke(state)

    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log="TBD")
    return {"intermediate_steps": [action_out]}


def router(state: dict) -> str:
    if isinstance(state["intermediate_steps"], list):
        return state["intermediate_steps"][-1].tool
    return "final_answer"


tool_str_to_func = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_arxiv": fetch_arxiv,
    "web_search": web_search,
    "image_search": image_search,
    "final_answer": final_answer,
}


def run_tool(state: dict) -> dict:
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input
    out = tool_str_to_func[tool_name].invoke(input=tool_args)
    action_out = AgentAction(tool=tool_name, tool_input=tool_args, log=str(out))
    return {"intermediate_steps": [action_out]}


class AgentState(TypedDict):
    input: str
    chat_history: List
    intermediate_steps: Annotated[List, operator.add]


# ── Graph ────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END

graph = StateGraph(AgentState)
graph.add_node("oracle", run_oracle)
graph.add_node("rag_search_filter", run_tool)
graph.add_node("rag_search", run_tool)
graph.add_node("fetch_arxiv", run_tool)
graph.add_node("web_search", run_tool)
graph.add_node("image_search", run_tool)
graph.add_node("final_answer", run_tool)

graph.set_entry_point("oracle")
graph.add_conditional_edges(source="oracle", path=router)

for tool_obj in tools:
    if tool_obj.name != "final_answer":
        graph.add_edge(tool_obj.name, "oracle")

graph.add_edge("final_answer", END)

runnable = graph.compile()


def build_report(output: dict) -> str:
    research_steps = output["research_steps"]
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])

    sources = output["sources"]
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])

    return f"""
INTRODUCTION
------------
{output['introduction']}

RESEARCH STEPS
--------------
{research_steps}

REPORT
------
{output['main_body']}

CONCLUSION
----------
{output['conclusion']}

SOURCES
-------
{sources}
"""


# ── FastAPI app ──────────────────────────────────────────────────────
app = FastAPI(title="Research Agent API")

# Allow your Lovable frontend (or any origin, for a demo) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your Lovable app's domain once deployed
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask(query: Query):
    output = runnable.invoke(
        {
            "input": query.question,
            "chat_history": [],
        }
    )
    report = build_report(output["intermediate_steps"][-1].tool_input)
    return {"report": report}
