import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from metrics.prometheus_metrics import get_metrics


load_dotenv()

SYSTEM_PROMPT = (
    """Eres un agente especializado en saludos e interacciones iniciales.

    Tu misión es responder de forma amable, clara y apropiada a preguntas relacionadas con saludos, presentaciones o conversaciones iniciales.

    Responde brevemente y de manera natural.

    Si no entiendes la pregunta, pide aclaración.

    """
)

app = FastAPI()

DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # only openai supported
DEFAULT_MODEL = os.getenv("MODEL_NAME", "gpt-4o-mini")          # por proveedor
DEFAULT_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))

# (Opcional) URLs/keys por proveedor
# Only OpenAI is supported now

class GreetingAgentRequest(BaseModel):
    text: str
    session_id: Optional[str] = None
    context_summary: Optional[str] = None
    provider: Optional[str] = DEFAULT_PROVIDER
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = DEFAULT_TEMPERATURE

class GreetingAgentResponse(BaseModel):
    result: str



# =========================
# Fábrica de LLMs
# =========================
def make_llm(
    provider: str,
    model: str,
    temperature: float
):
    provider = (provider or DEFAULT_PROVIDER).lower()
    if provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature)

    raise ValueError(f"Proveedor LLM no soportado: {provider}. Solo 'openai' está soportado ahora.")



@app.get("/metrics")
def metrics():
    return get_metrics()

@app.post("/greeting_agent", response_model=GreetingAgentResponse)
def greeting_agent_endpoint(req: GreetingAgentRequest):
    print(f"[API] Greeting request: '{req.text}' (session_id: {req.session_id})")
    if req.context_summary:
        print(f"[API] Context summary received: {req.context_summary}")

    llm = make_llm(req.provider, req.model, req.temperature)
    tools = []

    system_prompt = SYSTEM_PROMPT
    if req.context_summary:
        system_prompt = SYSTEM_PROMPT + "\n\nCONTEXT SUMMARY: " + req.context_summary

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("ai", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    # Ejecuta el agente de forma completamente automática
    result = executor.invoke({"input": req.text, "session_id": req.session_id, "context_summary": req.context_summary})
    # El resultado puede estar en diferentes campos según el modelo
    if isinstance(result, dict) and "output" in result:
        return GreetingAgentResponse(result=str(result["output"]))
    return GreetingAgentResponse(result=str(result))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
