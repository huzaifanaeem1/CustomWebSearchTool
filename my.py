import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings
from tavily import AsyncTavilyClient

load_dotenv(find_dotenv())


gemini_api_key = os.environ.get("GEMINI_API_KEY")
tavily_api_key = os.environ.get("TAVILY_API_KEY")


tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

set_tracing_disabled(disabled=True)

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=provider
)

@function_tool()
async def search(query: str) -> str:
    print("[TOOL....] SEARCH QUERY", query)
    response = await tavily_client.search(query)
    return response


agent: Agent = Agent(
    name="Search Agent",
    instructions="Your are Deep Search Agent!",
    model=llm_model,
    tools=[search],
    model_settings=ModelSettings(
        tempreture=1.9, tool_choice="none"
    )
)

result = Runner.run_sync(agent, "What is tavily??")
print("\n \n DEEP SEARCH AGENT", result.final_output)