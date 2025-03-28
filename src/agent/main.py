from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
)
from openai.types.responses import ResponseTextDeltaEvent

import chainlit as cl
from tavily import TavilyClient

from typing import cast
from dotenv import load_dotenv
import os, datetime

# Load the environment variables
_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")


@function_tool
@cl.step(type="tool")
def web_search(query: str):
    """
    A tool to search the web using natural language query.
    """
    # Initialize Tavily Client
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # Search the web
    response = tavily_client.search(query)

    return response

@cl.on_chat_start
async def start():
    # Initialize the AsyncOpenAI client
    client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    # Create the model instance
    model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)

    # Configure the run settings for the agent
    config = RunConfig(
        model=model,
        model_provider=client,
        tracing_disabled=True, 
    )

    # Create Agent
    agent = Agent(
        instructions=(
            "You are a helpful AI Agent. "
            "You answer user queries in a professional style. "
            "Use `web_search` tool for real-time web search. "
            f"Today's date: {datetime.datetime.now().strftime('%B %d, %Y')}."
        ),
        name="Tylon",
        tools=[web_search],
    )
    # Set Agent and Config in session
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent)

    # Initialize the chat history in the user session
    cl.user_session.set("history", [])


@cl.on_message
async def handle_messages(message: cl.Message):

    # Retrieve the chat history from the user session
    chat_history = cl.user_session.get("history")

    # Append the user's message to the chat history
    chat_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    # Get Config and Agent from user session
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Stream agent's response
    result = Runner.run_streamed(
        starting_agent=agent,
        input=chat_history,
        run_config=config
    )

    full_response = "" 

    # Stream the message in app
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if token := event.data.delta or "":
                await msg.stream_token(token)

            full_response += token 

    chat_history.append({"role": "assistant", "content": full_response})
    # Update chat history
    cl.user_session.set("history", chat_history)