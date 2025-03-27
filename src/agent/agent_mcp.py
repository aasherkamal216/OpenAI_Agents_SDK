from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    OpenAIResponsesModel,
    AsyncOpenAI,
    function_tool,
)
from openai.types.responses import ResponseTextDeltaEvent

import chainlit as cl
from mcp import ClientSession

from typing import cast
from dotenv import load_dotenv
import os, datetime

# Load the environment variables
_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

    
@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    # List available tools
    result = await session.list_tools()
    print(result)
    # Store tools for later use
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = [t for t in result.tools]
    cl.user_session.set("mcp_tools", mcp_tools)


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
            "You answer user queries related to documentations in a professional style. Use the available tools."
        ),
        name="Tylon",
    )
    # Set Agent and Config in session
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent)

    # Initialize the chat history in the user session
    cl.user_session.set("history", [])


@cl.on_message
async def main(message: cl.Message):
    # Get the chat history from the user session
    chat_history = cl.user_session.get("chat_history")

    # Get Config and Agent from user session
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    mcp_tools = cl.user_session.get("mcp_tools")
    tools = next(iter(mcp_tools.values())) if mcp_tools else None
    print(tools)

    agent.tools = [t for t in tools]
    # Run the agent with the user's message
    result = await Runner.run(agent, message.content, run_config=config)
    response_text = result.final_output

    # Update the chat history (append the user message and the agent's response)
    chat_history.append({"user": message.content, "agent": response_text})
    cl.user_session.set("chat_history", chat_history)

    # Send the agent's response to the user
    await cl.Message(content=response_text).send()
