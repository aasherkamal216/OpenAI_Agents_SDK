from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunContextWrapper,
    function_tool,
    handoff
)
from openai.types.responses import ResponseTextDeltaEvent

import chainlit as cl
from tavily import TavilyClient

from agent.prompts import DOCTOR_AGENT_PROMPT

from typing import cast
from dotenv import load_dotenv
import os

# Load the environment variables
_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

def handoff_func(agent: Agent, ctx: RunContextWrapper[None]):
    agent_name = agent.name
    print("=" * 40)
    print(f"\n>> Handing off to: {agent_name} <<\n")
    print("=" * 40)
    
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

    # Create Doctor Agent
    doctor_agent = Agent(
        instructions=DOCTOR_AGENT_PROMPT,
        name="Doctor AI",
        tools=[web_search],
        handoffs=[],
        handoff_description="A chief specialist doctor"
    )

    # Create Specialist Agents
    cardio_agent = Agent(
        instructions=(
            "You are a cardiology specialist AI agent. "
            "You provide expert advice on heart-related issues. "
            "Use the `web_search` tool to research medical information. "
            "Handoff to `doctor_agent` if the user wants to talk."
        ),
        name="Cardiologist AI",
        tools=[web_search],
        handoffs=[doctor_agent],
        handoff_description="A cardiology specialist doctor"
    )

    derm_agent = Agent(
        instructions=(
            "You are a dermatology specialist AI agent. "
            "You provide expert advice on skin-related issues. "
            "Use the `web_search` tool to research medical information. "
            "Handoff to `doctor_agent` if the user wants to talk."
        ),
        name="Dermatologist AI",
        tools=[web_search],
        handoffs=[doctor_agent],
        handoff_description="A dermatology specialist doctor"
    )

    neuro_agent = Agent(
        instructions=(
            "You are a neurology specialist AI agent. "
            "You provide expert advice on brain-related issues. "
            "Use the `web_search` tool to research medical information. "
            "Handoff to `doctor_agent` if the user wants to talk."
        ),
        name="Neurologist AI",
        tools=[web_search],
        handoffs=[doctor_agent],
        handoff_description="A neurology specialist doctor"
    )

    doctor_agent.handoffs = [
        handoff(cardio_agent, on_handoff=lambda ctx: handoff_func(cardio_agent, ctx)),
        handoff(derm_agent, on_handoff=lambda ctx: handoff_func(derm_agent, ctx)),
        handoff(neuro_agent, on_handoff=lambda ctx: handoff_func(neuro_agent, ctx))
        ]

    # Set Agent and Config in session
    cl.user_session.set("config", config)
    # cl.user_session.set("agent", agent) # Removed the default agent
    cl.user_session.set("agent", doctor_agent) # Set the doctor agent

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