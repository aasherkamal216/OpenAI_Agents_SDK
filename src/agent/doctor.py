from agents import (
    Agent,
    Runner,
    RunConfig,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    RunContextWrapper,
    function_tool,
    handoff,
    HandoffInputData
)
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from openai.types.responses import ResponseTextDeltaEvent
from agents.extensions import handoff_filters

import chainlit as cl
from tavily import TavilyClient


from prompts import DOCTOR_AGENT_PROMPT

from typing import cast
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load the environment variables
_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")


def agent_handoff_message_filter(handoff_message_data: HandoffInputData) -> HandoffInputData:
    # First, we'll remove any tool-related messages from the message history
    handoff_message_data = handoff_filters.remove_all_tools(handoff_message_data)

    # Second, we'll only pass the last two items from history
    history = (
        tuple(handoff_message_data.input_history[-2:])
        if isinstance(handoff_message_data.input_history, tuple)
        else handoff_message_data.input_history
    )

    return HandoffInputData(
        input_history=history,
        pre_handoff_items=tuple(handoff_message_data.pre_handoff_items),
        new_items=tuple(handoff_message_data.new_items),
    )

class SpecializedAgentData(BaseModel):
    agent_name: str
    reason: str

@cl.step(type="llm")
async def on_handoff(ctx: RunContextWrapper[None], input_data: SpecializedAgentData):
    agent_name = input_data.agent_name
    border = "+" + "-" * 60 + "+"

    print(border)
    print(f"|{f'Handing off to: {agent_name}':^60}|")
    print(f"|{f'Reason: {input_data.reason}':^60}|")
    print(border)

@cl.step(type="llm")
def handoff_func(agent: Agent, ctx: RunContextWrapper[None]):
    agent_name = agent.name
    border = "+" + "-" * 40 + "+"

    print(border)
    print(f"|{f'Handing off to: {agent_name}':^40}|")
    print(border)

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
        instructions=RECOMMENDED_PROMPT_PREFIX + DOCTOR_AGENT_PROMPT,
        name="Doctor AI",
        tools=[web_search],
        handoffs=[],
        handoff_description="A Chief Doctor Agent for general inquiries and diagnosis"
    )

    # Create Specialist Agents
    cardio_agent = Agent(
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}"
            "You are a Cardiology Specialist AI agent. "
            "If you are speaking to a patient, you probably were transferred to from the doctor AI Agent."
            "You provide expert advice on heart-related issues. "
            "Use the `web_search` tool to research medical information. "
            "ALWAYS handoff to `doctor_agent` if the user wants to talk. "
            "REMEMBER: Do NOT pretend to be Chief Doctor Agent"
        ),
        name="Cardiologist AI",
        tools=[web_search],
        handoffs=[handoff(doctor_agent, 
                          on_handoff=lambda ctx: handoff_func(doctor_agent, ctx))],
        handoff_description="A Cardiology Specialist Doctor"
    )

    derm_agent = Agent(
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}"
            "You are a Dermatology Specialist AI agent. "
            "If you are speaking to a patient, you probably were transferred to from the doctor AI Agent."
            "You provide expert advice on skin-related issues. "
            "Use the `web_search` tool to research medical information. "
            "ALWAYS handoff to `doctor_agent` if the user wants to talk. "
            "REMEMBER: Do NOT pretend to be Chief Doctor Agent"
        ),
        name="Dermatologist AI",
        tools=[web_search],
        handoffs=[handoff(doctor_agent, on_handoff=lambda ctx: handoff_func(doctor_agent, ctx))],
        handoff_description="A Dermatology Specialist Doctor"
    )

    neuro_agent = Agent(
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}"
            "You are a Neurology Specialist AI agent. "
            "If you are speaking to a patient, you probably were transferred to from the doctor AI Agent."
            "You provide expert advice on brain-related issues. "
            "Use the `web_search` tool to research medical information. "
            "ALWAYS handoff to `doctor_agent` if the user wants to talk. "
            "REMEMBER: Do NOT pretend to be Chief Doctor Agent"
        ),
        name="Neurologist AI",
        tools=[web_search],
        handoffs=[handoff(doctor_agent, on_handoff=lambda ctx: handoff_func(doctor_agent, ctx))],
        handoff_description="A Neurology Specialist Doctor"
    )

    # Define handoffs for the Main Doctor Agent
    specialist_agents = [cardio_agent, derm_agent, neuro_agent]
    doctor_agent.handoffs = [
        handoff(
            agent,
            on_handoff=on_handoff,
            input_type=SpecializedAgentData,
            input_filter=agent_handoff_message_filter,
        )
        for agent in specialist_agents
    ]

    # Set Agent and Config in session
    cl.user_session.set("config", config)
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
    current_agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    print("========", current_agent.name, "==========")
    # Stream agent's response
    result = Runner.run_streamed(
        starting_agent=current_agent,
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
    await msg.send()
    # Update chat history
    chat_history.append({"role": "assistant", "content": full_response})

    cl.user_session.set("history", chat_history)
    cl.user_session.set("agent", result.last_agent) # set the last agent to the current agent
