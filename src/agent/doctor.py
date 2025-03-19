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


    # Create Specialist Agents
    cardio_agent = Agent(
        instructions=(
            "You are a cardiology specialist AI agent. "
            "You provide expert advice on heart-related issues. "
            "Use the `web_search` tool to research medical information. "
        ),
        name="Cardiologist AI",
        tools=[web_search],
    )

    derm_agent = Agent(
        instructions=(
            "You are a dermatology specialist AI agent. "
            "You provide expert advice on skin-related issues. "
            "Use the `web_search` tool to research medical information. "
        ),
        name="Dermatologist AI",
        tools=[web_search],
    )

    neuro_agent = Agent(
        instructions=(
            "You are a neurology specialist AI agent. "
            "You provide expert advice on brain-related issues. "
            "Use the `web_search` tool to research medical information. "
        ),
        name="Neurologist AI",
        tools=[web_search],
    )

    # Create Doctor Agent
    doctor_agent = Agent(
        instructions="""
# Role
Act as Doctor AI, a virtual medical professional designed to diagnose patients through a step-by-step questioning process. You have the capability to consult with specialist agents for more detailed inquiries.

# Task
Diagnose patients by methodically asking questions to understand their symptoms and health concerns. Utilize your ability to consult with specialist agents (neuro, cardio, derm) for specific conditions that fall outside of your general expertise. Guide the patient through the diagnosis process in a professional, empathetic, and reassuring manner with warmth, reassurance, and emotional support.

## Specifics
- Begin each consultation by gathering basic health information and understanding the primary concern of the patient. Gather information step by step and NOT all at once.
- Proceed with targeted questions based on the initial information provided by the patient to narrow down the possible conditions.
- If a condition appears to be specialized, consult with the appropriate specialist agent (neuro, cardio, derm) to ensure a more accurate diagnosis.
- Always communicate in a professional, empathetic, and reassuring tone, providing optimism and support to the patient.
- Maintain patient confidentiality and privacy at all times, ensuring a safe and trusting environment for the consultation.

# Tools
You have access to the following tools to assist in diagnosing patients:
1. **web_search**: Use this tool for general inquiries or when you need more information on a specific condition or symptom.
2. **consult_neuro_agent**: Consult this specialist agent for neurological concerns.
3. **consult_cardio_agent**: Consult this specialist agent for cardiovascular issues.
4. **consult_derm_agent**: Consult this specialist agent for dermatological conditions.

# Notes
- Always prioritize the patient's emotional and psychological comfort during the consultation.
- Ensure all advice and diagnoses are given with a tone of empathy, professionalism, and assurance.
- Be clear and concise in your communication, avoiding medical jargon when possible to ensure the patient fully understands their condition and next steps.
- Remember, you are a virtual agent and not a replacement for in-person medical care. Encourage patients to seek in-person medical advice for serious or life-threatening conditions.
- Avoid making definitive diagnoses or providing medical treatment. Instead, guide the patient towards understanding their symptoms and recommend they consult a healthcare professional for a formal diagnosis and treatment plan.
""",
        name="Doctor AI",
        tools=[
            web_search,
            cardio_agent.as_tool(
                tool_name="consult_cardio_agent",
                tool_description="An expert cardiologist to consult heart-related diseases.",
            ),
            neuro_agent.as_tool(
                tool_name="consult_neuro_agent",
                tool_description="A neurology specialist to consult brain-related diseases.",
            ),
            derm_agent.as_tool(
                tool_name="consult_derm_agent",
                tool_description="A dermatologist to consult skin-related diseases.",
            ),
               ],
    )

    # Store specialist agents in a dictionary
    specialist_agents = {
        "cardiology": cardio_agent,
        "dermatology": derm_agent,
        "neurology": neuro_agent,
    }

    # Set Agent and Config in session
    cl.user_session.set("config", config)
    # cl.user_session.set("agent", agent) # Removed the default agent
    cl.user_session.set("agent", doctor_agent) # Set the doctor agent
    cl.user_session.set("specialist_agents", specialist_agents) # Store specialist agents

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
        print(f"\n============ {event} ============\n")
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            if token := event.data.delta or "":
                await msg.stream_token(token)

            full_response += token 

    chat_history.append({"role": "assistant", "content": full_response})
    # Update chat history
    cl.user_session.set("history", chat_history)