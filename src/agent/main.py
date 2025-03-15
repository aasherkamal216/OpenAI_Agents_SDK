
from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
import chainlit as cl

from dotenv import load_dotenv
import os

_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True
)

agent = Agent(
    instructions="You are Tylon, a helpful AI Agent.",
    name="Tylon"
)

@cl.on_chat_start
async def handle_chat_history():
    cl.user_session.set("history", [])
    await cl.Message(content="Hi there! I'm Tylon, your AI Agent.").send()

@cl.on_message
async def handle_message(msg: cl.Message):

    chat_history = cl.user_session.get("history")

    chat_history.append({"role": "user", "content": msg.content})

    result = await Runner.run(
        starting_agent=agent,
        input=chat_history,
        run_config=config
    )
    chat_history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", chat_history)

    await cl.Message(content=result.final_output).send()