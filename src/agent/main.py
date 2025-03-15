
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
    instructions="You are a helpful assistant.",
    name="Tylon"
)

# response = Runner.run_sync(
#     starting_agent=agent,
#     input="What is an AI Agent??",
#     run_config=config
# )

# print(response.final_output)

@cl.on_message
async def handle_message(msg: cl.Message):
    result = await Runner.run(
        starting_agent=agent,
        input=msg.content,
        run_config=config
    )
    await cl.Message(content = result.final_output).send()