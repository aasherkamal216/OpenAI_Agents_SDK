from agents import Agent, Runner, RunConfig, OpenAIChatCompletionsModel, AsyncOpenAI
import chainlit as cl

from dotenv import load_dotenv
import os

# Load the environment variables
_: bool = load_dotenv()

gemini_api_key = os.getenv("GOOGLE_API_KEY")

# Initialize the AsyncOpenAI client
client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Create the model instance
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

# Configure the run settings for the agent
config = RunConfig(
    model=model,
    model_provider=client,
    tracing_disabled=True  # Disable tracing for this run
)

# Create Agent
agent = Agent(
    instructions="You are Tylon, a helpful AI Agent.",
    name="Tylon"
)

# Handle Chat History
@cl.on_chat_start
async def handle_chat_history():
    # Initialize the chat history in the user session
    cl.user_session.set("history", [])
    # Send a welcome message to the user
    await cl.Message(content="Hi there! I'm Tylon, your AI Agent.").send()

@cl.on_message
async def handle_message(msg: cl.Message):
    # Retrieve the chat history from the user session
    chat_history = cl.user_session.get("history")

    # Append the user's message to the chat history
    chat_history.append({"role": "user", "content": msg.content})

    # Run the agent
    result = await Runner.run(
        starting_agent=agent,
        input=chat_history,
        run_config=config
    )
    
    chat_history.append({"role": "assistant", "content": result.final_output})
    # Update chat history
    cl.user_session.set("history", chat_history)

    # Display Agent's response
    await cl.Message(content=result.final_output).send()