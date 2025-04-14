import os
from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient, GrazieAgent
from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.profiles import Profile
from grazie.api.client.endpoints import GrazieApiGatewayUrls

token = os.getenv("AI_TOKEN")

client = GrazieApiGatewayClient(
    url = GrazieApiGatewayUrls.PRODUCTION,
    grazie_jwt_token = token, # Provide the authentication token
    auth_type = AuthType.USER, # Set the user authentication type
    grazie_agent = GrazieAgent(name="grazie-api-gateway-client-heval-test", version="dev") # Define the agent name and version
)

SYSTEM_PROMPT = """
You are a helpful assistant.
""" # TODO: Change into an actual prompt

def makeCall(task, sys_prompt=SYSTEM_PROMPT):
    chat = (
                ChatPrompt()
                .add_system(sys_prompt)
                .add_user(task)
            )
    response = client.chat(
            chat = chat,
            profile = Profile.OPENAI_GPT_4
            )
    return response.content

if __name__ == "__main__":
    print(makeCall("Write a python function to get all prime numbers up to 20"))
