from lan_chain import *
import os
from dotenv import load_dotenv

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("openai_api_key")
os.environ["FIREWORKS_API_KEY"]=os.getenv("fireworks_api_key")
os.environ["COHERE_API_KEY"]=os.getenv("cohere_api_key")


def language_model_chain():
    st.title("Language Model Chain - Multi_LLM")
    
    provider = st.selectbox("Select provider", ["openai", "cohere", "fireworks"])

    if provider == "openai":
        available_models_openai = ["gpt-3.5-turbo", "model2", "model3"]
        model_name = st.selectbox("Select OpenAI model", available_models_openai)

    if provider == "cohere":
        available_models_cohere = ["model1", "command-r", "model3"] 
        model_name = st.selectbox("Select Cohere model", available_models_cohere)

    if provider == "fireworks":
        available_models_mixtral = ["model1", "model2", "accounts/fireworks/models/mixtral-8x7b-instruct"] 
        model_name = st.selectbox("Select Fireworks model", available_models_mixtral)

    top_p = st.slider("Top P", min_value=0.1, max_value=1.0, step=0.1)
    temp = st.slider("Temperature", min_value=0.1, max_value=2.0, step=0.1)
    max_tokens = st.slider("Max Tokens", min_value=10, max_value=100, step=10)
    question = st.text_input("Chat with me.")

    submit_button = st.button("Generate Output")

    if submit_button:
        output = chain(provider=provider, question=question, model_name=model_name, top_p=top_p, temp=temp, max_tokens=max_tokens)
        st.write(output)
    
language_model_chain()