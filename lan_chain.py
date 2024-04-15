
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_fireworks import ChatFireworks
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st


def multimodel(model_name: str, top_p: float, temp: float, max_tokens: int):

    """
    Attribute configurable_alternatives from langchain gives the flexibility to add n number of models
    which can be called based on a keyword to make that particular model function.
    """

    kwargs = {"top_p":top_p}
    model = ChatOpenAI(model_name=model_name,
                       model_kwargs=kwargs,
                       temperature=temp,
                       max_tokens=max_tokens
                      ).configurable_alternatives(ConfigurableField(id="provider"),
                                                 default_key="openai",
                                                 cohere=ChatCohere(model_name=model_name,
                                                                   model_kwargs=kwargs,
                                                                   temperature=temp,
                                                                   max_tokens=max_tokens),
                                                 fireworks=ChatFireworks(model_name=model_name,
                                                                         model_kwargs=kwargs,
                                                                         temperature=temp,
                                                                         max_tokens=max_tokens))
    return model

def chain(provider:str, question, **kwargs):
    
    """
    Chaining together a prompt and a chatbot model based on the selected provider.
    """

    prompt=ChatPromptTemplate.from_messages(
                [
                    ("system","You are a helpful assistant. Your name is Yuva. Please respond to the user."),
                    ("user","{question}")
                ])
    try:
        if provider=="openai":
            model = multimodel(**kwargs)
            chain = prompt | model | StrOutputParser()
        elif provider == "cohere":
            model = multimodel(**kwargs)
            chain = prompt | model.with_config(configurable={"provider": "cohere"}) | StrOutputParser()
        elif provider == "fireworks":
            model = multimodel(**kwargs)
            chain = prompt | model.with_config(configurable={"provider": "fireworks"}) | StrOutputParser()
        else:
            raise ValueError("Invalid provider specified.")
    except Exception as e:
        return f"Error: {e}"
    return chain.invoke({"question":question})
