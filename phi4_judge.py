import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import os
    import pandas as pd
    import json
    import rich
    from rich.pretty import pprint
    import numpy as np
    from typing import Optional, Literal, List, Dict, Any, Callable, Union
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from pydantic import BaseModel, Field
    from tqdm.auto import tqdm
    from openai import OpenAIError
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from os import getenv
    from langchain.chains import LLMChain
    from dotenv import load_dotenv



@app.cell
def _():

    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])


    llm = ChatOpenAI(
      openai_api_key = os.environ["judge_key"],
      openai_api_base = "https://openrouter.ai/api/v1",
      model_name = "qwen/qwen3-1.7b:free",
    )
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    print(llm_chain.run(question))

    return


if __name__ == "__main__":
    app.run()
