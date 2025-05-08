import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it to use the OpenAI API.")


    from langchain.chat_models import init_chat_model

    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    return (model,)


@app.cell
def _(json):
    import pandas as pd

    # Read the JSONL file
    with open('filtered_output.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
        print(data)
    return (data,)


@app.cell
def _(example, pprint):
    pprint(example)
    return


@app.cell
def _(data):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

    prompt_template = ChatPromptTemplate.from_template("""
    You will receive an input in JSON format, containing:
    A context describing a situation.
    A question related to the context.
    Several possible answers (e.g., "ans0," "ans1," "ans2," etc.).
    Your task:
    Interpret the context carefully, noting any relevant details.
    Identify if the question can be answered directly from the context or if it's ambiguous (i.e., "Not
    enough information").
    Critically assess each provided answer choice.
    Provide your final answer by selecting the best matching answer choice.
    Output your your final answer in the format described below.
    Response Format:
    Final Answer: Output exactly one of the provided answers (e.g., "ans0," "ans1," "ans2," etc.) that
    you deem correct.

    Context: {context}
    Question: {question}
    Answer choices:
    - ans0: {ans0}
    - ans1: {ans1}
    - ans2: {ans2}
    """)


    example = data[0]
    formatted_prompt = prompt_template.format_messages(
        context=example["context"],
        question=example["question"],
        ans0=example["ans0"],
        ans1=example["ans1"],
        ans2=example["ans2"]
    )

    # Display the formatted prompt
    formatted_prompt
    return example, formatted_prompt


@app.cell
def _(formatted_prompt, model):
    from typing import Optional, Literal
    from pydantic import BaseModel, Field

    class FinalAnswer(BaseModel):
        """Answer of the question"""
    
        answer: Literal["ans0", "ans1", "ans2"] = Field(description="Answer of the question among ['ans0', 'ans1', 'ans2']")


    structured_llm = model.with_structured_output(FinalAnswer)
    response = structured_llm.invoke(formatted_prompt)

    # Example usage
    response.answer

    return


@app.cell(disabled=True, hide_code=True)
def _():
    # from typing import Optional

    # from pydantic import BaseModel, Field


    # # Pydantic
    # class Joke(BaseModel):
    #     """Joke to tell user."""

    #     setup: str = Field(description="The setup of the joke")
    #     punchline: str = Field(description="The punchline to the joke")
    #     rating: Optional[int] = Field(
    #         default=None, description="How funny the joke is, from 1 to 10"
    #     )


    # structured_llm = model.with_structured_output(Joke)

    # structured_llm.invoke("Tell me a joke about cats")
    return


@app.cell
def _():
    import marimo as mo
    import json
    return (json,)


@app.cell
def _():
    import rich
    from rich.pretty import pprint
    return (pprint,)


if __name__ == "__main__":
    app.run()
