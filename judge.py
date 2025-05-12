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
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from os import getenv
    from langchain.chains import LLMChain
    from dotenv import load_dotenv
    from prompts import format_qwen_judge_prompt, format_prompt_no_cot
    from answer_questions import save_checkpoint, load_checkpoint
    from langchain_deepseek import ChatDeepSeek


    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)


@app.cell
def _():
    # if not os.environ.get("judge_key"):
    #     raise ValueError("judge_key environment variable is not set. Please set it to use the DeepSeek API.")

    # model = ChatOpenAI(
    #     api_key = os.environ["Judge_key"],
    #     base_url = "https://openrouter.ai/api/v1",
    #     model_name = "qwen/qwen3-1.7b:free",
    # )

    model = ChatOpenAI(
        openai_api_key=os.environ["judge_key"],
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="meta-llama/llama-3-8b-instruct",
    )
    return (model,)


@app.cell
def _():
    # Read the JSONL file
    with open('reasoning_steps.jsonl', 'r') as file:
        data = [json.loads(line) for line in file]
        print(data)

    bbq_df = pd.DataFrame(data)
    bbq_df
    return (bbq_df,)


@app.function
def answer_multiple_choice_with_llm(
    llm: BaseChatModel,
    prompt_formatter: Callable[[pd.Series], str],
    desc: str,
    df: pd.DataFrame,
    max_concurrency: int = 10,
    checkpoint_file: Optional[str] = None
) -> List[str]:
    """Process multiple-choice questions using an LLM in batches with checkpointing.

    Args:
        llm: Language model instance
        prompt_formatter: Function to format prompts from question data
        desc: Description for progress bar
        df: DataFrame containing questions
        max_concurrency: Maximum number of concurrent requests
        checkpoint_file: Optional path to checkpoint file

    Returns:
        List of answer strings
    """
    # Try to load checkpoint if available
    if checkpoint_file:
        checkpoint_data = load_checkpoint(checkpoint_file)
        if checkpoint_data is not None:
            results = checkpoint_data['answers']
            last_processed_idx = checkpoint_data['last_processed_idx']
            rich.print(f"[yellow]Continuing from checkpoint:[/yellow] Processed {last_processed_idx + 1} questions")
            rich.print(f"[yellow]Remaining questions:[/yellow] {len(df) - (last_processed_idx + 1)}")
            # Start from the next unanswered question
            df = df.iloc[last_processed_idx + 1:]
        else:
            results = []
            rich.print("[green]No checkpoint found. Starting from beginning.[/green]")
    else:
        results = []
        rich.print("[green]No checkpoint file specified. Starting from beginning.[/green]")

    try:
         # Process dataframe in chunks with progress bar
         for i in tqdm(range(0, len(df), max_concurrency), desc=desc):
             chunk = df.iloc[i:i+max_concurrency]
             # Create prompts for this chunk
             chunk_prompts = [prompt_formatter(bias_question_data) for _, bias_question_data in chunk.iterrows()]
             save_checkpoint = chunk_prompts
             # Process this chunk

             config = RunnableConfig(max_concurrency=max_concurrency)
             chunk_responses = llm.batch(chunk_prompts, config=config)
             
             # Extract reasoning from responses
             chunk_answers = [response.content for response in chunk_responses]
             results.extend(chunk_answers)

             # Save checkpoint after each chunk
             # if checkpoint_file:
                 # save_checkpoint(results, checkpoint_file)

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        if checkpoint_file:
            rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
        raise e

    return results


@app.cell
def _(bbq_df, model):
    _judge_checkpoint_file = os.path.join("checkpoints", "judge_checkpoint.json")
    j = answer_multiple_choice_with_llm(
        model, 
        format_qwen_judge_prompt, 
        "Generating Judge", 
        bbq_df,
        max_concurrency=10,
        checkpoint_file=_judge_checkpoint_file
    )

    return (j,)


@app.cell
def _(bbq_df, j):
    bbq_df["judge"] = j
    return


@app.cell
def _(bbq_df):
    bbq_df.head()
    return


if __name__ == "__main__":
    app.run()
