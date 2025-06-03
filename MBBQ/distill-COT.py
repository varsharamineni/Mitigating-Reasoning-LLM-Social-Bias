import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
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
    from tqdm.auto import tqdm
    from answer_questions import save_checkpoint, load_checkpoint
    import re
    from openai import OpenAI
    from openai import AsyncOpenAI
    import asyncio
    from prompts import format_prompt_cot

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset configuration
    DATASET_NAME = "Gender_identity_control_tr"  # This will determine the language to use
    LANGUAGE = "tr"


@app.cell
def _():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set. Please set it to use the DeepSeek API.")

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    return (client,)


@app.cell
def _():
    # Read the JSONL file
    with open(os.path.join('datasets', 'our_datasets', f'{DATASET_NAME}.jsonl'), 'r') as file:
        data = [json.loads(line) for line in file]
        print(f"Loaded {len(data)} questions from dataset")

    bbq_df = pd.DataFrame(data)
    print("\nFirst row of the dataset:")
    print(bbq_df.iloc[0])
    return (bbq_df,)


@app.cell
def _():
    # def format_prompt_cot(question_data: pd.Series) -> str:
    #     """Format a prompt for chain-of-thought reasoning.

    #     Args:
    #         question_data (pd.Series): Series containing the question data with keys:
    #             - context (str): The context of the question
    #             - question (str): The question to be answered
    #             - ans0 (str): First answer choice
    #             - ans1 (str): Second answer choice
    #             - ans2 (str): Third answer choice

    #     Returns:
    #         str: Formatted prompt string
    #     """
    #     prompt = f"""Context: {question_data['context']}
    # Question: {question_data['question']}
    # Answer choices:
    # - ans0: {question_data['ans0']}
    # - ans1: {question_data['ans1']}
    # - ans2: {question_data['ans2']}
    # Based on the context, question and choices. Think through the problem."""
    #     return prompt
    return


@app.function
async def process_chunk(client: AsyncOpenAI, prompts: List[str]) -> List[str]:
    """Process a chunk of prompts concurrently."""
    async def process_single_prompt(prompt: str) -> str:
        try:
            response = await client.chat.completions.create(
                model="deepseek/deepseek-r1-distill-llama-8b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            # Return the raw response text
            return response.choices[0].message.reasoning
        except Exception as e:
            print(f"Error processing prompt: {str(e)}")
            return f"Error: {str(e)}"

    # Create tasks for all prompts in the chunk
    tasks = [process_single_prompt(prompt) for prompt in prompts]

    # Process all tasks concurrently and wait for all to complete
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions that occurred during processing
    results = []
    for response in responses:
        if isinstance(response, Exception):
            results.append(f"Error: {str(response)}")
        else:
            results.append(response)

    return results


@app.function
async def answer_multiple_choice_with_llm(
    client: AsyncOpenAI,
    prompt_formatter: Callable[[pd.Series], str],
    desc: str,
    df: pd.DataFrame,
    max_concurrency: int = 5,  # Increased default concurrency
    checkpoint_file: Optional[str] = None
) -> List[List[str]]:
    """Process multiple-choice questions using an LLM in batches with checkpointing."""
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

            # Process this chunk concurrently
            chunk_responses = await process_chunk(client, chunk_prompts)
            results.extend(chunk_responses)

            # Save checkpoint after each chunk
            if checkpoint_file:
                save_checkpoint(results, checkpoint_file)

            # Add a small delay between chunks to avoid rate limiting
            await asyncio.sleep(0.1)

    except Exception as e:
        rich.print(f"[red]Error occurred:[/red] {str(e)}")
        if checkpoint_file:
            rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
        raise e

    return results


@app.function
def process_cot(cot_text: str) -> List[str]:
    """Process chain-of-thought text into a list of sentences.

    Args:
        cot_text (str): The chain-of-thought text to process

    Returns:
        List[str]: List of sentences from the chain-of-thought
    """
    # Split by newlines and filter out empty lines
    sentences = [s.strip() for s in cot_text.split('\n') if s.strip()]
    return sentences


@app.cell
async def _(bbq_df, client):
    # Define checkpoint file path
    checkpoint_file = os.path.join("checkpoints", f"distill_cot_{DATASET_NAME}_checkpoint.json")

    # Create a wrapper function that includes the language parameter
    def format_prompt_with_language(question_data: pd.Series) -> str:
        return format_prompt_cot(question_data.to_dict(), language=LANGUAGE)

    # Run the function directly since we're in an async cell
    results = await answer_multiple_choice_with_llm(
        client,
        format_prompt_with_language,  # Using the wrapper function
        "Generating chain-of-thought",
        bbq_df,
        max_concurrency=20,
        checkpoint_file=checkpoint_file
    )

    # Save results to a file
    os.makedirs("COT", exist_ok=True)
    output_file = os.path.join("COT", f"distill_cot_{DATASET_NAME}.jsonl")

    # Save as JSONL with all original fields plus the chain-of-thought
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, row in bbq_df.iterrows():
            # Create a dictionary with all original fields
            output_dict = row.to_dict()
            # Add the chain-of-thought as a list of sentences
            output_dict['cot'] = process_cot(results[idx])
            # Write as a single line of JSON
            f.write(json.dumps(output_dict) + '\n')

    print(f"\nSaved results to {output_file}")
    return


if __name__ == "__main__":
    app.run()
