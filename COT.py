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
    from prompts import format_prompt_cot
    from typing import Optional, Literal, List, Dict, Any, Callable, Union
    from langchain.chat_models.base import BaseChatModel
    from langchain_core.runnables import RunnableConfig
    from tqdm.auto import tqdm
    from langchain_deepseek import ChatDeepSeek
    from answer_questions import save_checkpoint, load_checkpoint
    import re

    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset configuration
    DATASET_NAME = "Age"


@app.cell
def _():
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise ValueError("DEEPSEEK_API_KEY environment variable is not set. Please set it to use the DeepSeek API.")

    model = ChatDeepSeek(
        model_name="deepseek-reasoner",
        openai_api_key=os.environ["DEEPSEEK_API_KEY"],
        openai_api_base=os.environ["API_URL_DEEPSEEK"]
    )
    return (model,)


@app.cell
def _():
    # Read the JSONL file
    with open(os.path.join('dataset', 'BBQ', f'{DATASET_NAME}.jsonl'), 'r') as file:
        data = [json.loads(line) for line in file]
        print(data)

    bbq_df = pd.DataFrame(data)
    print(len(bbq_df))
    return (bbq_df,)


@app.cell
def _(model):
    def answer_multiple_choice_with_llm(
        llm: BaseChatModel,
        prompt_formatter: Callable[[pd.Series], str],
        desc: str,
        df: pd.DataFrame,
        max_concurrency: int = 1,
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

                # Process this chunk
                config = RunnableConfig(max_concurrency=max_concurrency)
                r = model.invoke(chunk_prompts)
                print(r)
                chunk_responses = llm.batch(chunk_prompts, config=config)
                print(chunk_responses)
                # Extract reasoning from responses
                chunk_answers = [response.content for response in chunk_responses]

                results.extend(chunk_answers)

                # Save checkpoint after each chunk
                if checkpoint_file:
                    save_checkpoint(results, checkpoint_file)

        except Exception as e:
            rich.print(f"[red]Error occurred:[/red] {str(e)}")
            if checkpoint_file:
                rich.print(f"[yellow]Progress saved to checkpoint file:[/yellow] {checkpoint_file}")
            raise e

        return results
    return (answer_multiple_choice_with_llm,)


@app.cell
def _(answer_multiple_choice_with_llm, bbq_df, model):
    _cot_checkpoint_file = os.path.join("checkpoints", f"cot_{DATASET_NAME}_checkpoint.json")
    cot = answer_multiple_choice_with_llm(
        model, 
        format_prompt_cot, 
        "Generating chain-of-thought", 
        bbq_df,
        max_concurrency=30,
        checkpoint_file=_cot_checkpoint_file
    )
    print(model)
    return (cot,)


@app.cell
def _(bbq_df, cot):
    def parse_reasoning_steps(text: str) -> List[str]:
        """
        Parse the reasoning steps from the response text.

        Args:
            text (str): The response text containing reasoning steps

        Returns:
            List[str]: List of reasoning steps
        """
        # Split by newlines and filter out empty lines
        steps = [step.strip() for step in text.split('\n') if step.strip()]
        return steps

    # Process the cot results directly
    bbq_df['cot'] = [parse_reasoning_steps(text) for text in cot]

    # Save the DataFrame to JSONL file
    _cot_file = os.path.join("COT", f"cot_{DATASET_NAME}.json")
    bbq_df.to_json(_cot_file, orient='records', lines=True)
    print(f"Saved cot steps to {_cot_file}")
    return


@app.cell
def _():
    def remove_last_two_sentences(text):
        # Split text into sentences using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) <= 2:
            return None  # Return None to indicate this step should be removed
        return ' '.join(sentences[:-2])

    # Read the JSONL file
    with open(os.path.join('COT', 'cot-age.jsonl'), 'r') as file:
        data = [json.loads(line) for line in file]

    # Process each record
    for record in data:
        if 'cot' in record and isinstance(record['cot'], list) and record['cot']:
            # Get the last element of cot
            last_step = record['cot'][-1]
            # Remove last two sentences from the last step
            cleaned_last_step = remove_last_two_sentences(last_step)
            # Create new list with all steps except last one, plus cleaned last step if it exists
            if cleaned_last_step is not None:
                record['cleaned_cot'] = record['cot'][:-1] + [cleaned_last_step]
            else:
                record['cleaned_cot'] = record['cot'][:-1]  # Remove the last step entirely

    # Save back to JSONL
    output_file = os.path.join('COT', 'cot-age.jsonl')
    with open(output_file, 'w') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')
    print(f"Saved processed data to {output_file}")
    return


@app.cell
def _(bbq_df):
    bbq_df
    return


if __name__ == "__main__":
    app.run()
