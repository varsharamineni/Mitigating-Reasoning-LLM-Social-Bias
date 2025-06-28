import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import pandas as pd
    import marimo as mo
    import io

    file_uploader = mo.ui.file(
        filetypes=[".jsonl"],
        multiple=False,
        kind="area",
        label="Upload a dataset to visualize (.jsonl)",
    )
    mo.md(f"**Upload a dataset:** {file_uploader}")
    return file_uploader, io, mo, pd


@app.cell
def _(file_uploader, io, mo, pd):
    # This cell will re-execute whenever file_uploader.value changes
    uploaded_files = file_uploader.value

    if uploaded_files:
        # Get the contents of the first (and only, since multiple=False) uploaded file
        file_name = file_uploader.name()
        file_contents_bytes = file_uploader.contents()

        if file_contents_bytes:
            try:
                # Convert bytes to a string, then to a file-like object for pandas
                file_contents_str = file_contents_bytes.decode("utf-8")
                df = pd.read_json(
                    io.StringIO(file_contents_str), orient="records", lines=True
                )
                mo.output.replace(mo.ui.dataframe(df))
                mo.md(f"Displaying data from uploaded file: **{file_name}**")
            except Exception as e:
                mo.output.replace(mo.md(f"Error processing file {file_name}: {e}"))
        else:
            mo.output.replace(mo.md("Error: Uploaded file is empty."))
    else:
        mo.output.replace(
            mo.md("Please upload a .jsonl dataset file using the element above.")
        )
    return (df,)


@app.cell
def _(df):
    df[df["cot_answer"] != df["unbiased_cot_answer"]].index
    return


@app.cell
def _(df):
    df[df["target_loc"] == 100].index
    return


app._unparsable_cell(
    r"""
    df[(df[\"cot_answer\"] == df[\"unknown_label\"]) && (df[\"context_condition\"] == \"ambig\")]
    """,
    name="_"
)


if __name__ == "__main__":
    app.run()
