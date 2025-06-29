# Mitigating LLM Social Bias by Assessing and Filtering Reasoning Steps with a Multi-Judge Pipeline

## Authors

- Fatemehzahra Ghafari Ghomi (<fatemezahra.ghafari@studio.unibo.it>)
- Shafagh Rastegari (<shafagh.rastegari@studio.unibo.it>)
- Habib Kazemi (<habib.kazemi2@studio.unibo.it>)

## Project Structure

The repository is organized into two main directories:

- `BBQ/`: Contains the code and data for running experiments on the original English BBQ dataset.
- `MBBQ/`: Contains the code and data for our multilingual extension of the BBQ benchmark, covering English, Spanish, Dutch, and Turkish.

## Setup and Installation

### Prerequisites

- Python 3.12 or higher
- uv (Python package manager)

### Installation

1. **Install uv** (if not already installed)

   ```bash
   # macOS/Linux with curl
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows with PowerShell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

   ```

2. **Set up the project**

   ```bash
   # Clone the repository
   git clone <repository-url>
   cd path/to/project
   
   # Create virtual environment and install dependencies using uv
   uv sync

   # Activate a virtual environment using uv
   source .venv/bin/activate
   ```

## Running the projects

The main project notebooks are interactive [Marimo](https://marimo.io/) notebooks. They provide a reactive and reproducible environment for running the experiments.

To run notebooks of BBQ, First enter the BBQ directory:

```bash
cd BBQ
```

To run notebooks of MBBQ, First enter the MBBQ directory:

```bash
cd MBBQ
```

Then use the following command:

```bash
marimo edit <notebook-name>.py
```

## License

MIT License

Copyright (c) 2024 Fatemehzahra Ghafari Ghomi, Shafagh Rastegari, Habib Kazemi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
