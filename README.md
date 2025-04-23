# Trusted Editing Red-Team Experiment

This project implements a red-team experiment for testing trusted editing in code generation using the APPS dataset and OpenAI's GPT models.

## Setup

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```bash
   echo "OPENAI_API_KEY=your-api-key-here" > .env
   ```

## Running the Experiment

The main script supports several command-line arguments:

```bash
python trusted_editing.py \
    --num_tasks 20 \
    --untrusted_model gpt-4o \
    --trusted_model gpt-4
```

Key Arguments:
- `--num_tasks`: Number of APPS problems to process (default: 20)
- `--untrusted_model`: Model to use for initial code generation (default: gpt-4o)
- `--trusted_model`: Model to use for code editing/review (default: gpt-4)
- `--openai_api_key`: Optional API key (if not set in .env)
- `--seed`: Random seed for reproducibility (default: 42)

## Results

Results are saved in the `results/` directory as JSONL files with timestamps. Each line contains:
- Problem ID
- Whether the solution passed all test cases
- Whether a payload was detected
- Token usage
- Processing time

## Implementation Details

- Uses the HuggingFace `codeparrot/apps` dataset
- Focuses on interview-difficulty problems
- Implements a secure execution environment with resource limits
- Uses OpenAI API for both code generation and review

## Security Notes

- Code execution is sandboxed with CPU and memory limits (5s CPU time, 1GB memory)
- All generated code is reviewed by the trusted model before execution