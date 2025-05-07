import os
from pathlib import Path

from metrics import perform_metrics
from execution import run_all_tasks, evaluate_all, LLMProvider
from config import Config, get_config

SYSTEM_PROMPT = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
    Your response must consist of a single Python function definition only — no explanations, comments, or additional output. Return strictly valid Python code.
"""

DATASETS = {
    "heval": "HumanEval.jsonl",
    "hevalplus": "HumanEvalPlus-OriginFmt.jsonl",
}

if __name__ == "__main__":
    token = os.getenv("AI_TOKEN")
    if token is None:
        raise RuntimeError("AI_TOKEN is not provided")
    config = get_config()

    provider = LLMProvider(token, config)
    file_prefix = config.get_label()

    work_path = Path("runs").joinpath(file_prefix)
    os.makedirs(work_path, exist_ok=True)

    solutions_path = work_path.joinpath("generated_solutions.jsonl")
    eval_results_path = work_path.joinpath("evaluation_results.jsonl")
    metrics_run_path = work_path.joinpath("metrics.jsonl")

    if not solutions_path.exists():
        run_all_tasks(provider=provider, config=config, output_path=solutions_path)

    if not eval_results_path.exists():
        evaluate_all(config=config, solutions_path=solutions_path, output_path=eval_results_path)

    if not metrics_run_path.exists():
        perform_metrics(config=config, solutions_path=solutions_path, eval_results_path=eval_results_path,
                        output_path=metrics_run_path)
