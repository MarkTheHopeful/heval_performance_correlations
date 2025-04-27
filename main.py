import os
from pathlib import Path
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.profiles import Profile
from metrics import perform_metrics
from execution import run_all_tasks, evaluate_all, LLMProvider
import argparse

token = os.getenv("AI_TOKEN")
url = GrazieApiGatewayUrls.PRODUCTION if os.getenv("IS_PROD") else GrazieApiGatewayUrls.STAGING

SYSTEM_PROMPT = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
    Your response must consist of a single Python function definition only — no explanations, comments, or additional output. Return strictly valid Python code.
"""

DATASETS = {
    "heval":     "HumanEval.jsonl",
    "hevalplus": "HumanEvalPlus-OriginFmt.jsonl",
}

def name_from_config(llm_name, dataset_label, tasks, candidates, k):
    return f"{llm_name}-{dataset_label}-{tasks}-{candidates}-{k}"

LLM_USED = {"claude-3.7": Profile.ANTHROPIC_CLAUDE_37_SONNET,
            "gpt-4": Profile.OPENAI_GPT_4,
            "claude-3h": Profile.ANTHROPIC_CLAUDE_3_HAIKU,
            "gpt-4o-mini": Profile.OPENAI_GPT_4_O_MINI}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="HEval Performance Correlations Runner",
        description="Runs HumanEval on specified model and measures some metrics on solutions, tasks text and canonical solutions.",
    )
    parser.add_argument("--llm",           default="claude-3.7",     help="Model key")
    parser.add_argument("--dataset",       default="heval",         help="Dataset key (heval|hevalplus) or path to JSONL")
    parser.add_argument("--max_tasks",     default=10,   type=int,   help="Number of tasks to run")
    parser.add_argument("--num_candidates", default=5,    type=int,   help="Number of candidate solutions per task")
    parser.add_argument("--k",             default=3,    type=int,   help="pass@k value for evaluation")

    parsed = parser.parse_args()
    llm = parsed.llm

    if parsed.dataset in DATASETS:
        dataset_path = DATASETS[parsed.dataset]
        dataset_label = parsed.dataset
    else:
        dataset_path = parsed.dataset
        dataset_label = Path(dataset_path).stem

    max_tasks = parsed.max_tasks
    num_candidates = parsed.num_candidates
    k = parsed.k
    
    provider = LLMProvider(url, token, LLM_USED[llm], SYSTEM_PROMPT)
    file_prefix = name_from_config(llm, dataset_label, max_tasks, num_candidates, k)

    solutions_filename = f"{file_prefix}-generated_solutions.jsonl"
    eval_results_filename = f"{file_prefix}-evaluation_results.jsonl"
    metrics_run_filename = f"{file_prefix}-with-metrics.jsonl"

    if not Path(solutions_filename).exists():
        run_all_tasks(
            dataset_path,
            provider=provider,
            output_path=solutions_filename,
            max_index=max_tasks,
            num_candidates=num_candidates,
        )

    evaluate_all(
        dataset_path,
        solutions_filename,
        output_path=eval_results_filename,
        max_tasks=max_tasks,
        k=k
    )

    perform_metrics(
        dataset_path,
        solutions_filename,
        evaluated_path=eval_results_filename,
        output_path=metrics_run_filename,
        max_tasks=max_tasks,
        k=k
    )
