import os
import sys
from pathlib import Path
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.profiles import Profile
from metrics import perform_metrics
from execution import run_all_tasks, evaluate_all, LLMProvider

token = os.getenv("AI_TOKEN")
url = GrazieApiGatewayUrls.PRODUCTION if os.getenv("IS_PROD") else GrazieApiGatewayUrls.STAGING

SYSTEM_PROMPT = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
    Your response must consist of a single Python function definition only — no explanations, comments, or additional output. Return strictly valid Python code.
"""


def name_from_config(llm_name, tasks, k):
    return f"{llm_name}-{tasks}-{k}"


LLM_USED = {"claude-3.7": Profile.ANTHROPIC_CLAUDE_37_SONNET,
            "gpt-4": Profile.OPENAI_GPT_4}

if __name__ == "__main__":
    llm = sys.argv[1] if len(sys.argv) > 1 else "claude-3.7"
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    provider = LLMProvider(url, token, LLM_USED[llm], SYSTEM_PROMPT)

    file_prefix = name_from_config(llm, max_tasks, k)

    solutions_filename = f"{file_prefix}-generated_solutions.jsonl"
    eval_results_filename = f"{file_prefix}-evaluation_results.jsonl"
    metrics_run_filename = f"{file_prefix}-with-metrics.jsonl"
    if not Path(solutions_filename).exists():
        run_all_tasks("HumanEval.jsonl", provider=provider, output_path=solutions_filename, max_index=max_tasks)

    evaluate_all("HumanEval.jsonl", solutions_filename, output_path=eval_results_filename, max_tasks=max_tasks, k=k)
    perform_metrics("HumanEval.jsonl", solutions_filename, evaluated_path=eval_results_filename,
                    output_path=metrics_run_filename, max_tasks=max_tasks, k=k)
