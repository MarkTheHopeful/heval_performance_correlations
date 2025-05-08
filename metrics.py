from difflib import SequenceMatcher
import json
from pathlib import Path

from config import Config
from utils import load_tasks
from graph_building import code_cfg_similarity, cfg_triviality


class SolutionMetric:
    def __init__(self, name, metric_function, full_name=None):
        self.name = name
        self.full_name = name if full_name is None else full_name
        self.f = metric_function

    def __call__(self, solution_candidate):
        return self.f(solution_candidate)


class ComparativeMetric:
    def __init__(self, name, metric_function, full_name=None):
        self.name = name
        self.full_name = name if full_name is None else full_name
        self.f = metric_function

    def __call__(self, solution_candidate, reference_solution):
        return self.f(solution_candidate, reference_solution)


class TaskMetric:
    def __init__(self, name, metric_function, full_name=None):
        self.name = name
        self.full_name = name if full_name is None else full_name
        self.f = metric_function

    def __call__(self, task_text):
        return self.f(task_text)


def total_text_length(solution):
    return sum(map(lambda x: len(x.strip().rstrip()), solution.split("\n")))


def lines_count(solution):
    return len(list(filter(lambda x: len(x) > 0, solution.split("\n"))))


def words_count(task_text):
    return len(list(filter(lambda x: len(x.strip().rstrip()) > 0, task_text.split())))


def gestalt_text_similarity(task_text, ref_text):
    matcher = SequenceMatcher(None, task_text, ref_text)
    return matcher.ratio()


ALL_METRICS = {
    "solution_length": SolutionMetric("solution_length", total_text_length, "Solution length (characters)"),
    "solution_lines": SolutionMetric("solution_lines", lines_count, "Solution length (lines)"),
    "triviality": SolutionMetric("triviality", cfg_triviality, "Solution CFG triviality"),
    "gestalt_similarity": ComparativeMetric("gestalt_similarity", gestalt_text_similarity, "Gestalt similarity between"),
    "cfg_similarity": ComparativeMetric("cfg_similarity", code_cfg_similarity, "CFG similarity between"),
    "task_length": TaskMetric("task_length", total_text_length, "Task length (characters)"),
    "task_lines": TaskMetric("task_lines", lines_count, "Task length (lines)"),
    "task_words": TaskMetric("task_words", words_count, "Task length (words)"),
}


def get_metrics_split(config: Config):
    all_metrics_keys = ALL_METRICS.keys() if len(config.evaluation.metrics) == 0 else config.evaluation.metrics
    all_metrics = [ALL_METRICS[x] for x in all_metrics_keys]
    solution_metrics = list(filter(lambda metric: isinstance(metric, SolutionMetric), all_metrics))
    comparative_metrics = list(filter(lambda metric: isinstance(metric, ComparativeMetric), all_metrics))
    task_metrics = list(filter(lambda metric: isinstance(metric, TaskMetric), all_metrics))
    return solution_metrics, comparative_metrics, task_metrics


def perform_metrics(config: Config, solutions_path: Path, eval_results_path: Path, output_path: Path):
    tasks = {task["task_id"]: task for task in load_tasks(config.data.dataset_path)}
    solution_metrics, comparative_metrics, task_metrics = get_metrics_split(config)
    grouped_solutions = {}
    evaluation_results = {}
    pass_at_k = {}

    with solutions_path.open() as f, eval_results_path.open() as f_e:
        for line in f:
            record = json.loads(line)
            grouped_solutions.setdefault(record["task_id"], []).append(record["solution"])
        for line in f_e:
            record = json.loads(line)
            evaluation_results[record["task_id"]] = record["passes"]
            pass_at_k[record["task_id"]] = record["pass@k"]

    task_ids = list(grouped_solutions.keys())
    task_ids = task_ids[:config.evaluation.tasks]

    results = []
    for task_id in task_ids:
        print(f"Metrics calculation starting for {task_id}")
        task = tasks[task_id]
        prompt = task["prompt"]
        reference_solution = task["canonical_solution"]

        result = {"task_id": task_id, "passes": evaluation_results[task_id], "pass@k": pass_at_k[task_id]}
        for metric in task_metrics:
            result[metric.name] = metric(prompt)

        solutions = grouped_solutions[task_id]
        n = len(solutions)
        for metric in comparative_metrics:
            result[metric.name] = []
            for idx, solution in enumerate(solutions):
                result[metric.name].append(metric(solution, reference_solution))
            result[f"Mean {metric.name}"] = sum(result[metric.name]) / n

        for metric in solution_metrics:
            result[metric.name] = []
            for idx, solution in enumerate(solutions):
                result[metric.name].append(metric(solution))
            result[f"Mean {metric.name}"] = sum(result[metric.name]) / n

        interest = []
        triviality, cfg_similarity, text_similarity = result["triviality"], result[
            "cfg_similarity"], result["gestalt_similarity"]
        for i in range(len(triviality)):
            interest.append(cfg_similarity[i] * (1 - triviality[i]) * (1 - text_similarity[i]))
        result["Interest"] = interest
        result["Mean interest"] = sum(interest) / n

        results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results
