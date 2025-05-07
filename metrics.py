from difflib import SequenceMatcher
import json
from pathlib import Path

from config import Config
from utils import load_tasks
from graph_building import code_cfg_similarity, cfg_triviality


class SolutionMetric:
    def __init__(self, name, metric_function):
        self.name = f"SolM: {name}"
        self.f = metric_function

    def __call__(self, solution_candidate):
        return self.f(solution_candidate)


class ComparativeMetric:
    def __init__(self, name, metric_function):
        self.name = f"ComM: {name}"
        self.f = metric_function

    def __call__(self, solution_candidate, reference_solution):
        return self.f(solution_candidate, reference_solution)


class TaskMetric:
    def __init__(self, name, metric_function):
        self.name = f"TasM: {name}"
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

SOLUTION_METRICS = {
    "total_text_length": SolutionMetric("Total text length", total_text_length),
    "lines_count": SolutionMetric("Lines count", lines_count),
    "triviality": SolutionMetric("CFG Triviality", cfg_triviality)
}

COMPARATIVE_METRICS = {
    "gestalt_text_similarity": ComparativeMetric("Gestalt text similarity", gestalt_text_similarity),
    "cfg_code_similarity": ComparativeMetric("CFG code similarity", code_cfg_similarity),
}

TASK_METRICS = {
    "total_text_length": TaskMetric("Total text length", total_text_length),
    "lines_count": TaskMetric("Lines count", lines_count),
    "words_count": TaskMetric("Words count", words_count),
}


def perform_metrics(config: Config, solutions_path: Path, eval_results_path: Path, output_path: Path,
                    solution_metrics=SOLUTION_METRICS.keys(),
                    comparative_metrics=COMPARATIVE_METRICS.keys(),
                    task_metrics=TASK_METRICS.keys()):
    tasks = {task["task_id"]: task for task in load_tasks(config.data.dataset_path)}
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
        for task_metric in task_metrics:
            metric = TASK_METRICS[task_metric]
            result[metric.name] = metric(prompt)

        solutions = grouped_solutions[task_id]
        n = len(solutions)
        for c_metric in comparative_metrics:
            metric = COMPARATIVE_METRICS[c_metric]
            result[metric.name] = []
            for idx, solution in enumerate(solutions):
                result[metric.name].append(metric(solution, reference_solution))
            result[f"Mean {metric.name}"] = sum(result[metric.name]) / n

        for s_metric in solution_metrics:
            metric = SOLUTION_METRICS[s_metric]
            result[metric.name] = []
            for idx, solution in enumerate(solutions):
                result[metric.name].append(metric(solution))
            result[f"Mean {metric.name}"] = sum(result[metric.name]) / n

        interest = []
        triviality, cfg_similarity, text_similarity = result["SolM: CFG Triviality"], result["ComM: CFG code similarity"], result["ComM: Gestalt text similarity"]
        for i in range(len(triviality)):
            interest.append(cfg_similarity[i] * (1 - triviality[i]) * (1 - text_similarity[i]))
        result["Interest"] = interest
        result["Mean interest"] = sum(interest) / n

        results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results
