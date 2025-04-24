from difflib import SequenceMatcher
import json
from utils import load_tasks
from graph_building import code_cfg_similarity


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


def perform_metrics(dataset_path, solutions_path, evaluated_path, output_path, max_tasks=None, k=0,
                    solution_metrics=SOLUTION_METRICS.keys(),
                    comparative_metrics=COMPARATIVE_METRICS.keys(),
                    task_metrics=TASK_METRICS.keys()):
    tasks = {task["task_id"]: task for task in load_tasks(dataset_path)}
    grouped_solutions = {}
    evaluation_results = {}
    pass_at_k = {}

    with open(solutions_path, 'r') as f, open(evaluated_path, 'r') as f_e:
        for line in f:
            record = json.loads(line)
            grouped_solutions.setdefault(record["task_id"], []).append(record["solution"])
        for line in f_e:
            record = json.loads(line)
            evaluation_results[record["task_id"]] = record["passes"]
            pass_at_k[record["task_id"]] = record["pass@k"]

    task_ids = list(grouped_solutions.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    results = []
    for task_id in task_ids:
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

        results.append(result)

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results
