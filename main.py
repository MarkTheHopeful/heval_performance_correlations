import os
import sys
from pathlib import Path
import json
from math import comb
from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient, GrazieAgent
from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.profiles import Profile
from grazie.api.client.endpoints import GrazieApiGatewayUrls

token = os.getenv("AI_TOKEN")
url = GrazieApiGatewayUrls.PRODUCTION if os.getenv("IS_PROD") else GrazieApiGatewayUrls.STAGING

client = GrazieApiGatewayClient(
    url = url,
    grazie_jwt_token = token, # Provide the authentication token
    auth_type = AuthType.USER, # Set the user authentication type
    grazie_agent = GrazieAgent(name="grazie-api-gateway-client-heval-test", version="dev") # Define the agent name and version
)

SYSTEM_PROMPT = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
    Your response must consist of a single Python function definition only — no explanations, comments, or additional output. Return strictly valid Python code.
"""

def makeCall(task, profile, sys_prompt=SYSTEM_PROMPT):
    chat = (
                ChatPrompt()
                .add_system(sys_prompt)
                .add_user(task)
            )
    response = client.chat(
            chat = chat,
            profile = profile
            )
    return response.content

def load_tasks(data):
    with open(data, 'r') as f:
        return [json.loads(line) for line in f]

def run_some_task(i, path, llm_profile):
    tasks = load_tasks(path)
    if i < 0 or i >= len(tasks):
        raise IndexError(f"Index {i} is out of range")

    task = tasks[i]
    task_id = task["task_id"]
    prompt = task["prompt"]

    print(f"\n=== Task {i} | ID: {task_id} ===")
    try:
        solution = makeCall(prompt, llm_profile)
        print("Generated solution:\n", solution)
        return {
            "task_id": task_id,
            "prompt": prompt,
            "entry_point": task["entry_point"],
            "solution": solution,
            "test": task["test"]
        }
    except Exception as e:
        print(f"Error on task {task_id}: {e}")
        return None

def evaluate_some(i, llm_profile, dataset_path="HumanEval.jsonl"):
    result = run_some_task(i, dataset_path, llm_profile)
    if result is None:
        return

    task_id = result["task_id"]
    prompt = result["prompt"]
    solution = result["solution"]
    entry_point = result["entry_point"]
    test_code = result["test"]

    print("Running solution and test...")

    namespace = {}
    try:
        full_code = prompt + solution + f"\ncandidate = {entry_point}"
        exec(full_code, namespace)

        exec(test_code, namespace)

        namespace["check"](namespace["candidate"])

        passed = True
        print(f"PASS: All tests passed for {task_id}")
    except AssertionError as e:
        passed = False
        print(f"FAIL: Assertion failed for {task_id}: {e}")
    except Exception as e:
        passed = False
        print(f"FAIL: Error during evaluation for {task_id}: {e}")

    return {
        "task_id": task_id,
        "passed": passed,
        "solution": solution
    }

# def run_all_tasks(dataset_path, output_path="generated_solutions.jsonl", max_index=None):
#     tasks = load_tasks(dataset_path)
#     with open(output_path, 'w') as out_file:
#         for i, task in enumerate(tasks):
#             if max_index is not None and i >= max_index:
#                 break

#             task_id = task["task_id"]
#             prompt = task["prompt"]
#             entry_point = task["entry_point"]

#             print(f"\n=== Task {i} | ID: {task_id} ===")
#             try:
#                 solution = makeCall(prompt)
#                 if not solution.strip():
#                     raise ValueError("Empty response from model")

#                 print("Solution generated.")
#                 record = {
#                     "task_id": task_id,
#                     "entry_point": entry_point,
#                     "solution": solution
#                 }
#                 out_file.write(json.dumps(record) + "\n")
#             except Exception as e:
#                 print(f"Error in task {task_id}: {e}")

# def evaluate_all(dataset_path, solutions_path="generated_solutions.jsonl", max_index=None):
#     tasks = {task["task_id"]: task for task in load_tasks(dataset_path)}
#     results = []

#     with open(solutions_path, 'r') as f:
#         for i, line in enumerate(f):
#             if max_index is not None and i >= max_index:
#                 break
#             record = json.loads(line)
#             task_id = record["task_id"]
#             entry_point = record["entry_point"]
#             solution = record["solution"]
#             task = tasks[task_id]
#             test_code = task["test"]
#             prompt = task["prompt"]

#             print(f"\n--- Evaluating task {i} | ID: {task_id} ---")

#             namespace = {}
#             try:
#                 full_code = prompt + solution + f"\ncandidate = {entry_point}"
#                 exec(full_code, namespace)
#                 exec(test_code, namespace)
#                 namespace["check"](namespace["candidate"])
#                 passed = True
#                 print(f"PASS")
#             except AssertionError as e:
#                 passed = False
#                 print(f"FAIL: Assertion failed: {e}")
#             except Exception as e:
#                 passed = False
#                 print(f"FAIL: Execution error: {e}")

#             results.append({
#                 "task_id": task_id,
#                 "passed": passed,
#                 "solution": solution
#             })

#     passed = sum(1 for r in results if r["passed"])
#     total = len(results)
#     print(f"\nSummary: {passed}/{total} tasks passed")

#     return results

def run_all_tasks(dataset_path, llm_profile, output_path="generated_solutions.jsonl", max_index=5, num_candidates=3):
    tasks = load_tasks(dataset_path)
    with open(output_path, 'w') as out_file:
        for i, task in enumerate(tasks):
            if max_index is not None and i >= max_index:
                break

            task_id = task["task_id"]
            prompt = task["prompt"]
            entry_point = task["entry_point"]

            print(f"\n=== Task {i} | ID: {task_id} ===")
            # Generate multiple candidates per task
            for cand in range(num_candidates):
                try:
                    solution = makeCall(prompt, llm_profile)
                    if not solution.strip():
                        raise ValueError("No response.")
                    print(f"Candidate {cand} solution generated.")
                    record = {
                        "task_id": task_id,
                        "entry_point": entry_point,
                        "solution": solution,
                        "candidate_index": cand
                    }
                    out_file.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"Error in task {task_id}, candidate {cand}: {e}")

def compute_pass(n, c, k):
    k = min(n, k)
    if n == 0 or c == 0:
        return 0.0
    return 1 - comb(n - c, k) / comb(n, k)

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

SOLUTION_METRICS = {
    "total_text_length": SolutionMetric("Total text length", total_text_length),
    "lines_count": SolutionMetric("Lines count", lines_count),
}

COMPARATIVE_METRICS = {}

TASK_METRICS = {
    "total_text_length": TaskMetric("Total text length", total_text_length),
    "lines_count": TaskMetric("Lines count", lines_count),
    "words_count": TaskMetric("Words count", words_count),
}

def evaluate_all(dataset_path, solutions_path="generated_solutions.jsonl", output_path="evaluation_results.jsonl", max_tasks=None, k=0):
    """
    Evaluate candidate solutions grouped by task and compute pass@k.

    max_tasks: Optional limit on the number of tasks to evaluate.
    k: The value 'k' for computing pass@k.
    """
    tasks = {task["task_id"]: task for task in load_tasks(dataset_path)}
    grouped_solutions = {}

    with open(solutions_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            grouped_solutions.setdefault(record["task_id"], []).append(record["solution"])

    task_ids = list(grouped_solutions.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    results = []
    for task_id in task_ids:
        task = tasks[task_id]
        prompt = task["prompt"]
        entry_point = task["entry_point"]
        test_code = task["test"]

        solutions = grouped_solutions[task_id]
        n = len(solutions)
        c = 0

        for idx, solution in enumerate(solutions):
            namespace = {}
            full_code = f"{prompt.strip()}\n{solution.strip()}\ncandidate = {entry_point}"
            try:
                exec(full_code, namespace)
                exec(test_code, namespace)
                namespace["check"](namespace["candidate"])
                c += 1
                print(f"Task {task_id} candidate {idx} PASS")
            except Exception as e:
                print(f"Task {task_id} candidate {idx} FAIL: {e}")

        pass_at_k = compute_pass(n, c, k)
        print(f"\nTask {task_id}: n = {n}, correct = {c}, pass@{k} = {pass_at_k:.4f}\n")
        results.append({
            "task_id": task_id,
            "total_candidates": n,
            "correct_candidates": c,
            "pass@k": pass_at_k,
        })

    if results:
        avg_pass_at_k = sum(r["pass@k"] for r in results) / len(results)
        print(f"Average pass@{k}: {avg_pass_at_k:.4f}")
    else:
        print("No tasks were evaluated.")

    with open(output_path, 'w') as f:
        json.dump(results, f)
    return results

def perform_metrics(dataset_path, solutions_path, output_path, max_tasks=None, k=0,
                    solution_metrics = SOLUTION_METRICS.keys(),
                    comparative_metrics = COMPARATIVE_METRICS.keys(),
                    task_metrics = TASK_METRICS.keys()):
    tasks = {task["task_id"]: task for task in load_tasks(dataset_path)}
    grouped_solutions = {}

    with open(solutions_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            grouped_solutions.setdefault(record["task_id"], []).append(record["solution"])

    task_ids = list(grouped_solutions.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    results = []
    for task_id in task_ids:
        task = tasks[task_id]
        prompt = task["prompt"]
        reference_solution = task["canonical_solution"]

        result = {"task_id": task_id}
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
        json.dump(results, f)
    return results


def name_from_config(llm_name, tasks, k):
    return f"{llm_name}-{tasks}-{k}"

LLM_USED = {"claude-3.7": Profile.ANTHROPIC_CLAUDE_37_SONNET,
            "gpt-4": Profile.OPENAI_GPT_4}

if __name__ == "__main__":
    llm = sys.argv[1] if len(sys.argv) > 1 else "claude-3.7"
    max_tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    file_prefix = name_from_config(llm, max_tasks, k)

    solutions_filename = f"{file_prefix}-generated_solutions.jsonl"
    eval_results_filename = f"{file_prefix}-evaluation_results.jsonl"
    metrics_run_filename = f"{file_prefix}-with-metrics.jsonl"
    if not Path(solutions_filename).exists():
        run_all_tasks("HumanEval.jsonl", llm_profile=LLM_USED[llm], output_path=solutions_filename, max_index=max_tasks)

    evaluate_all("HumanEval.jsonl", solutions_filename, output_path=eval_results_filename, max_tasks=max_tasks, k=k)
    perform_metrics("HumanEval.jsonl", solutions_filename, output_path=metrics_run_filename, max_tasks=max_tasks, k=k)
