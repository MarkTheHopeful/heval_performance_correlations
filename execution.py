import json
import multiprocessing
from math import comb
from pathlib import Path

from grazie.api.client.profiles import Profile
from grazie.api.client.endpoints import GrazieApiGatewayUrls
from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient, GrazieAgent
from grazie.api.client.chat.prompt import ChatPrompt

from config import Config
from utils import load_tasks

LLM_USED = {"claude-3.7": Profile.ANTHROPIC_CLAUDE_37_SONNET,
            "claude-3.5-sonnet": Profile.ANTHROPIC_CLAUDE_35_SONNET,
            "gpt-4": Profile.OPENAI_GPT_4,
            "gpt-4o-mini": Profile.OPENAI_GPT_4_O_MINI,
            "claude-3h": Profile.ANTHROPIC_CLAUDE_3_HAIKU,
            }

class LLMProvider:
    def __init__(self, token, config):
        url = url = GrazieApiGatewayUrls.PRODUCTION if config.is_prod else GrazieApiGatewayUrls.STAGING
        self.profile = LLM_USED[config.model.llm]
        self.prompt = config.model.prompt
        self.client = GrazieApiGatewayClient(
            url=url,
            grazie_jwt_token=token,
            auth_type=AuthType.USER,
            grazie_agent=GrazieAgent(name="grazie-api-gateway-client-heval-test", version="dev")
        )

    def make_call(self, task):
        chat = (
            ChatPrompt()
            .add_system(self.prompt)
            .add_user(task)
        )
        response = self.client.chat(
            chat=chat,
            profile=self.profile
        )
        return response.content


def run_some_task(i, path, provider):
    tasks = load_tasks(path)
    if i < 0 or i >= len(tasks):
        raise IndexError(f"Index {i} is out of range")

    task = tasks[i]
    task_id = task["task_id"]
    prompt = task["prompt"]

    print(f"\n=== Task {i} | ID: {task_id} ===")
    try:
        solution = provider.make_call(prompt)
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


def evaluate_some(i, provider, dataset_path="HumanEval.jsonl"):
    result = run_some_task(i, dataset_path, provider)
    if result is None:
        return None

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


def run_all_tasks(config: Config, provider: LLMProvider, output_path: Path):
    max_index = config.evaluation.tasks
    num_candidates = config.evaluation.candidates
    tasks = load_tasks(config.data.dataset_path)
    with output_path.open(mode='w') as out_file:
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
                    solution = provider.make_call(prompt)
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


def run_exec_in_thread(queue, full_code, test_code):
    namespace = {}
    try:
        exec(full_code, namespace)
        exec(test_code, namespace)
        result = namespace["check"](namespace["candidate"])
        queue.put(("pass", result))
    except Exception as e:
        queue.put(("fail", str(e)))

def run_exec(full_code, test_code, timeout):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=run_exec_in_thread, args=(queue, full_code, test_code))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        return False, "Timeout"

    status, result = queue.get()
    return status == "pass", result

def evaluate_all(config: Config, solutions_path: Path, output_path: Path):
    tasks = {task["task_id"]: task for task in load_tasks(config.data.dataset_path)}
    grouped_solutions = {}

    with solutions_path.open() as f:
        for line in f:
            record = json.loads(line)
            grouped_solutions.setdefault(record["task_id"], []).append(record["solution"])

    max_tasks = config.evaluation.tasks
    k = config.evaluation.k

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
        passes = []

        for idx, solution in enumerate(solutions):
            full_code = f"{prompt.strip()}\n{solution.strip()}\ncandidate = {entry_point}"
            status, message = run_exec(full_code, test_code, timeout=10)
            passes.append(status)
            if status:
                c += 1
                print(f"Task {task_id} candidate {idx} PASS")
            else:
                print(f"Task {task_id} candidate {idx} FAIL: {message}")

        pass_at_k = compute_pass(n, c, k)
        print(f"\nTask {task_id}: n = {n}, correct = {c}, pass@{k} = {pass_at_k:.4f}\n")
        results.append({
            "task_id": task_id,
            "total_candidates": n,
            "correct_candidates": c,
            "pass@k": pass_at_k,
            "passes": passes,
        })

    if results:
        avg_pass_at_k = sum(r["pass@k"] for r in results) / len(results)
        print(f"Average pass@{k}: {avg_pass_at_k:.4f}")
    else:
        print("No tasks were evaluated.")

    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    return results
