import os
import json
from math import comb
from grazie.api.client.gateway import AuthType, GrazieApiGatewayClient, GrazieAgent
from grazie.api.client.chat.prompt import ChatPrompt
from grazie.api.client.profiles import Profile
from grazie.api.client.endpoints import GrazieApiGatewayUrls

token = os.getenv("AI_TOKEN")

client = GrazieApiGatewayClient(
    url = GrazieApiGatewayUrls.STAGING,
    grazie_jwt_token = token, # Provide the authentication token
    auth_type = AuthType.APPLICATION, # Set the user authentication type
    grazie_agent = GrazieAgent(name="grazie-api-gateway-client-heval-test", version="dev") # Define the agent name and version
)

SYSTEM_PROMPT = """
    You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
    Your response must consist of a single Python function definition only — no explanations, comments, or additional output. Return strictly valid Python code.
"""

def makeCall(task, sys_prompt=SYSTEM_PROMPT):
    chat = (
                ChatPrompt()
                .add_system(sys_prompt)
                .add_user(task)
            )
    response = client.chat(
            chat = chat,
            profile = Profile.OPENAI_GPT_4
            )
    return response.content

def load_tasks(data):
    with open(data, 'r') as f:
        return [json.loads(line) for line in f]

def run_some_task(i, path):
    tasks = load_tasks(path)
    if i < 0 or i >= len(tasks):
        raise IndexError(f"Index {i} is out of range")

    task = tasks[i]
    task_id = task["task_id"]
    prompt = task["prompt"]

    print(f"\n=== Task {i} | ID: {task_id} ===")
    try:
        solution = makeCall(prompt)
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

def evaluate_some(i, dataset_path="HumanEval.jsonl"):
    result = run_some_task(i, dataset_path)
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

def run_all_tasks(dataset_path, output_path="generated_solutions.jsonl", max_index=5, num_candidates=3):
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
                    solution = makeCall(prompt)
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

def evaluate_all(dataset_path, solutions_path="generated_solutions.jsonl", max_tasks=None, k=0):
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

    return results

if __name__ == "__main__":
    run_all_tasks("HumanEval.jsonl", "generated_solutions.jsonl", max_index=10)
    evaluate_all("HumanEval.jsonl", "generated_solutions.jsonl", max_tasks=10, k=5)