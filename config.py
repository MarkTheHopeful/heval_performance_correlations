import argparse
import json
import os
from typing import List
from pydantic import BaseModel


class ModelConfig(BaseModel):
    llm: str
    prompt: str


class DataConfig(BaseModel):
    dataset: str
    dataset_path: str


class EvaluationConfig(BaseModel):
    tasks: int
    candidates: int
    k: int
    metrics: List[str]


class Config(BaseModel):
    is_prod: bool
    model: ModelConfig
    data: DataConfig
    evaluation: EvaluationConfig

    def get_label(self) -> str:
        return f"{self.model.llm}-{self.data.dataset}-{self.evaluation.tasks}-{self.evaluation.candidates}-{self.evaluation.k}"


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    return Config(**data)


def merge_args_with_config(args, config: Config) -> Config:
    # Override only if args are explicitly passed
    if args.is_prod:
        config.is_prod = True
    if args.llm is not None:
        config.model.llm = args.llm
    if args.dataset is not None:
        config.data.dataset = args.dataset
    if args.max_tasks is not None:
        config.evaluation.tasks = args.max_tasks
    if args.num_candidates is not None:
        config.evaluation.candidates = args.num_candidates
    if args.k is not None:
        config.evaluation.k = args.k
    return config


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="HEval Performance Correlations Runner",
        description="Runs HumanEval (or other dataset) on specified model and measures some metrics on solutions, tasks text and canonical solutions.",
    )
    parser.add_argument("--config", default="config.json", help="Path to config JSON file")

    parser.add_argument("--is_prod", action="store_true", help="Run on the production API")
    parser.add_argument("--llm", help="Model name")
    parser.add_argument("--prompt", help="System prompt")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--dataset_path", help="Path to the dataset file")
    parser.add_argument("--max_tasks", type=int, help="Number of tasks to run")
    parser.add_argument("--num_candidates", type=int, help="Number of candidate solutions per task")
    parser.add_argument("--k", type=int, help="pass@k value for evaluation")

    parsed = parser.parse_args()
    return parsed


def get_config() -> Config:
    args = parse_cli()
    config = load_config(args.config)
    return merge_args_with_config(args, config)
