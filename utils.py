import json


def load_tasks(data):
    with open(data, 'r') as f:
        return [json.loads(line) for line in f]
