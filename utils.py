import json
import signal


def load_tasks(data):
    with open(data, 'r') as f:
        return [json.loads(line) for line in f]

def handler(signum, frame):
    raise Exception("Timeout")

def set_timeout(seconds):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)