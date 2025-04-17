How-To-Run:

1. Install Grazie api client: `pip3 install grazie_api_gateway_client`
2. Download and unpack the dataset from the [humaneval repo](https://github.com/openai/human-eval) (data)
3. Acquire token and provide it as an environment variable `AI_TOKEN`
4. Run `python3 main.py`. Parameters can be specified via command line (no documentation yet)

Things to do:

- [x] Get the right system promt for human eval/human eval+
- [x] Add iteration over downloaded dataset (that sends the task to the model)
- [x] Add evaluation function for the results produced by an LLM (probably copy the one from the original repo)
- [x] Add pass@k function (optionally)
- [x] Add a basic metric on task just as an example
- [x] Add a basic metric on generated solution just as an example
- [ ] After everything else is done: some visualization

P.S.
Don't forget to use your Grazie token. Probably it would be meaningful to change access from Production to Staging/Dev if you have one.
