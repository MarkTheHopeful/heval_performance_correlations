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
- [x] After everything else is done: some visualization
- [ ] Add more possible visualizations
- [x] Fix current CFG metric (it produces negative values sometimes) (Mark)
- [ ] Add config files to load and store run configurations (Mark)
- [ ] Locate and highlight situations of kind "same nontrivial CFG, different text"
- [x] Add HumanEval+ and other datasets of this kind
- [ ] Add ability to run metrics candidate-vise, not only task-vise
- [ ] Parse generated solutions more accurate (strip ```python from the beginning for some models, for example)
- [ ] Add more possible prompts and processing of them (i.e. allow to think before submitting, related to the previous task)

P.S.
Don't forget to use your Grazie token. Probably it would be meaningful to change access from Production to Staging/Dev if you have one.
