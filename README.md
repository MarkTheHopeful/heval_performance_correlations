How-To-Run:

1. Install Grazie api client: `pip3 install grazie_api_gateway_client`
2. Download and unpack the dataset from the [humaneval repo](https://github.com/openai/human-eval) (data)
3. Acquire token and provide it as an environment variable `AI_TOKEN`
4. Copy `default_config.json` into `config.json` or provide custom path to your own config
5. Specify which metrics to perform in the `metrics` field of the default config or leave it blank to run all of them
6. Run `python3 main.py`. Parameters can be specified via config file and overwritten using command line arguments

Things to do:

- [ ] Add more possible visualizations
- [ ] Add config files to load and store run configurations (Mark)
- [ ] Locate and highlight situations of kind "same nontrivial CFG, different text"
- [ ] Add ability to run metrics candidate-vise, not only task-vise
- [ ] Parse generated solutions more accurate (strip ```python from the beginning for some models, for example)
- [ ] Add more possible prompts and processing of them (i.e. allow to think before submitting, related to the previous
  task)
