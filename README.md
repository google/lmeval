# LMEval: Large Model Evaluation Framework

## Installation

```bash
pip install lmeval-framework
```

### Dev setup

LMeval use `uv` that you need to install then

```bash
uv venv
source .venv/bin/activate  # or open a new term in vscode after accepting the new venv
uv pip install -e .
```

## Disclaimer

- This is not a Google product.

- When using private datasets, remember that questions may be sent to your chosen model providers for evaluation if the requested results aren't already available. Ensure this aligns with your data policies regarding external processing of sensitive information.
