# Program for visual search of an image collection

## Prerequisites

---

[Python 3.12.3](https://www.python.org/downloads/release/python-3123/) If on Windows, if you have multiple versions of Python installed, make sure this is the foremost python in your Paths system environment variables.

[Poetry 2.2.1](https://python-poetry.org/)

If running inside VSCode, and desiring to run data_exploration.ipynb have the following extensions:

1. Jupyter (from Microsoft)
2. Jupyter Cell Tags (from Microsoft)
3. Jupyter Keymap (from Microsoft)
4. Jupyter Notebook Renderers (from Microsoft)
5. Python (from Microsoft)

## Local Setup

---

In the root of this repository run the following:

```
pip install poetry==2.2.1
poetry config --local virtualenvs.in-project true
poetry install
```

`poetry install` will resolve the exact versions listed in `pyproject.toml`, and `poetry shell` drops you into the managed virtualenv. 

Make a copy of example.env and rename it to .env. This example env file contains the `Base_Path` for all folders in the repo. When `BASE_PATH` is unset the scripts default to the repository root, so keeping everything under the cloned directory is usually the easiest option.

## Run visual search program with evaluations

To run the full search-and-evaluation workflow, tweak `DESCRIPTOR_SUBFOLDER`, `EXPERIMENT_NAME`, and other constants near the top of `src/cvpr_visualsearch.py`, then execute `poetry run python src/cvpr_visualsearch.py`. The script loads the chosen descriptors based on `DESCRIPTOR_SUBFOLDER`, samples queries, renders the ranked grids (saved as `results/queries_and_top_results_<experiment>.png`), and also writes metrics such as precision/recall CSVs and confusion matrices under `results/`. Re-run the script whenever you switch descriptor folders or want updated evaluation artifacts.

## Compute Descriptors

To compute new descriptors, specify the feature extract function desired by adjusting line 45 that creates the `F` variable. If using the `rgb_histogram()` function then specify `Q` under Constants. Also tweak `OUT_SUBFOLDER` to distinguish the descriptors that you are creating. Otherwise, the program will overwrite the descriptors in the `OUT_SUBFOLDER` specified.
