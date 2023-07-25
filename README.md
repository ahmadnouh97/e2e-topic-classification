# Mini MENT

# Directories

- **labs:** contains jupyter notebook labs.
- **data:** raw & processed data.
- **assets:** contains assets.
- **config:** contains:
    - **args.json** the model's training parameters.
    - **performance.json** the model's evaluation results.
    - **config.py** configurations and paths.
    - **run_id.txt** the latest mlflow run id for the latest experiment.
- **stores:** for storing the experiments' results, models and metadata.
- **UI:** a streamlit user interface.
---

# Requirements
- conda >= 23.5.0 (simply install [anaconda](https://www.anaconda.com/download))
- Python == 3.10
---

# Setup
- create virtual environment.
    ```
    conda create -p venv python=3.10
    conda activate ./venv
    python -m pip install -e .
    ```
---

# Run
- UI Demo:
    ```
    streamlit run ui/app.py
    ```

- Train:
    ```
    python mini_ment/main.py train
    ```

- Optimize (hyperparameter tuning):
    ```
    python mini_ment/main.py optimize
    ```