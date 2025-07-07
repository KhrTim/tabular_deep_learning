# 🧠 Tabular Deep Learning

A framework for training and evaluating deep learning models on tabular data, with support for YAML-based configuration and (upcoming) multi-GPU parallelism.

---

## 🚀 Project Setup

Clone the repository **with submodules** and run the setup script:

```bash
git clone --recursive git@github.com:KhrTim/tabular_deep_learning.git
cd tabular_deep_learning
chmod +x setup_project.sh
./setup_project.sh
```

This will:

- Create and activate a Conda environment from `environment.yml`
- Download required datasets via `download_data.sh`

---

## 📦 Usage

Run the main script with your desired config file:

```bash
python model_level_parallel_execution.py <config.yaml>
```

### 🔧 Examples

```bash
# Run all models with MLflow logging
python model_level_parallel_execution.py all_models_run.yaml

# Run in evaluation mode, log results to CSV (no MLflow)
python model_level_parallel_execution.py save_logs_to_csv.yaml
```

---

## ⚠️ Multi-GPU Support

> 🚧 **Note:** Multi-GPU execution is under development and currently disabled.

---

## 📁 Project Structure (Optional)

```
tabular_deep_learning/
├── configs/              # YAML config files
├── data/                 # Downloaded datasets
├── models/               # Model definitions
├── utils/                # Utility functions
├── setup_project.sh      # Environment and data setup
├── download_data.sh      # Dataset downloader
└── model_level_parallel_execution.py  # Main entry point
```

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
