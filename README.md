# 🧠 Tabular Deep Learning

## 🚀 Project Setup

Clone the repository **with submodules** and run the setup script:

```bash
git clone --recursive git@github.com:KhrTim/tabular_deep_learning.git
cd tabular_deep_learning
chmod +x setup_project.sh
./setup_project.sh
```

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

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
