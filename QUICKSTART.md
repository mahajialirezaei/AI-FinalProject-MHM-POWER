# Quick Start Guide

Get up and running with the Bank Marketing EDA project in minutes!

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**What this installs:**
- `pandas` - Data manipulation
- `matplotlib` - Plotting
- `seaborn` - Statistical visualizations
- `numpy` - Numerical computing
- `pyyaml` - Configuration parsing

### Step 2: Run EDA

```bash
python run_eda.py
```

**That's it!** The script will:
- ✅ Load the dataset
- ✅ Generate 6 visualizations
- ✅ Save figures to `reports/figures/`
- ✅ Display statistics in console

## 📊 What You'll Get

After running, check `reports/figures/` for:

1. **class_imbalance_analysis.png** - Target variable distribution
2. **categorical_conversion_rate.png** - Conversion by job/marital/education
3. **numerical_outliers_analysis.png** - Age & balance analysis
4. **correlation_heatmap.png** - Feature correlations
5. **seasonality_analysis.png** - Monthly trends
6. **duration_analysis.png** - Call duration patterns

## 🎯 Common Tasks

### Run EDA Analysis
```bash
python run_eda.py
```

### Use Makefile Commands
```bash
make install    # Install dependencies
make run        # Run EDA
make test       # Run tests
make clean      # Clean generated files
```

### Use as Python Module
```python
from src.eda.data_loader import load_data
from src.eda.visualizations import plot_class_imbalance

# Load data
df = load_data()

# Create custom visualization
plot_class_imbalance(df, save=False)  # Display without saving
```

### Use in Jupyter Notebook
```python
# In notebooks/01_exploratory_data_analysis.ipynb
from src.eda.data_loader import load_data
df = load_data()
```

## ⚙️ Customization

Edit `config/config.yaml` to customize:

```yaml
data:
  raw: "data/raw/bank/bank-full.csv"  # Change data path
  delimiter: ";"                       # Change delimiter

output:
  figures_dir: "reports/figures"      # Change output directory

eda:
  numerical_vars: [age, balance, ...]  # Select variables
  categorical_vars: [job, marital, ...]
  
plotting:
  style: "whitegrid"                   # Change plot style
  dpi: 300                             # Change resolution
```

**No code changes needed!** Just edit the YAML file.

## 📁 Project Structure at a Glance

```
project/
├── run_eda.py          # ← Start here!
├── config/config.yaml  # ← Customize here
├── src/eda/           # ← Source code
├── reports/figures/   # ← Outputs go here
└── notebooks/         # ← Interactive analysis
```

## 🔧 Troubleshooting

### "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError"
- Make sure you're in the project root directory
- Check that `data/raw/bank/bank-full.csv` exists
- Verify paths in `config/config.yaml`

### "Import Error"
```bash
pip install -e .  # Install package in development mode
```

## 📚 Next Steps

1. **Explore the code**: Check `src/eda/` for reusable functions
2. **Customize**: Edit `config/config.yaml` for your needs
3. **Extend**: Add new analysis functions in `src/eda/visualizations.py`
4. **Test**: Write tests in `tests/` for new features
5. **Document**: Update README when adding features

## 💡 Pro Tips

- **Use Makefile**: `make run` is faster than typing the full command
- **Configuration**: Keep all settings in `config/config.yaml`
- **Modular Code**: Import functions instead of copying code
- **Notebooks**: Use `notebooks/` for interactive exploration
- **Version Control**: Data files are gitignored (as they should be)

## 🎓 Learning Path

1. **Beginner**: Run `python run_eda.py` and explore outputs
2. **Intermediate**: Modify `config/config.yaml` and see changes
3. **Advanced**: Add new functions to `src/eda/visualizations.py`
4. **Expert**: Create new modules in `src/` for model training

## 📖 Documentation

- **README.md** - Full project documentation
- **docs/PROJECT_STRUCTURE.md** - Detailed structure guide
- **MIGRATION_GUIDE.md** - If migrating from old structure
- **docs/RESTRUCTURING_SUMMARY.md** - What changed and why

## ⚡ Quick Reference

| Task | Command |
|------|---------|
| Install | `pip install -r requirements.txt` |
| Run EDA | `python run_eda.py` |
| Run Tests | `pytest tests/` |
| Clean Files | `make clean` |
| Install Dev | `make install-dev` |

## 🆘 Need Help?

1. Check the **README.md** for detailed documentation
2. Review **docs/PROJECT_STRUCTURE.md** for structure details
3. Look at example code in `src/eda/`
4. Check test files in `tests/` for usage examples

---

**Ready to go?** Run `python run_eda.py` and start exploring! 🚀

