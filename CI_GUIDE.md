# CI/CD Pipeline Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Overview](#pipeline-overview)
3. [What Gets Checked](#what-gets-checked)
4. [How to Use](#how-to-use)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)
7. [Customization](#customization)
8. [Best Practices](#best-practices)

---

## Introduction

### What is CI/CD?

**Continuous Integration (CI)** is a development practice where code changes are automatically built, tested, and verified whenever developers push code to a shared repository. This helps catch bugs early, ensures code quality, and maintains a stable codebase.

**Continuous Deployment (CD)** extends CI by automatically deploying code that passes all tests to production or staging environments.

For this project, we're implementing **CI** - automated testing and code quality checks that run on every push and pull request.

### Why Use CI/CD?

- **Early Bug Detection**: Catch errors before they reach production
- **Code Quality**: Enforce consistent coding standards across the team
- **Automated Testing**: Ensure all tests pass before code is merged
- **Confidence**: Know that your code works before merging
- **Documentation**: CI serves as living documentation of project requirements
- **Time Saving**: Automate repetitive checks instead of running them manually

---

## Pipeline Overview

The CI pipeline for this project consists of three main jobs that run in parallel:

```
┌─────────────────┐
│   Push/PR       │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Code Quality │  │   Testing    │  │  Verify      │
│   Checks     │  │              │  │  Scripts     │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Pipeline Jobs

1. **Code Quality Checks** (`code-quality`)
   - Code formatting (Black)
   - Linting (flake8)
   - Type checking (mypy)

2. **Testing** (`test`)
   - Runs pytest test suite
   - Requires dataset download

3. **Script Verification** (`verify-scripts`)
   - Runs EDA script
   - Runs preprocessing script
   - Uploads generated artifacts

### When Does It Run?

The pipeline automatically triggers on:
- **Every push** to any branch
- **Every pull request** to any branch

You don't need to do anything - GitHub Actions runs the pipeline automatically!

---

## What Gets Checked

### 1. Code Quality Checks Job

#### Black Format Check
- **Tool**: Black (Python code formatter)
- **What it does**: Verifies that code follows Black's formatting standards
- **Command**: `black --check src/ tests/`
- **What fails**: If code doesn't match Black's formatting style
- **How to fix**: Run `black src/ tests/` locally to auto-format your code

#### Flake8 Linting
- **Tool**: flake8 (Python linter)
- **What it does**: Checks for code style violations, potential bugs, and complexity issues
- **Command**: `flake8 src/ tests/ --max-line-length=100`
- **What fails**: Style violations, unused imports, undefined names, etc.
- **How to fix**: Address the issues reported by flake8 in your code

#### Mypy Type Checking
- **Tool**: mypy (Static type checker)
- **What it does**: Checks for type errors and inconsistencies
- **Command**: `mypy src/ --ignore-missing-imports`
- **What fails**: Type errors, incompatible type assignments
- **Note**: Currently runs with `--ignore-missing-imports` and won't fail the build (indicated by `|| true`)
- **How to fix**: Add type hints and fix type errors in your code

### 2. Testing Job

#### Pytest Test Suite
- **Tool**: pytest (Python testing framework)
- **What it does**: Runs all unit tests in the `tests/` directory
- **Command**: `pytest tests/ -v`
- **What fails**: If any test fails or raises an exception
- **Requirements**: Downloads the dataset before running tests
- **How to fix**: Fix failing tests or update test expectations

### 3. Script Verification Job

#### EDA Script Execution
- **What it does**: Runs the complete EDA pipeline (`run_eda.py`)
- **Purpose**: Verifies that the EDA script executes without errors
- **What fails**: If the script crashes, raises exceptions, or can't find required files
- **How to fix**: Debug the script locally and fix any issues

#### Preprocessing Script Execution
- **What it does**: Runs the preprocessing pipeline (`src.preprocessing.main`)
- **Purpose**: Verifies that preprocessing completes successfully
- **What fails**: If preprocessing crashes or produces errors
- **How to fix**: Check preprocessing code and fix any bugs

#### Artifact Upload
- **What it does**: Uploads generated files (figures, processed data) as artifacts
- **Purpose**: Allows you to download and inspect generated outputs
- **Location**: GitHub Actions artifacts (available for 1 day)
- **Contents**: 
  - `reports/figures/` - EDA visualization outputs
  - `data/processed/` - Processed dataset files

---

## How to Use

### Viewing Pipeline Status

1. **On GitHub Repository**:
   - Navigate to the "Actions" tab in your repository
   - You'll see a list of all workflow runs
   - Click on any run to see detailed status

2. **On Pull Requests**:
   - CI status appears as checks at the bottom of the PR
   - Green checkmark ✅ = All checks passed
   - Red X ❌ = One or more checks failed
   - Yellow circle ⏳ = Checks are still running

3. **Status Badge** (if added to README):
   - Shows the latest pipeline status
   - Updates automatically

### Viewing Pipeline Logs

1. Go to the "Actions" tab
2. Click on the workflow run you want to inspect
3. Click on a specific job (e.g., "Code Quality Checks")
4. Expand individual steps to see detailed logs
5. Use the search/filter to find specific errors

### Running Checks Locally

Before pushing, you can run the same checks locally:

```bash
# Install dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy scikit-learn joblib

# Check formatting
black --check src/ tests/

# Format code (fixes issues automatically)
black src/ tests/

# Run linting
flake8 src/ tests/ --max-line-length=100

# Run type checking
mypy src/ --ignore-missing-imports

# Run tests
pytest tests/ -v

# Run EDA script
python run_eda.py

# Run preprocessing
python -m src.preprocessing.main
```

### Pre-commit Workflow

To avoid failing CI checks:

1. **Before committing**:
   ```bash
   # Format code
   black src/ tests/
   
   # Run tests
   pytest tests/ -v
   
   # Check linting
   flake8 src/ tests/
   ```

2. **Fix any issues** before pushing

3. **Commit and push** - CI will verify everything again

---

## Understanding Results

### Pipeline Status Indicators

- ✅ **Success (Green)**: All jobs completed successfully
- ❌ **Failure (Red)**: One or more jobs failed
- ⏳ **In Progress (Yellow)**: Pipeline is still running
- ⚠️ **Cancelled (Grey)**: Pipeline was cancelled (usually by user)

### Job Status

Each job can have one of these statuses:

- **Success**: Job completed without errors
- **Failure**: Job encountered errors and failed
- **Cancelled**: Job was cancelled before completion
- **Skipped**: Job was skipped (e.g., due to conditions)

### Interpreting Failures

#### Code Quality Failures

**Black Format Check Failed**:
```
would reformat src/eda/main.py
Oh no! 💥 💔 💥
```
- **Meaning**: Code doesn't match Black's formatting
- **Fix**: Run `black src/ tests/` to auto-format

**Flake8 Linting Failed**:
```
src/eda/main.py:45:1: E303 too many blank lines (3)
src/preprocessing/main.py:10:80: E501 line too long (120 > 100 characters)
```
- **Meaning**: Code style violations detected
- **Fix**: Address each error reported (line numbers included)

**Mypy Type Check**:
- Currently set to not fail the build (`|| true`)
- Check logs for warnings if type checking is important to you

#### Test Failures

**Pytest Failure**:
```
FAILED tests/test_data_loader.py::test_load_data - AssertionError: assert False
```
- **Meaning**: A test assertion failed
- **Fix**: 
  - Check the test output for details
  - Fix the code or update the test
  - Ensure dataset is available if tests require it

#### Script Execution Failures

**EDA Script Failed**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/bank/bank-full.csv'
```
- **Meaning**: Script can't find required files
- **Fix**: Check file paths and ensure data files exist

**Preprocessing Failed**:
```
ValueError: Input contains NaN, infinity or a value too large for dtype('float64')
```
- **Meaning**: Script encountered data processing errors
- **Fix**: Debug the preprocessing code and handle edge cases

### Common Error Patterns

1. **Import Errors**: Missing dependencies in `requirements.txt`
2. **File Not Found**: Missing data files or incorrect paths
3. **Syntax Errors**: Python syntax mistakes
4. **Test Failures**: Logic errors or outdated test expectations
5. **Timeout**: Jobs taking too long (rare, but possible)

---

## Troubleshooting

### Pipeline Not Running

**Problem**: Pipeline doesn't trigger on push/PR

**Solutions**:
- Check that `.github/workflows/ci.yml` exists
- Verify YAML syntax is valid (check for indentation errors)
- Ensure you're pushing to a branch (not just committing locally)
- Check GitHub Actions is enabled in repository settings

### Tests Fail Locally but Pass in CI (or vice versa)

**Problem**: Different behavior between local and CI environments

**Solutions**:
- Check Python version matches (CI uses 3.10)
- Verify all dependencies are installed locally
- Clear local caches: `rm -rf __pycache__ .pytest_cache`
- Ensure dataset is available in both environments

### Dataset Download Fails

**Problem**: CI can't download the dataset

**Solutions**:
- Check UCI ML repository URL is correct and accessible
- Verify network connectivity (rare issue)
- Consider using a cached dataset or alternative source
- Check if UCI repository structure has changed

### Type Checking Warnings

**Problem**: Mypy reports many warnings

**Solutions**:
- Current config ignores missing imports (third-party libraries)
- To enable stricter checking, modify `mypy.ini`
- Add type hints gradually to your code
- Use `# type: ignore` for specific lines if needed

### Slow Pipeline Execution

**Problem**: Pipeline takes too long

**Solutions**:
- Jobs run in parallel, so total time = longest job
- Consider caching dataset downloads (if frequently accessed)
- Optimize tests (remove unnecessary setup)
- Split large jobs into smaller ones

### "Permission Denied" Errors

**Problem**: Pipeline fails with permission errors

**Solutions**:
- Check workflow permissions (should be read-only by default)
- Verify file paths don't require special permissions
- Ensure GitHub Actions has necessary permissions in repo settings

---

## Customization

### Modifying the Pipeline

The pipeline configuration is in `.github/workflows/ci.yml`. You can customize:

#### Change Python Version

```yaml
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'  # Change from 3.10 to 3.11
```

#### Test Multiple Python Versions

```yaml
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
steps:
  - uses: actions/setup-python@v4
    with:
      python-version: ${{ matrix.python-version }}
```

#### Add New Check

Add a new step to any job:

```yaml
- name: Run custom check
  run: |
    python scripts/my_custom_check.py
```

#### Modify Trigger Events

```yaml
on:
  push:
    branches: ['main', 'develop']  # Only specific branches
  pull_request:
    branches: ['main']  # Only PRs to main
  schedule:
    - cron: '0 0 * * 0'  # Weekly on Sunday
```

#### Add Coverage Reporting

```yaml
- name: Install coverage
  run: pip install pytest-cov

- name: Run tests with coverage
  run: pytest tests/ --cov=src --cov-report=xml

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

#### Skip CI for Specific Commits

Add `[skip ci]` or `[ci skip]` to commit message:
```
git commit -m "Update README [skip ci]"
```

### Configuration Files

#### Mypy Configuration (`mypy.ini`)

Customize type checking behavior:

```ini
[mypy]
python_version = 3.10
warn_return_any = True
disallow_untyped_defs = True  # Stricter checking
```

#### Flake8 Configuration

Create `.flake8` file:

```ini
[flake8]
max-line-length = 100
extend-ignore = E203, W503
exclude = .git,__pycache__,build,dist
```

#### Black Configuration

Already in `pyproject.toml`:

```toml
[tool.black]
line-length = 100
target-version = ['py37', 'py38', 'py39', 'py310', 'py311']
```

### Environment Variables

Add secrets or environment variables:

```yaml
env:
  MY_SECRET: ${{ secrets.MY_SECRET }}
  CUSTOM_VAR: "value"

steps:
  - name: Use environment variable
    run: echo $MY_SECRET
```

---

## Best Practices

### Development Workflow

1. **Run Checks Locally First**
   - Don't rely only on CI to catch issues
   - Run `black`, `flake8`, and `pytest` before pushing
   - Saves time and reduces failed CI runs

2. **Small, Focused Commits**
   - Make small commits with clear messages
   - Easier to identify which change broke CI
   - Better for code review

3. **Fix CI Failures Promptly**
   - Don't let CI failures accumulate
   - Fix failures in the same PR that introduced them
   - Keep main branch always passing

4. **Write Tests**
   - Write tests for new features
   - Tests should be fast and reliable
   - Aim for good test coverage

5. **Keep Dependencies Updated**
   - Regularly update `requirements.txt`
   - Test updates before committing
   - Pin versions for reproducibility

### Code Quality

1. **Follow Style Guidelines**
   - Use Black for formatting (automatic)
   - Follow PEP 8 (enforced by flake8)
   - Add type hints when possible

2. **Code Review**
   - Use PRs for all changes (except trivial)
   - Require CI to pass before merging
   - Review code, not just CI status

3. **Documentation**
   - Document complex code
   - Update docs when changing behavior
   - Keep this guide updated

### CI/CD Specific

1. **Fast Feedback**
   - Keep pipeline execution time reasonable
   - Run fastest checks first (if sequential)
   - Use parallel jobs when possible

2. **Reliable Tests**
   - Make tests deterministic (no random failures)
   - Don't depend on external services if possible
   - Use fixtures for test data

3. **Clear Error Messages**
   - Write tests with descriptive names
   - Add helpful error messages in code
   - Document known issues

4. **Monitor CI Health**
   - Check CI status regularly
   - Investigate flaky tests
   - Update CI configuration as needed

### Security

1. **Don't Commit Secrets**
   - Use GitHub Secrets for sensitive data
   - Don't hardcode API keys or passwords
   - Review workflow files in PRs

2. **Minimal Permissions**
   - Use read-only permissions when possible
   - Only grant write access when needed
   - Review third-party actions before using

---

## Additional Resources

### GitHub Actions Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)

### Tool Documentation
- [Black Documentation](https://black.readthedocs.io/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Mypy Documentation](https://mypy.readthedocs.io/)

### Project Documentation
- `README.md` - Project overview and setup
- `QUICKSTART.md` - Quick start guide
- `docs/PROJECT_STRUCTURE.md` - Project structure details

---

## Questions or Issues?

If you encounter issues with the CI pipeline:

1. Check this guide's troubleshooting section
2. Review the workflow logs in GitHub Actions
3. Test locally with the same commands
4. Check tool documentation for specific errors
5. Open an issue in the repository if needed

---

**Last Updated**: Created with the initial CI/CD pipeline implementation

