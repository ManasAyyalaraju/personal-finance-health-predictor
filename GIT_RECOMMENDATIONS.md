# Git Push Recommendations

This document provides recommendations on what should and should not be pushed to git for this Personal Finance Health Predictor project.

## 📌 QUICK REFERENCE

| Item | Push? | Reason |
|------|-------|--------|
| **Source Code** | ✅ YES | All `.py` files in `deployment/app/`, `deployment/config/`, `deployment/tests/` |
| **Documentation** | ✅ YES | All `.md` files |
| **Notebooks** | ✅ YES | All `.ipynb` files |
| **Requirements** | ✅ YES | `requirements.txt`, `deployment/requirements.txt` |
| **Visualizations** | ✅ YES | PNG files in `results/` (images are small) |
| **Result CSVs** | ⚠️ CHECK SIZE | Small CSVs in `results/` are OK, large ones use LFS or exclude |
| **Virtual Envs** | ❌ NO | `venv/`, `deployment/venv/` |
| **Raw Data** | ❌ NO | `data/raw/` (large, may contain sensitive data) |
| **Processed Data** | ❌ NO | `data/processed/` (regenerable, large) |
| **Models** | ❌ NO | `models/` (regenerable, large binary files) |
| **Environment Files** | ❌ NO | `.env` files (may contain secrets) |
| **Cache Files** | ❌ NO | `__pycache__/`, `*.pyc` |
| **OS Files** | ❌ NO | `.DS_Store`, `Thumbs.db` |

## ✅ SHOULD BE PUSHED TO GIT

### Source Code & Configuration
- ✅ **All Python source files** (`*.py`)
  - `deployment/app/*.py` - API application code
  - `deployment/config/config.py` - Configuration (no secrets)
  - `deployment/tests/*.py` - Test files

### Documentation
- ✅ **All documentation files** (`*.md`)
  - `README.md` - Main project documentation
  - `docs/*.md` - All documentation files
  - `deployment/README.md` - Deployment documentation
  - `deployment/API_ENDPOINTS.md` - API documentation
  - `deployment/API_EXAMPLES.md` - API examples

### Dependency Files
- ✅ **Requirements files**
  - `requirements.txt` - Root project dependencies
  - `deployment/requirements.txt` - Deployment dependencies

### Notebooks
- ✅ **Jupyter notebooks** (`*.ipynb`)
  - `notebooks/*.ipynb` - All analysis notebooks
  - These are essential for reproducibility

### Project Structure
- ✅ **Empty directories** (with `.gitkeep` if needed)
  - `data/raw/` - Structure (but not contents)
  - `data/external/` - Structure (but not contents)
  - `dashboards/` - Structure (but not contents)

### Results & Visualizations
- ✅ **Analysis results** (small files)
  - `results/**/*.png` - Visualization images
  - `results/**/*.csv` - Analysis results (if small)
  - ⚠️ **Note**: Check file sizes - images should be fine, but large CSVs might need Git LFS

### Version Control Files
- ✅ `.gitignore` - Already properly configured
- ✅ `.gitattributes` - If using Git LFS (recommended for large files)

---

## ❌ SHOULD NOT BE PUSHED TO GIT

### Virtual Environments
- ❌ **`venv/`** - Root virtual environment
  - Already ignored by `.gitignore` ✅
- ❌ **`deployment/venv/`** - Deployment virtual environment
  - Already ignored by `.gitignore` ✅

### Python Cache & Compiled Files
- ❌ **`__pycache__/`** directories
  - Already ignored by `.gitignore` ✅
- ❌ **`*.pyc`, `*.pyo`, `*.pyd`** files
  - Already ignored by `.gitignore` ✅

### Data Files (Large & Regenerable)
- ❌ **`data/raw/`** - Raw datasets
  - Already ignored by `.gitignore` ✅
  - These are large and may contain sensitive data
  - **Recommendation**: Use Git LFS if you need to track them, or store externally

- ❌ **`data/processed/`** - Processed data files
  - Already ignored by `.gitignore` ✅
  - Can be regenerated from notebooks
  - Includes: `*.csv`, `*.pkl` files

### Model Files
- ❌ **`models/`** - Trained model files
  - Already ignored by `.gitignore` ✅
  - Can be regenerated from notebooks
  - Large binary files (`.pkl`)
  - **Recommendation**: Use Git LFS if you need version control for models

### Environment & Secrets
- ❌ **`.env`** files
  - Already ignored by `.gitignore` ✅
  - May contain API keys, secrets, credentials
- ❌ **`*.log`** files
  - Already ignored by `.gitignore` ✅

### IDE & Editor Files
- ❌ **`.vscode/`** - VS Code settings (if present)
  - Consider ignoring unless sharing team settings
- ❌ **`.idea/`** - PyCharm settings (if present)
- ❌ **`.cursor/`** - Cursor editor files (if present)

### OS-Specific Files
- ❌ **`.DS_Store`** (macOS)
- ❌ **`Thumbs.db`** (Windows)
- ❌ **`desktop.ini`** (Windows)

### Temporary & Build Files
- ❌ **`*.egg-info/`** - Package metadata
- ❌ **`dist/`, `build/`** - Build artifacts
- ❌ **`.pytest_cache/`** - Test cache
- ❌ **`.coverage`** - Coverage reports

---

## ⚠️ SPECIAL CONSIDERATIONS

### Large Files (>100MB)
GitHub has a **100MB file size limit**. Files larger than this will be rejected. Based on your README, you mentioned:
- Some files exceed 100MB
- 6 files exceed GitHub's hard limit

**Recommendations:**
1. **Use Git LFS** for large files if you need to track them:
   ```bash
   git lfs install
   git lfs track "*.pkl"
   git lfs track "*.csv"
   git lfs track "models/**"
   ```

2. **Or exclude them** (current approach - recommended):
   - Keep them ignored
   - Document how to regenerate them
   - Store externally (cloud storage, data registry)

### Sensitive Data
- ⚠️ **Raw datasets** may contain sensitive financial information
- ⚠️ **Never commit**:
  - API keys
  - Database credentials
  - Personal information
  - Production secrets

### Model Files Strategy
**Option 1: Ignore (Current - Recommended)**
- Models can be regenerated from notebooks
- Keeps repository lightweight
- Document regeneration process (already done in README)

**Option 2: Git LFS**
- If you need version control for models
- Useful for model versioning
- Requires Git LFS setup

**Option 3: External Storage**
- Store models in cloud storage (S3, GCS, Azure Blob)
- Reference in documentation
- Download during deployment

---

## 📋 CURRENT STATUS CHECK

Based on your `.gitignore`, you already have good coverage:

✅ **Already Ignored:**
- Virtual environments (`venv/`)
- Python cache (`__pycache__/`)
- Processed data (`data/processed/*.csv`, `data/processed/*.pkl`)
- Raw data (`data/raw/`)
- Model files (`models/`)
- Environment files (`.env`)

⚠️ **Consider Adding to `.gitignore`:**

```gitignore
# OS-specific files
.DS_Store
Thumbs.db
desktop.ini

# IDE files (optional - uncomment if needed)
# .vscode/
# .idea/

# Large result files (if any exceed reasonable size)
# results/**/*.csv  # Uncomment if CSV results are large

# Jupyter notebook checkpoints (already covered, but explicit)
.ipynb_checkpoints/
```

---

## 🎯 SUMMARY

### Push These:
1. ✅ All source code (`.py` files)
2. ✅ All documentation (`.md` files)
3. ✅ Requirements files (`requirements.txt`)
4. ✅ Jupyter notebooks (`.ipynb`)
5. ✅ Small visualization files (`results/**/*.png`)
6. ✅ Small CSV result files (`results/**/*.csv` - check sizes first)
7. ✅ Configuration files (without secrets)
8. ✅ Test files
9. ✅ `.gitkeep` files (to preserve empty directory structure)
10. ✅ `.gitignore` and `.gitattributes` files

### Don't Push These:
1. ❌ Virtual environments (`venv/`, `deployment/venv/`)
2. ❌ Raw datasets (`data/raw/` - especially large CSV files)
3. ❌ Processed data (`data/processed/` - all `.csv` and `.pkl` files)
4. ❌ Model files (`models/` - all `.pkl` files)
5. ❌ Environment files (`.env`)
6. ❌ Python cache (`__pycache__/`)
7. ❌ OS-specific files (`.DS_Store`, `Thumbs.db`, `desktop.ini`)

### Your Current Setup:
✅ **Excellent!** Your `.gitignore` is well-configured and follows best practices.

⚠️ **Note**: You have a `.gitattributes` file that uses Git LFS for large files, but your `.gitignore` also ignores these files. This is actually fine - the `.gitignore` takes precedence. If you want to use Git LFS for some files, you'll need to:
1. Remove those patterns from `.gitignore`
2. Ensure Git LFS is installed and initialized
3. Track the files with `git lfs track`

---

## 🚀 NEXT STEPS

### Initial Setup (if starting fresh):

1. **Add essential files first:**
   ```bash
   git add .gitignore .gitattributes
   git add README.md
   git add requirements.txt deployment/requirements.txt
   git add docs/
   git add notebooks/
   git add deployment/app/*.py
   git add deployment/config/*.py
   git add deployment/tests/
   git add deployment/README.md deployment/API_*.md
   git add results/**/*.png
   git add results/**/*.csv  # Only if files are small (<10MB)
   ```

2. **Review before committing:**
   ```bash
   git status
   ```

3. **Check file sizes** (Windows PowerShell):
   ```powershell
   Get-ChildItem -Recurse -File | Where-Object {$_.Length -gt 10MB} | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}
   ```

4. **Verify** no sensitive/large files are staged:
   ```powershell
   git status --short | Select-String -Pattern "\.(env|log|pkl|csv|pkl)$"
   ```

### If Files Are Already Tracked (Cleanup):

1. **Remove accidentally tracked files:**
   ```bash
   git rm --cached -r venv/
   git rm --cached -r deployment/venv/
   git rm --cached -r data/raw/
   git rm --cached -r data/processed/
   git rm --cached -r models/
   git rm --cached __pycache__/
   git rm --cached deployment/app/__pycache__/
   ```

2. **Commit the cleanup:**
   ```bash
   git commit -m "Remove large files and build artifacts from tracking"
   ```

### Recommended First Commit:

```bash
# Stage essential files
git add .gitignore .gitattributes
git add README.md requirements.txt
git add docs/ notebooks/
git add deployment/app/*.py deployment/config/*.py deployment/tests/
git add deployment/README.md deployment/requirements.txt
git add deployment/API_*.md
git add results/**/*.png

# Review
git status

# Commit
git commit -m "Initial commit: Add source code, documentation, and notebooks"
```

### Verify Before Pushing:

1. **Check repository size:**
   ```bash
   git count-objects -vH
   ```

2. **Ensure no large files:**
   - No files > 100MB (GitHub limit)
   - Ideally keep files < 50MB for better performance

3. **Test clone** (optional - test in a different directory):
   ```bash
   cd ..
   git clone <your-repo-url> test-clone
   ```

