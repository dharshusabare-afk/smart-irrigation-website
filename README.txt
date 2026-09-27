# Smart Water Management and Irrigation Recommendation

## Run in VS Code

### 1. Open this folder
Open `smart_irrigation_website` in VS Code.

### 2. Open terminal
VS Code -> Terminal -> New Terminal

### 3. Create virtual environment
Windows:
```bash
python -m venv venv
```

### 4. Activate it
PowerShell:
```bash
venv\Scripts\Activate.ps1
```

If PowerShell blocks it, use Command Prompt:
```cmd
venv\Scripts\activate
```

### 5. Install packages
```bash
pip install -r requirements.txt
```

### 6. Start the website
```bash
python app.py
```

### 7. Open browser
Go to:
http://127.0.0.1:5000

The model is trained automatically from `irrigation_prediction.csv`.
