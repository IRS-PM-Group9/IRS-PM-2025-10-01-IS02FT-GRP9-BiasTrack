# 💡BiasTrack 📈

## 📝 Project Description

BiasTrack is designed to help organizations detect and mitigate **gender pay gaps** in employee datasets. Its main features include:

1. **Detect and Reason:** Automatically detects gender pay gaps in the dataset and provides reasoning behind these disparities.  
2. **Salary Adjuster:** Provides HR with an interface to modify employee salaries and immediately visualize the impact of these changes on the pay gap.  
3. **Constraint-based Optimizations and Recommendations:** Allows HR to input constraints such as maximum budget, hierarchy, or department, and generates optimized recommendations for salaries that align with these constraints.

BiasTrack combines **data analysis, reasoning, and optimization techniques** to provide actionable insights and decision support for HR departments aiming to ensure fair and equitable compensation.

This project is developed as part of the Practise Module for the **Intelligent Reasoning System Graduate Certification (MTech AIS) at NUS**.

### 👩‍💻 Contributors:

Arshi Saxena

Pranjali Rajendra Sonawane

## 🛠️ Project Setup

This project uses **Python 3.12.7** and requires a virtual environment to manage dependencies. Follow these steps to set up the project locally:

### 1. Clone the repository
`git clone https://github.com/IRS-PM-Group9/IRS-PM-2025-10-01-IS02FT-GRP9-BiasTrack.git`

### 2. Python Version
Ensure you have python version **Python 3.12.x** installed before proceeding with the next steps.

### 3. Run the setup script
`cd <your_project_path>/BiasTrack`

#### Windows
`.\setup_env.bat`

#### macOS/Linux
`bash setup_env.sh`

### 4. Activate the virtual environment

This project uses **VS Code auto-activation** as configured in `.vscode/settings.json`.  

- **If you are using VS Code:**  
  Ensure that Python extension is installed and enabled. Opening a new terminal(cmd) in VS Code for this project will automatically activate the virtual environment, and it will automatically deactivate when you close VS Code.
  **Check:** (.venv) should be appended to the path in cmd terminal in VS Code for successful auto-activation
  `(.venv) path\BiasTrack>`

- **If you are using any other IDE or terminal:**  
  You will need to manually activate the environment **each time** you open the project:

    #### Windows - Powershell
    `.venv\Scripts\activate.bat`

    #### macOS/Linux
    `source .venv/bin/activate`

### 5. Start the Frontend (Streamlit)
In a **new terminal** (while keeping the backend running):
`streamlit run frontend/streamlit_app.py`
  - The frontend will run at: http://localhost:8501

### Model Training
From Project Root: `python -m src.biastrack.train.model_v1.main`
From Project Root: `python -m src.biastrack.train.model_v2.main`

### Notebooks
Notebooks are used for experimentation and validation in development phase.
They are run using Jupyter.
