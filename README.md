# 🚀 MyProject
AI 中医助理.

## 📌 Table of Contents
- [📥 Pre-installation](#pre-installation)
- [📥 Installation](#installation)

## **📥 Pre-installation**
Before installing this project, ensure your system meets the following requirements:

### **1️⃣ Install Python 3.9**
If not installed, download and install it from:
🔗 Python 3.9 Download(https://www.python.org/downloads/release/python-390/) 
This project requires **Python 3.9**. Check your Python version:
```sh
python --version
```

2️⃣ Create a Virtual Environment (venv)

Using a virtual environment ensures isolated dependencies.

💻 Create a virtual environment
```sh
python -m venv myenv
```
🔹 Activate the virtual environment

•	Windows (CMD/PowerShell)
```sh
myenv\Scripts\activate
```
•	macOS/Linux
```sh
source myenv/bin/activate
```

## 📥 Installation

Once you have Python 3.9, venv activated, and Hugging Face installed, follow these steps:

1️⃣ Clone the repository
```sh
git clone git@github.com:pony1179/traditional_chinese_medicine_llm_app.git
cd traditional_chinese_medicine_llm_app
```
2️⃣ Activate virtual environment
```sh
source myenv/bin/activate  # macOS/Linux
myenv\Scripts\activate      # Windows (CMD/PowerShell)
```
3️⃣ Install project dependencies
```sh
pip install -r requirements.txt
```
4️⃣ Start the services
```sh
python index2.py
```
5️⃣ Enter the tcm-frontend
```sh
npm install
npm run dev
```