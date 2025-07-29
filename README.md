
# RepayCheck - Your Loan Repayment Eligibility Checker

A fairness-aware machine learning web application to predict loan repayment eligibility. Built with Flask, MySQL, SQLAlchemy, Docker, and LightGBM. Includes both admin and user features.

---

## Features

- Public loan eligibility checker 
- Credit score estimator
- Secure admin login system with functions like model retraining, model prediction, downloadable prediction and retrain results
- Secure admin debug panel

---

## Tech Stack

- **Backend**: Flask + SQLAlchemy + Flask-Login
- **ML Models**: LightGBM + Fairlearn + SHAP
- **Database**: MySQL
- **Containerization**: Docker + Docker Compose
- **Frontend**: HTML, CSS, JS 
- **Security**: CSRF + `.env`

---

## Installation Guide

### 1. Prerequisite
- Windows 10/11 with WSL 2 enabled
- [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Visual Studio Code 


### 2. Open Visual Code Studio and run the command 
```bash
git clone https://github.com/wanxuen/FYP2025/
```
in terminal (CTRL+j). or Open the code files in Visual Code Studio. 
### 3. Setup .env file
-	Open terminal in VS Code (CTRL+j) and launch python shell
 ```bash
    python
    import secrets
    secrets.token_hex(32)
```
-   Run twice for ‘secrets.token_hex(32)’ to generate 2 different secret keys
-   Create a file named .env and copy the following details into that file:
 ```bash
    SECRET_KEY=<your first generated token>
    WTF_CSRF_SECRET_KEY=<your second generated token>
    FLASK_DEBUG=true
    PORT=5000

    MYSQL_USER=fyp
    MYSQL_PASSWORD=fyp
    MYSQL_DATABASE=fyp2025
    MYSQL_HOST=db
    MYSQL_PORT=3306

    TEST_ADMIN_USERNAME=admin
    TEST_ADMIN_PASSWORD=admin123
```

### 4.	Configure WSL2 Resources 
-	This step helps to increase resource limits to avoid memory issues on Windows with Docker + WSL 2 backend. 
1.	Open Notepad as Administrator
2.	Paste the following:
```bash
[wsl2]
memory=16GB
processors=4
swap=4GB
```
3.	Save it as: C:\Users\<YourUsername>\.wslconfig 
4.	Run wsl --shutdown in PowerShell (Admin) 
5.	Restart Docker Desktop

### 5.	Setup admin account 
-	Open terminal in VS Code (CTRL+j) and enter python shell
-	Enter following commands:
-	Run the App with Docker by executing 
```bash
    docker-compose up --build
```
in the terminal.(This may take some time)
-	In another terminal, run: 
```bash
    docker exec -it flask-app bash
```
to access the Flask container shell
-	Python to start Python shell inside the container
-	Run the admin setup code
```bash
    from app import create_admin
    create_admin()
```
-	You will be prompted to input a username and password. The account will be securely saved to the database.

### 6.	Access the web app at http://127.0.0.1:5000 
### 7.	If you want to terminate the app,
```bash
    docker compose stop
```
- You may also use crtl+c to exit the app in the terminal if this is the first time setting up the app. 
### 8.	To restart the App after setup, run 
```bash
    docker-compose restart
```

**Notes:**
- Docker must be running when starting the app.
- The admin account only needs to be created once unless the database is reset.
- If it shows the error 'Cannot found file start.sh', you may recreate the file with the same content. 

