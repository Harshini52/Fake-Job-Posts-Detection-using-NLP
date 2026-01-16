
# Fake Job Posts Detection using NLP

## Project Overview
Fake job postings pose serious risks to job seekers by spreading scams and misinformation.  
This project is an end-to-end Fake Job Post Detection System that uses Natural Language Processing (NLP) and Machine Learning to classify job postings as Real or Fake, with secure authentication and an admin monitoring dashboard.
The system covers the complete pipeline from data preprocessing and model training to web deployment with SQLite-based authentication.


##  Project Modules

### 🔹 Module 1: Data Preprocessing & Exploration
- Loaded and cleaned the job postings dataset
- Handled missing values and normalized text data
- Performed Exploratory Data Analysis (EDA) including:
  - Word frequency analysis
  - Word clouds
  - Identification of common scam-related keywords
- Prepared features using:
  - TF-IDF Vectorization
  - Basic NLP pipelines (tokenization, stopword removal)

### 🔹 Module 2: Model Training
- Trained baseline Machine Learning models for text classification
- Evaluated models using:
  - Cross-validation
  - Accuracy, Precision, Recall, Recall, and F1-Score
- Selected the best-performing model for deployment
- Saved the trained model for real-time inference

### 🔹 Module 3: Web Application Integration
- Built a user-friendly web interface where users can:
  - Paste job descriptions into an input form
  - Submit data for prediction
- Displayed prediction results as:
  - **Real / Fake classification**
  - **Confidence percentage**
- Integrated backend ML model with frontend UI using Flask

### 🔹 Module 4: Authentication, Admin Panel & Dashboard
- Implemented **user authentication using SQLite database**
- Secured routes using **JWT-based authentication**
- Designed an admin dashboard to:
  - View flagged (fake) job posts
  - Monitor daily prediction logs
  - Track registered users and system activity
- Enabled admin-only features:
  - Promote or demote users
  - Export prediction data
  - Retrain the model through dashboard controls

## 🔐 Authentication & Security
- User credentials are securely stored in **SQLite**
- Passwords are hashed before storage
- JWT tokens are used to protect user and admin routes
- Role-based access control ensures only admins can access sensitive features

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Framework:** Flask  
- **Database:** SQLite  
- **Libraries:** scikit-learn, pandas, numpy, nltk  
- **Frontend:** HTML, CSS, JavaScript  
- **Modeling:** NLP, TF-IDF, Machine Learning  
- **Authentication:** JWT (JSON Web Tokens)

## 🚀 How to Run the Project

1. Clone the repository:
   git clone https://github.com/Harshini52/Fake-Job-Posts-Detection-using-NLP.git

2. Navigate to the project directory:
   cd Fake-Job-Posts-Detection-using-NLP
   
3. Install required dependencies:
   pip install -r requirements.txt
   
4. Run the Flask application:
   python app.py
   
6. Open your browser and visit:
   http://127.0.0.1:5000/

##  Key Features
* Real-time fake job post detection
* Secure login and role-based access
* Admin monitoring and analytics dashboard
* Scalable and modular architecture

## 🔮 Future Enhancements

* Integration of deep learning models (LSTM / BERT)
* Advanced analytics and visual dashboards
* Email alerts for high-risk job postings

## 👩‍💻 Author

**Harshini Rachcha**
Project developed as part of internship milestone submission.




