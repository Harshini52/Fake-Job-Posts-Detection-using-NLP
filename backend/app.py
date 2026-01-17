from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, send_file
import os, sqlite3, uuid, json, csv, io, secrets, time, smtplib
from datetime import datetime, timedelta
from functools import wraps
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import joblib
import numpy as np
from scipy.sparse import hstack
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from backend.scam_keywords import scam_keywords
from backend.ocr_utils import extract_text_from_image
from backend import job_predict


# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "fakejob_secret_key_secure_2025"

UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = "database.db"

# -------------------------------------------------
# EMAIL CONFIGURATION
# -------------------------------------------------
# Configure your email settings here
EMAIL_CONFIG = {
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'SMTP_USERNAME': 'harshini5256@gmail.com',  # Your Gmail address
    'SMTP_PASSWORD': 'lobf hxuy hwgo gwpy',  # App password from Google
    'FROM_EMAIL': 'harshini5256@gmail.com',
    'FROM_NAME': 'JobGuard System'
}

# Test mode - if True, codes will be printed to console instead of email
TEST_MODE = False  # Set to False for real emails

# -------------------------------------------------
# PASSWORD RESET HELPERS
# -------------------------------------------------

# In-memory storage for password reset codes
reset_codes = {}

def generate_reset_code(email):
    """Generate a 6-digit reset code"""
    code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
    reset_codes[email] = {
        'code': code,
        'timestamp': time.time(),
        'verified': False
    }
    return code

def send_reset_email(email, code):
    """Send reset code email"""
    if TEST_MODE:
        # Test mode - print to console
        print(f"🔐 PASSWORD RESET REQUEST")
        print(f"📧 Email: {email}")
        print(f"🔢 Reset Code: {code}")
        print(f"⏰ This code expires in 15 minutes")
        print(f"📝 Note: In production mode, this would be sent via email")
        print("-" * 40)
        return True
    
    try:
        # Real email sending
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['FROM_NAME']} <{EMAIL_CONFIG['FROM_EMAIL']}>"
        msg['To'] = email
        msg['Subject'] = "JobGuard - Password Reset Code"
        
        # Email body
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                <h2 style="color: #ff6b8b; text-align: center;">JobGuard Password Reset</h2>
                <p>Hello,</p>
                <p>You have requested to reset your password for your JobGuard account.</p>
                
                <div style="background: #f8f9ff; padding: 20px; border-radius: 10px; text-align: center; margin: 20px 0;">
                    <h3 style="color: #6a67ce; margin: 0;">Your Reset Code</h3>
                    <div style="font-size: 32px; font-weight: bold; color: #ff6b8b; letter-spacing: 10px; margin: 15px 0;">
                        {code}
                    </div>
                    <p style="font-size: 14px; color: #666;">
                        Enter this 6-digit code in the password reset page.
                    </p>
                </div>
                
                <p><strong>Important:</strong></p>
                <ul>
                    <li>This code will expire in <strong>15 minutes</strong></li>
                    <li>If you didn't request this reset, please ignore this email</li>
                    <li>For security, don't share this code with anyone</li>
                </ul>
                
                <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee; text-align: center; color: #888; font-size: 12px;">
                    <p>This is an automated message from JobGuard Fake Job Detection System.</p>
                    <p>© {datetime.now().year} JobGuard. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'])
        server.starttls()
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Reset email sent to {email}")
        return True
        
    except Exception as e:
        print(f"❌ Error sending email to {email}: {e}")
        # Fall back to test mode
        print(f"📧 Falling back to console display. Code: {code}")
        return False

def verify_reset_code(email, code):
    """Verify if the reset code is valid"""
    if email not in reset_codes:
        return False
    
    reset_data = reset_codes[email]
    
    # Check if code matches
    if reset_data['code'] != code:
        return False
    
    # Check if code is expired (15 minutes)
    if time.time() - reset_data['timestamp'] > 15 * 60:
        del reset_codes[email]
        return False
    
    reset_data['verified'] = True
    return True

def is_reset_verified(email):
    """Check if reset is verified"""
    if email in reset_codes and reset_codes[email].get('verified'):
        return True
    return False

def clear_reset_code(email):
    """Clear reset code after use"""
    if email in reset_codes:
        del reset_codes[email]

# -------------------------------------------------
# DATABASE
# -------------------------------------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def check_and_fix_database():
    """Check if database schema is up-to-date and fix if needed"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if predictions table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not cursor.fetchone():
        print("Predictions table doesn't exist. It will be created in init_db().")
        conn.close()
        return
    
    # Check if predictions table has is_flagged column
    cursor.execute("PRAGMA table_info(predictions)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'is_flagged' not in columns:
        print("Adding missing column 'is_flagged' to predictions table...")
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN is_flagged INTEGER DEFAULT 0")
            conn.commit()
            print("Column 'is_flagged' added successfully.")
        except Exception as e:
            print(f"Error adding column: {e}")
    
    conn.close()

def run_database_migrations():
    """Run database migrations to update schema"""
    conn = get_db()
    cursor = conn.cursor()
    
    migrations = [
        # Migration 1: Add is_flagged column if missing
        ("ALTER TABLE predictions ADD COLUMN is_flagged INTEGER DEFAULT 0", 
         "is_flagged"),
        
        # Migration 2: Add flag_reason column if missing
        ("ALTER TABLE predictions ADD COLUMN flag_reason TEXT", 
         "flag_reason"),
    ]
    
    for sql, column_name in migrations:
        try:
            # Check if table exists first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
            if not cursor.fetchone():
                print(f"Predictions table doesn't exist, skipping migration for {column_name}")
                continue
                
            # Check if column exists
            cursor.execute("PRAGMA table_info(predictions)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if column_name not in columns:
                print(f"Running migration: Adding {column_name} column...")
                cursor.execute(sql)
                conn.commit()
                print(f"Added {column_name} column successfully.")
        except Exception as e:
            print(f"Error running migration for {column_name}: {e}")
    
    conn.close()

def init_db():
    """Initialize database with basic tables and admin user"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create predictions table with flag field
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            scan_id TEXT,
            method TEXT,
            result TEXT,
            confidence REAL,
            risk_score REAL,
            job_text TEXT,
            is_flagged INTEGER DEFAULT 0,
            flag_reason TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Create login_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Create model_logs table for retraining
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (admin_id) REFERENCES users(id)
        )
    ''')
    
    # Check if admin user exists, if not create one
    cursor.execute("SELECT * FROM users WHERE email='admin@jobguard.com'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (first_name, last_name, email, password, is_admin)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            "Admin",
            "User",
            "admin@jobguard.com",
            generate_password_hash("admin123"),
            1
        ))
        print("Created default admin user: admin@jobguard.com / admin123")
    
    # Create some test users if they don't exist
    test_users = [
        ("John", "Doe", "john@example.com", "password123"),
        ("Jane", "Smith", "jane@example.com", "password123"),
        ("Bob", "Johnson", "bob@example.com", "password123"),
        ("Alice", "Williams", "alice@example.com", "password123"),
    ]
    
    for first_name, last_name, email, password in test_users:
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        if not cursor.fetchone():
            cursor.execute('''
                INSERT INTO users (first_name, last_name, email, password, is_admin)
                VALUES (?, ?, ?, ?, ?)
            ''', (first_name, last_name, email, generate_password_hash(password), 0))
            print(f"Created test user: {email}")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")
    
    # Run migrations to ensure schema is up-to-date
    run_database_migrations()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
try:
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    model = joblib.load("svm_model.pkl")
    print("ML model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load ML model files: {e}")

# -------------------------------------------------
# DECORATORS
# -------------------------------------------------
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            flash("Please login first", "danger")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

def admin_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if not session.get("is_admin"):
            flash("Admin access only", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return wrap

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def count_scam_keywords(text):
    text = text.lower()
    return sum(1 for kw in scam_keywords if kw in text)

def predict_job(text):
    text_vec = vectorizer.transform([text])
    scam_count = count_scam_keywords(text)
    final_features = hstack([text_vec, np.array([[scam_count]])])
    pred = model.predict(final_features)[0]

    prediction = "Fake Job" if pred == 1 else "Legitimate Job"
    confidence = max(10, min(99, 85 - scam_count * 5))
    risk_score = min(100, scam_count * 20)

    return prediction, confidence, risk_score

def row_to_dict(row):
    """Convert sqlite3.Row to dictionary"""
    if row is None:
        return {}
    return dict(row)

def format_datetime(dt_string):
    """Format datetime string for display"""
    try:
        if isinstance(dt_string, str):
            for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%Y-%m-%d'):
                try:
                    dt = datetime.strptime(dt_string, fmt)
                    break
                except ValueError:
                    continue
            else:
                return "Unknown", ""
        elif isinstance(dt_string, datetime):
            dt = dt_string
        else:
            return "Unknown", ""
        
        return dt.strftime('%Y-%m-%d'), dt.strftime('%I:%M %p')
    except:
        return "Unknown", ""

# -------------------------------------------------
# ROUTES
# -------------------------------------------------
@app.route("/")
def root():
    return redirect(url_for("login"))

# ---------------- LOGIN ----------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = get_db()
        user = conn.execute(
            "SELECT * FROM users WHERE email=?", (email,)
        ).fetchone()

        if user:
            user_dict = row_to_dict(user)
            
            if check_password_hash(user_dict["password"], password):
                # Update user session
                session["user_id"] = user_dict["id"]
                session["email"] = user_dict["email"]
                session["first_name"] = user_dict["first_name"]
                session["is_admin"] = bool(user_dict.get("is_admin", 0))
                
                # Record login history
                conn.execute(
                    "INSERT INTO login_history (user_id) VALUES (?)",
                    (user_dict["id"],)
                )
                conn.commit()
                conn.close()

                flash(f"Welcome back, {user_dict['first_name']}!", "success")
                
                # Redirect based on user role
                if user_dict.get("is_admin"):
                    return redirect(url_for("admin_dashboard"))
                else:
                    return redirect(url_for("index"))

        conn.close()
        flash("Invalid email or password", "danger")

    return render_template("login.html")

# ---------------- REGISTER ----------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        conn = get_db()
        try:
            conn.execute("""
                INSERT INTO users (first_name, last_name, email, password, is_admin)
                VALUES (?, ?, ?, ?, 0)
            """, (
                request.form["first_name"],
                request.form["last_name"],
                request.form["email"],
                generate_password_hash(request.form["password"])
            ))
            conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already exists", "danger")
        finally:
            conn.close()

    return render_template("register.html")

# ---------------- FORGOT PASSWORD API ROUTES ----------------
@app.route("/api/forgot-password", methods=["POST"])
def api_forgot_password():
    """API endpoint for sending reset code"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        
        if not email:
            return jsonify({"success": False, "message": "Email is required"}), 400
        
        # Check if user exists
        conn = get_db()
        user = conn.execute(
            "SELECT id, email, first_name FROM users WHERE email=?", (email,)
        ).fetchone()
        conn.close()
        
        if not user:
            # For security, don't reveal if user exists or not
            return jsonify({"success": True, "message": "If an account exists, a reset code has been sent"})
        
        # Generate and send reset code
        code = generate_reset_code(email)
        email_sent = send_reset_email(email, code)
        
        if not email_sent and not TEST_MODE:
            return jsonify({"success": False, "message": "Failed to send email. Please try again later."}), 500
        
        # In test mode, return the code for debugging
        if TEST_MODE:
            return jsonify({
                "success": True, 
                "message": "Reset code generated (Test Mode)",
                "debug_code": code,
                "note": "In production, this code would be sent via email"
            })
        
        return jsonify({"success": True, "message": "Reset code sent to your email"})
        
    except Exception as e:
        print(f"Error in forgot-password: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route("/api/verify-reset-code", methods=["POST"])
def api_verify_reset_code():
    """API endpoint for verifying reset code"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        code = data.get('code', '').strip()
        
        if not email or not code:
            return jsonify({"success": False, "message": "Email and code are required"}), 400
        
        if verify_reset_code(email, code):
            return jsonify({"success": True, "message": "Code verified"})
        else:
            return jsonify({"success": False, "message": "Invalid or expired code"}), 400
            
    except Exception as e:
        print(f"Error in verify-reset-code: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route("/api/reset-password", methods=["POST"])
def api_reset_password():
    """API endpoint for resetting password"""
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        new_password = data.get('new_password', '').strip()
        
        if not email or not new_password:
            return jsonify({"success": False, "message": "Email and password are required"}), 400
        
        # Check if reset is verified
        if not is_reset_verified(email):
            return jsonify({"success": False, "message": "Reset not verified"}), 400
        
        # Validate password strength
        if len(new_password) < 8:
            return jsonify({"success": False, "message": "Password must be at least 8 characters"}), 400
        
        # Check if user exists
        conn = get_db()
        user = conn.execute(
            "SELECT id FROM users WHERE email=?", (email,)
        ).fetchone()
        
        if not user:
            conn.close()
            return jsonify({"success": False, "message": "User not found"}), 404
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        conn.execute(
            "UPDATE users SET password=? WHERE email=?",
            (hashed_password, email)
        )
        conn.commit()
        conn.close()
        
        # Clear reset code after successful reset
        clear_reset_code(email)
        
        print(f"✅ Password reset successful for {email}")
        return jsonify({"success": True, "message": "Password reset successful"})
        
    except Exception as e:
        print(f"Error in reset-password: {e}")
        return jsonify({"success": False, "message": "Server error"}), 500

@app.route("/test-reset-api")
def test_reset_api():
    """Test route to verify reset API is working"""
    return jsonify({
        "status": "Reset API is active",
        "test_mode": TEST_MODE,
        "email_config_configured": bool(EMAIL_CONFIG['SMTP_USERNAME'] != 'your-email@gmail.com'),
        "endpoints": [
            "/api/forgot-password (POST)",
            "/api/verify-reset-code (POST)",
            "/api/reset-password (POST)"
        ]
    })

# ---------------- EMAIL CONFIG TEST ----------------
@app.route("/test-email")
def test_email():
    """Test email configuration"""
    if TEST_MODE:
        return jsonify({
            "status": "Test Mode Active",
            "message": "Emails are printed to console instead of being sent",
            "to_enable_emails": "Set TEST_MODE = False in app.py and configure EMAIL_CONFIG"
        })
    
    # Try to send a test email
    try:
        test_email_address = "test@example.com"
        test_code = "123456"
        
        # Test connection
        server = smtplib.SMTP(EMAIL_CONFIG['SMTP_SERVER'], EMAIL_CONFIG['SMTP_PORT'])
        server.starttls()
        server.login(EMAIL_CONFIG['SMTP_USERNAME'], EMAIL_CONFIG['SMTP_PASSWORD'])
        server.quit()
        
        return jsonify({
            "status": "Email configuration successful",
            "smtp_server": EMAIL_CONFIG['SMTP_SERVER'],
            "from_email": EMAIL_CONFIG['FROM_EMAIL'],
            "note": "Connection test passed. Emails will be sent when resetting passwords."
        })
        
    except Exception as e:
        return jsonify({
            "status": "Email configuration failed",
            "error": str(e),
            "help": "Check your email credentials and make sure 'Allow less secure apps' is enabled for Gmail"
        }), 500

# ---------------- INDEX ----------------
@app.route("/index")
@login_required
def index():
    return render_template("index.html")

# ---------------- DETECT ----------------
@app.route("/detect", methods=["POST"])
@login_required
def detect():
    detection_type = request.form.get("type")
    scan_id = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------- TEXT DETECTION --------
    if detection_type == "text":
        text = request.form.get("job_text", "").strip()
        if not text:
            flash("Please enter job description text", "danger")
            return redirect(url_for("index"))
        method = "Text Analysis"

    # -------- IMAGE DETECTION --------
    elif detection_type == "image":
        file = request.files.get("image")
        if not file or file.filename == "":
            flash("Please upload an image", "danger")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        text = extract_text_from_image(image_path)
        if not text.strip():
            flash("Could not extract text from image", "danger")
            return redirect(url_for("index"))

        method = "OCR Image Analysis"

    else:
        flash("Invalid detection type", "danger")
        return redirect(url_for("index"))

    # -------- ML PREDICTION --------
    prediction, confidence, risk_score = predict_job(text)

    # -------- RESULT DATA (FOR TEMPLATE) --------
    result_data = {
        "scan_id": scan_id,
        "timestamp": timestamp,
        "method": method,
        "result": prediction,
        "confidence": round(confidence, 1),
        "score": round(risk_score, 1),
        "text_preview": text[:500] + "..." if len(text) > 500 else text
    }

    # -------- SAVE TO DATABASE --------
    conn = get_db()
    conn.execute("""
        INSERT INTO predictions
        (user_id, scan_id, method, result, confidence, risk_score, job_text)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        session["user_id"],
        scan_id,
        method,
        prediction,
        confidence,
        risk_score,
        text
    ))
    conn.commit()
    conn.close()

    # -------- RENDER RESULT --------
    return render_template(
        "results.html",
        result_data=result_data,
        is_fake=prediction == "Fake Job",
        is_suspicious=40 < risk_score < 70
    )

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    conn = get_db()
    history = conn.execute("""
        SELECT 
            id,
            scan_id,
            result, 
            confidence, 
            risk_score as score,
            created_at as timestamp,
            job_text as text_preview
        FROM predictions
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT 50
    """, (session["user_id"],)).fetchall()

    total = len(history)
    fake = 0
    
    # Format timestamps properly
    formatted_history = []
    for row in history:
        row_dict = row_to_dict(row)
        
        # Count fake jobs
        if row_dict.get('result') == 'Fake Job':
            fake += 1
        
        # Ensure timestamp is properly formatted
        timestamp = row_dict.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    dt = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    row_dict['timestamp'] = dt.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(timestamp, datetime):
                    row_dict['timestamp'] = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                row_dict['timestamp'] = str(timestamp)
        else:
            row_dict['timestamp'] = 'Unknown'
        
        # Add word count
        text = row_dict.get('text_preview', '')
        row_dict['word_count'] = len(text.split())
        
        formatted_history.append(row_dict)
    
    conn.close()

    return render_template(
        "dashboard.html",
        history=formatted_history,
        total_scans=total,
        fake_count=fake,
        legit_count=total - fake
    )

# ---------------- ADMIN DASHBOARD ----------------
@app.route("/admin")
@admin_required
def admin_dashboard():
    conn = get_db()
    
    try:
        # Get basic stats
        total_users = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()[0]
        total_admins = conn.execute("SELECT COUNT(*) as count FROM users WHERE is_admin=1").fetchone()[0]
        total_predictions = conn.execute("SELECT COUNT(*) as count FROM predictions").fetchone()[0]
        fake_predictions = conn.execute("SELECT COUNT(*) as count FROM predictions WHERE result='Fake Job'").fetchone()[0]
        real_predictions = conn.execute("SELECT COUNT(*) as count FROM predictions WHERE result='Legitimate Job'").fetchone()[0]
        
        # Safely get flagged predictions count
        try:
            flagged_predictions = conn.execute("SELECT COUNT(*) as count FROM predictions WHERE is_flagged=1").fetchone()[0]
        except sqlite3.OperationalError:
            # Column doesn't exist, run migrations
            run_database_migrations()
            flagged_predictions = 0
            print("Database schema updated - added missing columns")
        
        # Today's predictions
        today = datetime.now().strftime('%Y-%m-%d')
        today_predictions = conn.execute("""
            SELECT COUNT(*) as count FROM predictions 
            WHERE DATE(created_at) = ?
        """, (today,)).fetchone()[0]
        
        # Calculate rates
        fake_rate = round((fake_predictions / total_predictions * 100), 1) if total_predictions > 0 else 0
        real_rate = round((real_predictions / total_predictions * 100), 1) if total_predictions > 0 else 0
        
        # Get average confidence
        avg_confidence_result = conn.execute("SELECT AVG(confidence) as avg FROM predictions").fetchone()
        avg_confidence = round(float(avg_confidence_result[0]) if avg_confidence_result and avg_confidence_result[0] else 0, 1)
        
        # Get all users with their details
        users = conn.execute("""
            SELECT 
                id, 
                first_name, 
                last_name, 
                email, 
                is_admin, 
                created_at
            FROM users 
            ORDER BY created_at DESC
        """).fetchall()
        
        # Format user data
        formatted_users = []
        for user in users:
            user_dict = row_to_dict(user)
            
            # Format created date
            created_at = user_dict.get('created_at')
            created_date, created_time = format_datetime(created_at)
            
            # Get login count from login_history
            login_count_result = conn.execute(
                "SELECT COUNT(*) as count FROM login_history WHERE user_id=?", 
                (user_dict['id'],)
            ).fetchone()
            login_count = login_count_result[0] if login_count_result else 0
            
            # Get last login time
            last_login_result = conn.execute(
                "SELECT MAX(login_time) as last_login FROM login_history WHERE user_id=?", 
                (user_dict['id'],)
            ).fetchone()
            last_login = last_login_result[0] if last_login_result else None
            
            # Format last login
            if last_login:
                last_login_date, last_login_time = format_datetime(last_login)
            else:
                last_login_date, last_login_time = "Never", ""
            
            # Get user predictions count
            user_predictions = conn.execute(
                "SELECT COUNT(*) as count FROM predictions WHERE user_id=?", 
                (user_dict['id'],)
            ).fetchone()
            prediction_count = user_predictions[0] if user_predictions else 0
            
            # Prepare user data for template
            formatted_user = {
                'id': user_dict['id'],
                'first_name': user_dict.get('first_name', 'User'),
                'last_name': user_dict.get('last_name', ''),
                'email': user_dict.get('email', 'No email'),
                'is_admin': user_dict.get('is_admin', 0),
                'login_count': login_count,
                'prediction_count': prediction_count,
                'last_login_date': last_login_date,
                'last_login_time': last_login_time,
                'created_date': created_date,
                'created_time': created_time
            }
            
            formatted_users.append(formatted_user)
        
        # Safely get flagged posts
        try:
            flagged_posts = conn.execute("""
                SELECT 
                    p.id,
                    p.scan_id,
                    p.result,
                    p.confidence,
                    p.risk_score as score,
                    p.job_text,
                    p.flag_reason,
                    p.created_at,
                    u.first_name,
                    u.last_name,
                    u.email
                FROM predictions p
                JOIN users u ON p.user_id = u.id
                WHERE p.is_flagged = 1
                ORDER BY p.created_at DESC
                LIMIT 50
            """).fetchall()
        except sqlite3.OperationalError:
            # Column doesn't exist, initialize empty list
            flagged_posts = []
            run_database_migrations()
        
        formatted_flagged = []
        for post in flagged_posts:
            post_dict = row_to_dict(post)
            created_date, created_time = format_datetime(post_dict.get('created_at'))
            
            formatted_flagged.append({
                'id': post_dict['id'],
                'scan_id': post_dict.get('scan_id', 'N/A'),
                'user': f"{post_dict.get('first_name', '')} {post_dict.get('last_name', '')}".strip(),
                'email': post_dict.get('email', ''),
                'result': post_dict.get('result', 'Unknown'),
                'confidence': post_dict.get('confidence', 0),
                'score': post_dict.get('score', 0),
                'text_preview': post_dict.get('job_text', '')[:200] + '...' if len(post_dict.get('job_text', '')) > 200 else post_dict.get('job_text', ''),
                'flag_reason': post_dict.get('flag_reason', 'Manual flag'),
                'created_date': created_date,
                'created_time': created_time
            })
        
        # Get today's prediction logs
        todays_logs = conn.execute("""
            SELECT 
                p.id,
                p.scan_id,
                p.result,
                p.confidence,
                p.risk_score as score,
                p.method,
                p.created_at,
                u.first_name,
                u.last_name,
                u.email
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            WHERE DATE(p.created_at) = ?
            ORDER BY p.created_at DESC
            LIMIT 100
        """, (today,)).fetchall()
        
        formatted_logs = []
        for log in todays_logs:
            log_dict = row_to_dict(log)
            created_date, created_time = format_datetime(log_dict.get('created_at'))
            
            formatted_logs.append({
                'id': log_dict['id'],
                'scan_id': log_dict.get('scan_id', 'N/A'),
                'user': f"{log_dict.get('first_name', '')} {log_dict.get('last_name', '')}".strip(),
                'email': log_dict.get('email', ''),
                'result': log_dict.get('result', 'Unknown'),
                'confidence': log_dict.get('confidence', 0),
                'score': log_dict.get('score', 0),
                'method': log_dict.get('method', 'Unknown'),
                'created_date': created_date,
                'created_time': created_time
        })
        
        # Get model retraining logs
        model_logs = conn.execute("""
            SELECT 
                ml.id,
                ml.action,
                ml.details,
                ml.created_at,
                u.first_name,
                u.last_name
            FROM model_logs ml
            JOIN users u ON ml.admin_id = u.id
            ORDER BY ml.created_at DESC
            LIMIT 10
        """).fetchall()
        
        formatted_model_logs = []
        for log in model_logs:
            log_dict = row_to_dict(log)
            created_date, created_time = format_datetime(log_dict.get('created_at'))
            
            formatted_model_logs.append({
                'id': log_dict['id'],
                'admin': f"{log_dict.get('first_name', '')} {log_dict.get('last_name', '')}".strip(),
                'action': log_dict.get('action', ''),
                'details': log_dict.get('details', ''),
                'created_date': created_date,
                'created_time': created_time
            })
        
        conn.close()
        
        return render_template(
            "admin_dashboard.html",
            total_users=total_users,
            total_admins=total_admins,
            total_predictions=total_predictions,
            fake_predictions=fake_predictions,
            real_predictions=real_predictions,
            flagged_predictions=flagged_predictions,
            today_predictions=today_predictions,
            fake_rate=fake_rate,
            real_rate=real_rate,
            avg_confidence=avg_confidence,
            users=formatted_users,
            flagged_posts=formatted_flagged,
            todays_logs=formatted_logs,
            model_logs=formatted_model_logs,
            online_users=len(formatted_users)  # Simple estimate
        )
        
    except Exception as e:
        conn.close()
        flash(f"Error loading admin dashboard: {str(e)}", "danger")
        return redirect(url_for("dashboard"))

# ---------------- API: FLAG POST ----------------
@app.route("/admin/flag/<int:prediction_id>", methods=["POST"])
@admin_required
def flag_post(prediction_id):
    data = request.get_json()
    reason = data.get('reason', 'Manual flag by admin')
    
    conn = get_db()
    
    # Check if prediction exists
    prediction = conn.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not prediction:
        conn.close()
        return jsonify({"success": False, "message": "Prediction not found"}), 404
    
    # Flag the post
    conn.execute("""
        UPDATE predictions 
        SET is_flagged=1, flag_reason=?
        WHERE id=?
    """, (reason, prediction_id))
    conn.commit()
    conn.close()
    
    return jsonify({
        "success": True, 
        "message": "Post flagged successfully"
    })

# ---------------- API: UNFLAG POST ----------------
@app.route("/admin/unflag/<int:prediction_id>", methods=["POST"])
@admin_required
def unflag_post(prediction_id):
    conn = get_db()
    
    # Check if prediction exists
    prediction = conn.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not prediction:
        conn.close()
        return jsonify({"success": False, "message": "Prediction not found"}), 404
    
    # Unflag the post
    conn.execute("""
        UPDATE predictions 
        SET is_flagged=0, flag_reason=NULL
        WHERE id=?
    """, (prediction_id,))
    conn.commit()
    conn.close()
    
    return jsonify({
        "success": True, 
        "message": "Post unflagged successfully"
    })

# ---------------- API: DELETE POST ----------------
@app.route("/admin/delete-post/<int:prediction_id>", methods=["DELETE"])
@admin_required
def delete_post(prediction_id):
    conn = get_db()
    
    # Check if prediction exists
    prediction = conn.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
    if not prediction:
        conn.close()
        return jsonify({"success": False, "message": "Post not found"}), 404
    
    # Delete the post
    conn.execute("DELETE FROM predictions WHERE id=?", (prediction_id,))
    conn.commit()
    conn.close()
    
    return jsonify({
        "success": True, 
        "message": "Post deleted successfully"
    })

# ---------------- API: EXPORT DATA ----------------
@app.route("/admin/export/<string:data_type>", methods=["GET"])
@admin_required
def export_data(data_type):
    conn = get_db()
    
    if data_type == "users":
        # Export users data
        users = conn.execute("""
            SELECT id, first_name, last_name, email, is_admin, created_at
            FROM users
            ORDER BY created_at DESC
        """).fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'First Name', 'Last Name', 'Email', 'Is Admin', 'Created At'])
        
        for user in users:
            writer.writerow([user['id'], user['first_name'], user['last_name'], 
                           user['email'], user['is_admin'], user['created_at']])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'users_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    elif data_type == "predictions":
        # Export predictions data
        predictions = conn.execute("""
            SELECT 
                p.id, p.scan_id, p.method, p.result, p.confidence, 
                p.risk_score, p.job_text, p.is_flagged, p.flag_reason, p.created_at,
                u.first_name, u.last_name, u.email
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC
        """).fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Scan ID', 'User', 'Email', 'Method', 'Result', 
                        'Confidence', 'Risk Score', 'Text Preview', 'Flagged', 
                        'Flag Reason', 'Created At'])
        
        for pred in predictions:
            text_preview = pred['job_text'][:100] + '...' if len(pred['job_text']) > 100 else pred['job_text']
            writer.writerow([
                pred['id'], pred['scan_id'], 
                f"{pred['first_name']} {pred['last_name']}", pred['email'],
                pred['method'], pred['result'], pred['confidence'], pred['risk_score'],
                text_preview, pred['is_flagged'], pred['flag_reason'], pred['created_at']
            ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    elif data_type == "logs":
        # Export login logs
        logs = conn.execute("""
            SELECT lh.id, u.first_name, u.last_name, u.email, lh.login_time
            FROM login_history lh
            JOIN users u ON lh.user_id = u.id
            ORDER BY lh.login_time DESC
        """).fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'User', 'Email', 'Login Time'])
        
        for log in logs:
            writer.writerow([
                log['id'], 
                f"{log['first_name']} {log['last_name']}", 
                log['email'], 
                log['login_time']
            ])
        
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'login_logs_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    conn.close()
    return jsonify({"success": False, "message": "Invalid export type"}), 400

# ---------------- API: RETRAIN MODEL ----------------
@app.route("/admin/retrain-model", methods=["POST"])
@admin_required
def retrain_model():
    try:
        conn = get_db()
        
        # Get all predictions for retraining
        predictions = conn.execute("""
            SELECT job_text, result 
            FROM predictions 
            WHERE job_text IS NOT NULL AND job_text != ''
        """).fetchall()
        
        if len(predictions) < 10:
            conn.close()
            return jsonify({
                "success": False, 
                "message": f"Not enough data for retraining. Need at least 10 samples, have {len(predictions)}."
            }), 400
        
        # Prepare data for retraining
        texts = []
        labels = []
        for pred in predictions:
            texts.append(pred['job_text'])
            labels.append(1 if pred['result'] == 'Fake Job' else 0)
        
        # Here you would implement your retraining logic
        # For now, we'll just simulate it
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.svm import SVC
        
        # Create new vectorizer and model
        new_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X = new_vectorizer.fit_transform(texts)
        
        new_model = SVC(kernel='linear', probability=True)
        new_model.fit(X, labels)
        
        # Save the new model
        joblib.dump(new_vectorizer, 'tfidf_vectorizer.pkl')
        joblib.dump(new_model, 'svm_model.pkl')
        
        # Reload the model
        global vectorizer, model
        vectorizer = new_vectorizer
        model = new_model
        
        # Log the retraining
        conn.execute("""
            INSERT INTO model_logs (admin_id, action, details)
            VALUES (?, ?, ?)
        """, (session["user_id"], "model_retrain", f"Retrained with {len(predictions)} samples"))
        conn.commit()
        conn.close()
        
        return jsonify({
            "success": True, 
            "message": f"Model retrained successfully with {len(predictions)} samples",
            "samples": len(predictions),
            "fake_count": sum(labels),
            "real_count": len(labels) - sum(labels)
        })
        
    except Exception as e:
        return jsonify({
            "success": False, 
            "message": f"Error retraining model: {str(e)}"
        }), 500

# ---------------- PROMOTE USER ----------------
@app.route("/admin/promote/<int:user_id>", methods=["POST"])
@admin_required
def promote_user(user_id):
    conn = get_db()
    
    # Check if user exists and is not already admin
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404
    
    user_dict = row_to_dict(user)
    
    if user_dict["is_admin"]:
        conn.close()
        return jsonify({"success": False, "message": "User is already an admin"}), 400
    
    # Promote user to admin
    conn.execute("UPDATE users SET is_admin=1 WHERE id=?", (user_id,))
    conn.commit()
    
    # Get updated user info
    updated_user = conn.execute("SELECT first_name FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    
    if updated_user:
        updated_user_dict = row_to_dict(updated_user)
        return jsonify({
            "success": True, 
            "message": f"User {updated_user_dict['first_name']} has been promoted to Administrator"
        })
    else:
        return jsonify({"success": False, "message": "User not found"}), 404

# ---------------- DEMOTE USER ----------------
@app.route("/admin/demote/<int:user_id>", methods=["POST"])
@admin_required
def demote_user(user_id):
    # Prevent demoting yourself
    if user_id == session["user_id"]:
        return jsonify({"success": False, "message": "You cannot demote yourself"}), 400
    
    conn = get_db()
    
    # Check if user exists and is an admin
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404
    
    user_dict = row_to_dict(user)
    
    if not user_dict["is_admin"]:
        conn.close()
        return jsonify({"success": False, "message": "User is not an admin"}), 400
    
    # Demote user
    conn.execute("UPDATE users SET is_admin=0 WHERE id=?", (user_id,))
    conn.commit()
    
    # Get updated user info
    updated_user = conn.execute("SELECT first_name FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    
    if updated_user:
        updated_user_dict = row_to_dict(updated_user)
        return jsonify({
            "success": True, 
            "message": f"User {updated_user_dict['first_name']} has been demoted to regular user"
        })
    else:
        return jsonify({"success": False, "message": "User not found"}), 404

# ---------------- DELETE USER ----------------
@app.route("/admin/delete/<int:user_id>", methods=["DELETE"])
@admin_required
def delete_user(user_id):
    # Prevent deleting yourself
    if user_id == session["user_id"]:
        return jsonify({"success": False, "message": "You cannot delete yourself"}), 400
    
    conn = get_db()
    
    # Get user info before deletion
    user = conn.execute("SELECT first_name FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return jsonify({"success": False, "message": "User not found"}), 404
    
    user_dict = row_to_dict(user)
    
    try:
        # Delete user predictions
        conn.execute("DELETE FROM predictions WHERE user_id=?", (user_id,))
    except:
        pass
    
    try:
        # Delete user login history
        conn.execute("DELETE FROM login_history WHERE user_id=?", (user_id,))
    except:
        pass
    
    # Delete user
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    
    return jsonify({
        "success": True, 
        "message": f"User {user_dict['first_name']} has been deleted permanently"
    })

# ---------------- LOGOUT ----------------
@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out successfully", "success")
    return redirect(url_for("login"))

# ---------------- CLEAR HISTORY ----------------
@app.route("/clear_history", methods=["GET", "POST"])
@login_required
def clear_history():
    if request.method == "POST":
        conn = get_db()
        conn.execute("""
            DELETE FROM predictions 
            WHERE user_id = ?
        """, (session["user_id"],))
        conn.commit()
        conn.close()
        
        flash("Your prediction history has been cleared successfully.", "success")
        return redirect(url_for("dashboard"))
    
    return redirect(url_for("dashboard"))

# ---------------- RESET DATABASE ----------------
@app.route("/reset-db")
def reset_db():
    """Reset database - use only for debugging"""
    import os
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("Database deleted. Restart the application to recreate it.")
        return "Database reset. Please restart the application."
    return "Database file not found"

# ---------------- VIEW ALL USERS ----------------
@app.route("/debug/users")
def debug_users():
    """Debug route to see all users in database"""
    conn = get_db()
    users = conn.execute("SELECT * FROM users ORDER BY id").fetchall()
    user_list = []
    for user in users:
        user_dict = row_to_dict(user)
        # Remove password for security
        if 'password' in user_dict:
            user_dict['password'] = '********'
        user_list.append(user_dict)
    conn.close()
    return jsonify(user_list)

# -------------------------------------------------
if __name__ == "__main__":
    # Initialize database on startup (adds admin user if needed)
    init_db()
    print("=" * 50)
    print("JobGuard Fake Job Detection System")
    print("=" * 50)
    print("Admin credentials: admin@jobguard.com / admin123")
    print("Test user credentials:")
    print("  - john@example.com / password123")
    print("  - jane@example.com / password123")
    print("  - bob@example.com / password123")
    print("  - alice@example.com / password123")
    print("=" * 50)
    print("\nPassword Reset Configuration:")
    print(f"  - TEST_MODE: {TEST_MODE}")
    if TEST_MODE:
        print("  - Reset codes will be printed to console")
    else:
        print("  - Reset codes will be sent via email")
        if EMAIL_CONFIG['SMTP_USERNAME'] == 'your-email@gmail.com':
            print("  ⚠️  WARNING: Email not configured! Update EMAIL_CONFIG in app.py")
    print("=" * 50)
    print("\nAccess URLs:")
    print("  - http://localhost:5000/login - Login page")
    print("  - http://localhost:5000/admin - Admin dashboard")
    print("  - http://localhost:5000/debug/users - View all users in database")
    print("  - http://localhost:5000/reset-db - Reset database (debug only)")
    print("  - http://localhost:5000/test-reset-api - Test reset API")
    print("  - http://localhost:5000/test-email - Test email configuration")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)