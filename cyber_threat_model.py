"""
Machine Learning-Based Cyber Threat Trend Prediction
for Kenyan Government Digital Services

MSc Cybersecurity Project
Mount Kenya University

Author:
Stephen Musau Makau

UPDATED: Now supports custom data upload and retraining
"""

import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# =====================================
# GLOBAL VARIABLES
# =====================================
current_data_source = "default"  # "default" or "uploaded"
dataset = None
environment_encoder = None
threat_encoder = None
log_model = None
rf_model = None
xgb_model = None
X_test = None
y_test = None

# =====================================
# DEFAULT DATASET CONSTRUCTION
# =====================================

def get_default_data():
    """Return the original research dataset"""
    raw_data = {
        "Year": [2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2025],
        "Month": [3, 9, 2021, 10, 3, 8, 2, 11, 5, 7],
        "DDoS_Attacks": [
            500, 700, 900, 1200, 1800,
            2400, 3200, 2800, 2100, 1900
        ],
        "Malware_Attacks": [
            4000, 5000, 6500, 7500, 9000,
            12000, 15000, 13000, 11000, 9500
        ],
        "Phishing_Attacks": [
            600, 800, 1000, 1300, 1700,
            2200, 3000, 2600, 1800, 1500
        ],
        "Web_Attacks": [
            1000, 1200, 1600, 2000, 2500,
            3200, 4000, 3500, 3000, 2800
        ],
        "Critical_CVEs": [
            20, 25, 30, 35, 45,
            55, 70, 65, 50, 45
        ],
        "Patch_Delay_Days": [
            20, 18, 17, 15, 14,
            12, 10, 11, 13, 15
        ],
        "Traffic_Volume": [
            200000, 250000, 300000,
            350000, 450000, 600000,
            800000, 750000,
            700000, 650000
        ],
        "Inflation_Rate": [
            5.4, 5.6, 6.1, 6.4,
            7.9, 8.5, 9.2,
            7.8, 5.7, 4.5
        ],
        "GDP_Growth": [
            5.3, 5.0, 7.5, 5.9,
            5.4, 5.2, 4.8,
            5.6, 5.0, 5.5
        ],
        "Economic_Environment": [
            "Stable",
            "Stable",
            "Improving",
            "Stable",
            "High_Cost",
            "High_Cost",
            "High_Cost",
            "Pressure",
            "Improving",
            "Stable"
        ],
        "Threat_Level": [
            "Medium",
            "Medium",
            "Medium",
            "High",
            "High",
            "High",
            "Critical",
            "High",
            "Medium",
            "Medium"
        ]
    }
    return pd.DataFrame(raw_data)


# =====================================
# INITIALIZATION
# =====================================

def initialize_models(df):
    """Initialize and train models on provided dataframe"""
    global dataset, environment_encoder, threat_encoder
    global log_model, rf_model, xgb_model, X_test, y_test
    
    dataset = df.copy()
    
    # Encode categorical variables
    environment_encoder = LabelEncoder()
    threat_encoder = LabelEncoder()
    
    dataset["Economic_Environment_Encoded"] = environment_encoder.fit_transform(
        dataset["Economic_Environment"]
    )
    
    dataset["Threat_Level_Encoded"] = threat_encoder.fit_transform(
        dataset["Threat_Level"]
    )
    
    # Prepare features and target
    X = dataset.drop(["Threat_Level", "Threat_Level_Encoded", "Economic_Environment"], axis=1)
    y = dataset["Threat_Level_Encoded"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Initialize models
    log_model = LogisticRegression(max_iter=1000)
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    # Train models
    log_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    # Store test data for accuracy calculation
    X_test = X_test
    y_test = y_test
    
    return True


# Initialize with default data on module load
initialize_models(get_default_data())


# =====================================
# DATA UPLOAD FUNCTIONS
# =====================================

def generate_excel_template():
    """
    Generate an Excel template file with sample data and instructions
    Returns: BytesIO buffer containing the Excel file
    """
    # Create sample data (first 3 rows of default dataset)
    template_data = {
        "Year": [2024, 2024, 2025],
        "Month": [3, 9, 5],
        "DDoS_Attacks": [1800, 2400, 2100],
        "Malware_Attacks": [9000, 12000, 11000],
        "Phishing_Attacks": [1700, 2200, 1800],
        "Web_Attacks": [2500, 3200, 3000],
        "Critical_CVEs": [45, 55, 50],
        "Patch_Delay_Days": [14, 12, 13],
        "Traffic_Volume": [450000, 600000, 700000],
        "Inflation_Rate": [7.9, 8.5, 5.7],
        "GDP_Growth": [5.4, 5.2, 5.0],
        "Economic_Environment": ["High_Cost", "High_Cost", "Improving"],
        "Threat_Level": ["High", "High", "Medium"]
    }
    
    df_template = pd.DataFrame(template_data)
    
    # Create instructions sheet
    instructions_data = {
        "Column Name": [
            "Year", "Month", "DDoS_Attacks", "Malware_Attacks", 
            "Phishing_Attacks", "Web_Attacks", "Critical_CVEs",
            "Patch_Delay_Days", "Traffic_Volume", "Inflation_Rate",
            "GDP_Growth", "Economic_Environment", "Threat_Level"
        ],
        "Data Type": [
            "Integer", "Integer (1-12)", "Integer", "Integer",
            "Integer", "Integer", "Integer",
            "Integer", "Integer", "Float (decimal)",
            "Float (decimal)", "Text", "Text"
        ],
        "Description": [
            "Year of observation (e.g., 2024)",
            "Month number (1-12)",
            "Number of DDoS attack incidents",
            "Number of malware infections detected",
            "Number of phishing attempts recorded",
            "Number of web application attacks",
            "Count of critical CVEs published",
            "Average days to deploy security patches",
            "Network traffic volume (requests/hits)",
            "Inflation rate percentage",
            "GDP growth rate percentage",
            "Economic condition: Stable, High_Cost, Improving, or Pressure",
            "Historical threat level: Medium, High, or Critical"
        ],
        "Example": [
            "2024", "8", "4200", "18500",
            "4100", "5200", "95",
            "9", "1100000", "6.8",
            "5.2", "Stable", "High"
        ],
        "Required": [
            "Yes", "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes",
            "Yes", "Yes", "Yes"
        ]
    }
    df_instructions = pd.DataFrame(instructions_data)
    
    # Create additional notes
    notes_data = {
        "Important Notes": [
            "1. Minimum 5 rows of data required for training",
            "2. Column names must match exactly (case-sensitive)",
            "3. Economic_Environment must be one of: Stable, High_Cost, Improving, Pressure",
            "4. Threat_Level must be one of: Medium, High, Critical",
            "5. Do not leave any cells empty",
            "6. Save file as .xlsx (Excel) or .csv format before uploading",
            "7. More historical data (20+ rows) will produce better predictions",
            "8. Ensure data is chronological (earliest to latest)"
        ]
    }
    df_notes = pd.DataFrame(notes_data)
    
    # Write to Excel buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_template.to_excel(writer, sheet_name='Sample_Data', index=False)
        df_instructions.to_excel(writer, sheet_name='Column_Guide', index=False)
        df_notes.to_excel(writer, sheet_name='Instructions', index=False)
    
    buffer.seek(0)
    return buffer


def validate_uploaded_data(df):
    """
    Validate uploaded dataframe has correct structure
    Returns: dict with 'valid' (bool) and 'message' (str)
    """
    required_columns = [
        "Year", "Month", "DDoS_Attacks", "Malware_Attacks",
        "Phishing_Attacks", "Web_Attacks", "Critical_CVEs",
        "Patch_Delay_Days", "Traffic_Volume", "Inflation_Rate",
        "GDP_Growth", "Economic_Environment", "Threat_Level"
    ]
    
    # Check columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return {
            'valid': False,
            'message': f"Missing columns: {', '.join(missing_cols)}. Please use the template."
        }
    
    # Check minimum rows
    if len(df) < 5:
        return {
            'valid': False,
            'message': f"Insufficient data. Found {len(df)} rows, minimum 5 required."
        }
    
    # Check for empty values
    if df.isnull().sum().sum() > 0:
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].index.tolist()
        return {
            'valid': False,
            'message': f"Empty values found in columns: {', '.join(cols_with_nulls)}. Please fill all cells."
        }
    
    # Validate Economic_Environment values
    valid_econ = ["Stable", "High_Cost", "Improving", "Pressure"]
    invalid_econ = df[~df["Economic_Environment"].isin(valid_econ)]["Economic_Environment"].unique()
    if len(invalid_econ) > 0:
        return {
            'valid': False,
            'message': f"Invalid Economic_Environment values: {', '.join(invalid_econ)}. Use: Stable, High_Cost, Improving, or Pressure"
        }
    
    # Validate Threat_Level values
    valid_threats = ["Medium", "High", "Critical"]
    invalid_threats = df[~df["Threat_Level"].isin(valid_threats)]["Threat_Level"].unique()
    if len(invalid_threats) > 0:
        return {
            'valid': False,
            'message': f"Invalid Threat_Level values: {', '.join(invalid_threats)}. Use: Medium, High, or Critical"
        }
    
    # Check numeric columns are numeric
    numeric_cols = ["Year", "Month", "DDoS_Attacks", "Malware_Attacks", 
                  "Phishing_Attacks", "Web_Attacks", "Critical_CVEs",
                  "Patch_Delay_Days", "Traffic_Volume", "Inflation_Rate", "GDP_Growth"]
    
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            return {
                'valid': False,
                'message': f"Column '{col}' must contain numeric values only."
            }
    
    return {
        'valid': True,
        'message': f"Data validation passed! {len(df)} rows ready for training."
    }


def train_on_uploaded_data(df):
    """
    Retrain models on uploaded data
    Pass None to reset to default data
    Returns: (success: bool, message: str)
    """
    global current_data_source
    
    try:
        if df is None:
            # Reset to default
            initialize_models(get_default_data())
            current_data_source = "default"
            return True, "Models reset to default dataset successfully."
        
        # Train on uploaded data
        initialize_models(df)
        current_data_source = "uploaded"
        
        # Calculate accuracy to report
        accuracy = get_model_accuracy() * 100
        return True, f"Models trained successfully on {len(df)} rows. XGBoost accuracy: {accuracy:.2f}%"
        
    except Exception as e:
        return False, f"Training failed: {str(e)}"


def get_current_data_source():
    """Return current data source type"""
    return current_data_source


# =====================================
# ANALYSIS FUNCTIONS
# =====================================

def get_results():
    """Get accuracy results for all models"""
    return {
        "Logistic Regression": accuracy_score(y_test, log_model.predict(X_test)),
        "Random Forest": accuracy_score(y_test, rf_model.predict(X_test)),
        "XGBoost": accuracy_score(y_test, xgb_model.predict(X_test))
    }

def get_model_accuracy():
    """Get XGBoost accuracy"""
    return get_results()["XGBoost"]

def get_dataset():
    """Return current dataset (encoded version for display)"""
    # Return human-readable version
    display_df = dataset.copy()
    if "Economic_Environment_Encoded" in display_df.columns:
        display_df = display_df.drop("Economic_Environment_Encoded", axis=1)
    if "Threat_Level_Encoded" in display_df.columns:
        display_df = display_df.drop("Threat_Level_Encoded", axis=1)
    return display_df

# =====================================
# PARAMETERS FOR 2027
# =====================================

def get_parameters():
    return pd.DataFrame({
        "Parameter": [
            "Year",
            "Month",
            "DDoS Attacks",
            "Malware Attacks",
            "Phishing Attacks",
            "Web Attacks",
            "Critical CVEs",
            "Patch Delay Days",
            "Traffic Volume",
            "Inflation Rate",
            "GDP Growth",
            "Economic Environment"
        ],
        "2027 Projection": [
            2027,
            "August",
            4200,
            18500,
            4100,
            5200,
            95,
            9,
            1100000,
            "6.8%",
            "5.2%",
            "Stable"
        ]
    })

# =====================================
# PREDICTION ENGINE
# =====================================

def predict_2027():
    """Predict threat level for 2027 using current models"""
    future = pd.DataFrame({
        "Year": [2027],
        "Month": [8],
        "DDoS_Attacks": [4200],
        "Malware_Attacks": [18500],
        "Phishing_Attacks": [4100],
        "Web_Attacks": [5200],
        "Critical_CVEs": [95],
        "Patch_Delay_Days": [9],
        "Traffic_Volume": [1100000],
        "Inflation_Rate": [6.8],
        "GDP_Growth": [5.2],
        "Economic_Environment_Encoded": [2]  # Stable = 2
    })
    
    result = xgb_model.predict(future)
    return threat_encoder.inverse_transform(result)[0]
