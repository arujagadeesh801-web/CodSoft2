import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from src.preprocessing import load_and_preprocess_data

def train_model(train_path="data/fraudTrain.csv", test_path="data/fraudTest.csv", model_type="random_forest", model_save_path="models/model.pkl", scaler_save_path="models/scaler.pkl"):
    X_train, y_train, X_test, y_test = load_and_preprocess_data(train_path, test_path)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training classifier ({model_type})...")
    if model_type == "logistic":
        model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    elif model_type == "decision_tree":
        model = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unsupported model type specified.")
        
    model.fit(X_train_scaled, y_train)
    
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print("Saving trained model and scaler...")
    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print("Training pipeline finished successfully!")
    
    return model, scaler, X_test_scaled, y_test