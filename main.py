import argparse
from src.train import train_model
from src.evaluate import evaluate_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud Detection Pipeline")
    parser.add_argument(
        "--train_path", 
        type=str, 
        default="data/fraudTrain.csv", 
        help="Path to the training dataset file"
    )
    parser.add_argument(
        "--test_path", 
        type=str, 
        default="data/fraudTest.csv", 
        help="Path to the testing dataset file"
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="random_forest", 
        choices=["logistic", "decision_tree", "random_forest"], 
        help="Choose classifier type"
    )
    
    args = parser.parse_args()
    
    # Run training
    model, scaler, X_test, y_test = train_model(
        train_path=args.train_path,
        test_path=args.test_path,
        model_type=args.model_type
    )
    
    # Run evaluation on test set
    evaluate_model(X_test=X_test, y_test=y_test)