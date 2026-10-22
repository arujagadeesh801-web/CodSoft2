import pandas as pd

def load_and_preprocess_data(train_path="data/fraudTrain.csv", test_path="data/fraudTest.csv"):
    print("Loading train and test datasets...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Drop index column if present
    if "Unnamed: 0" in train_df.columns:
        train_df = train_df.drop("Unnamed: 0", axis=1)
    if "Unnamed: 0" in test_df.columns:
        test_df = test_df.drop("Unnamed: 0", axis=1)
        
    # Drop unique identifiers and heavy text columns that aren't useful for basic ML models
    drop_cols = ['cc_num', 'first', 'last', 'trans_num', 'merch_zipcode', 'trans_date_trans_time', 'dob', 'merchant', 'street', 'city', 'state', 'job']
    train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    
    # Handle categorical features ('category' and 'gender') using One-Hot Encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = pd.get_dummies(combined, columns=['category', 'gender'], drop_first=True)
    
    train_processed = combined[combined['is_train'] == 1].drop('is_train', axis=1)
    test_processed = combined[combined['is_train'] == 0].drop('is_train', axis=1)
    
    X_train = train_processed.drop('is_fraud', axis=1)
    y_train = train_processed['is_fraud']
    
    X_test = test_processed.drop('is_fraud', axis=1)
    y_test = test_processed['is_fraud']
    
    return X_train, y_train, X_test, y_test