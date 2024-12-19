import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def load_file(path_to_file):
    with open(path_to_file) as f:
        return json.load(f)
    
def load_data(path_to_data):
    return pd.read_csv(path_to_data)

def handle_missing_values(df, feature_handling):
    for feature, details in feature_handling.items():
        if 'feature_details' in details and 'missing_values' in details['feature_details']:
            if details['feature_variable_type'] == 'numerical' and details['feature_details']['missing_values'] == 'Impute':
                if details['feature_details']['impute_with'] == 'Average of values':
                    imputer = SimpleImputer(strategy='mean')
                    df[feature] = imputer.fit_transform(df[[feature]])
                elif details['feature_details']['impute_with'] == 'custom':
                    imputer = SimpleImputer(strategy='constant', fill_value=details['feature_details']['impute_value'])
                    df[feature] = imputer.fit_transform(df[[feature]])
    return df

def convert_to_numeric(df):
    for column in df.select_dtypes(include=['object']).columns:
        try:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        except ValueError:
            df[column] = LabelEncoder().fit_transform(df[column].astype(str))
    return df

def generate_features(df, feature_generation):
    
    df = convert_to_numeric(df)
    
    for interaction in feature_generation['linear_interactions']:
        if all(col in df.columns for col in interaction):
            new_feature = f'{interaction[0]}_x_{interaction[1]}'
            df[new_feature] = df[interaction[0]] * df[interaction[1]]
    
    for poly in feature_generation['polynomial_interactions']:
        features = poly.split('/')
        if features[0] in df.columns and features[1] in df.columns:
            df[features[0]] = pd.to_numeric(df[features[0]], errors='coerce')
            df[features[1]] = pd.to_numeric(df[features[1]], errors='coerce')
            df[features[1]].replace(0, np.nan, inplace=True)
            new_feature = f'{features[0]}_div_{features[1]}'
            df[new_feature] = df[features[0]] / df[features[1]]
    
    for pairwise in feature_generation['explicit_pairwise_interactions']:
        features = pairwise.split('/')
        if features[0] in df.columns and features[1] in df.columns:
            df[features[0]] = pd.to_numeric(df[features[0]], errors='coerce')
            df[features[1]] = pd.to_numeric(df[features[1]], errors='coerce')
            df[features[1]].replace(0, np.nan, inplace=True)
            new_feature = f'{features[0]}_over_{features[1]}'
            df[new_feature] = df[features[0]] / df[features[1]]
    
    return df

def reduce_features(df, feature_reduction, target_column):
    
    if feature_reduction['feature_reduction_method'] == 'Tree-based':
        num_features = int(feature_reduction['num_of_features_to_keep'])
        
        X = df.drop(columns=[target_column])
        X = convert_to_numeric(X)
        y = df[target_column]
         
        X = X.fillna(X.mean())  
        y = y.fillna(y.mean()) 
        
        rf = RandomForestRegressor(n_estimators=feature_reduction['num_of_trees'], 
                                   max_depth=feature_reduction['depth_of_trees'])
        rf.fit(X, y)
        
        feature_importances = rf.feature_importances_
        important_features = np.argsort(feature_importances)[-num_features:]
        
        return df.iloc[:, important_features.tolist() + [df.columns.get_loc(target_column)]]
    
    elif feature_reduction['feature_reduction_method'] == 'PCA':
        pca = PCA(n_components=int(feature_reduction['num_of_features_to_keep']))
        pca_result = pd.DataFrame(pca.fit_transform(df))
        
        pca_result[target_column] = df[target_column]
        return pca_result
    
    elif feature_reduction['feature_reduction_method'] == 'Corr with Target':
        X = df.drop(columns=[target_column])
        X = convert_to_numeric(X)
        y = df[target_column]
    
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        correlations = X.corrwith(y).abs()
        
        num_features = int(feature_reduction['num_of_features_to_keep'])
        top_features = correlations.nlargest(num_features).index
        return df[top_features.tolist() + [target_column]]
    
    elif feature_reduction['feature_reduction_method'] == 'No Reduction':
        return df
    
    return df

def prepare_model(prediction_type, algorithms, hyperparameters):

    models = []
    if prediction_type == 'Regression':
        if 'RandomForestRegressor' in algorithms and algorithms['RandomForestRegressor']['is_selected']:
            min_depth = int(algorithms['RandomForestRegressor']['min_depth'])
            max_depth = int(algorithms['RandomForestRegressor']['max_depth'])
            
            rf_params = {
                'n_estimators': [algorithms['RandomForestRegressor']['min_trees'], 
                                 algorithms['RandomForestRegressor']['max_trees']],
                'max_depth': [min_depth, max_depth]
            }
            rf_model = GridSearchCV(RandomForestRegressor(), rf_params, cv=5)
            models.append(('Random Forest', rf_model))

        if 'LinearRegression' in algorithms and algorithms['LinearRegression']['is_selected']:
            lr_params = {'fit_intercept': [True, False],'normalize': [True, False],
                         'copy_X': [True, False] }
            lr_model = GridSearchCV(LinearRegression(), lr_params, cv=5)
            models.append(('Linear Regression', lr_model))

        if 'DecisionTreeRegressor' in algorithms and algorithms['DecisionTreeRegressor']['is_selected']:
            dt_params = {'max_depth': [None, 5, 10, 20],'min_samples_split': [2, 5, 10],
                         'min_samples_leaf': [1, 2, 5],'max_features': ['auto', 'sqrt', 'log2']}
            dt_model = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5)
            models.append(('Decision Tree', dt_model))

        if 'SVR' in algorithms and algorithms['SVR']['is_selected']:
            svr_params = {'C': [1, 10], 'kernel': ['linear', 'rbf']}
            svr_model = GridSearchCV(SVR(), svr_params, cv=5)
            models.append(('SVR', svr_model))

    return models

def evaluate_model(models, X_train, y_train, X_test, y_test):
    
    for model_name, model in models:
        print(f'Training {model_name}...')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f'Mean Squared Error for {model_name}: {mse}')
       
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Mean Absolute Error (MAE): {mae}")

        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse}")

        r2 = r2_score(y_test, y_pred)
        print(f"R-Squared (R²): {r2}")


# Main function
def run_autopipeline(path_to_file, dataset_path):

    config = load_file(path_to_file)
    df = load_data(dataset_path)
    
    # Step 1: Feature handling 
    df = handle_missing_values(df, config['design_state_data']['feature_handling'])
    
    # Step 2: Feature generation
    df = generate_features(df, config['design_state_data']['feature_generation'])
    
    # Step 3: Feature reduction 
    target_column = config['design_state_data']['target']['target']
    df_reduced = reduce_features(df, config['design_state_data']['feature_reduction'], target_column)  # Pass target_column here
    
    print("Columns after feature reduction:", df_reduced.columns)

    # Step 4: Prepare target and features

    if target_column not in df_reduced.columns:
     raise ValueError(f"Target column '{target_column}' not found in the DataFrame after feature reduction")

    X = df_reduced.drop(columns=[target_column]) 
    y = df_reduced[target_column]
    
    # Step 5: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Step 6: Prepare models based on configuration
    models = prepare_model(config['design_state_data']['target']['prediction_type'], 
                           config['design_state_data']['algorithms'], 
                           config['design_state_data']['hyperparameters'])
    
    # Step 7: Evaluate models
    evaluate_model(models, X_train, y_train, X_test, y_test)

# Run the AutoML pipeline
path_to_file = "C:\\Users\\NANDHU\\Desktop\\algoparams.json"  
dataset_path = "C:\\Users\\NANDHU\\Desktop\\iris.csv" 
run_autopipeline(path_to_file, dataset_path)

