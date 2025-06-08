import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from xgboost import XGBClassifier
from scipy.stats import mode


#dataset

# Model hyperparameter, taken from previous exp. Best five
model_params = {
    "XGBOOST_D3_1": {
        "model_class": XGBClassifier,
        "params": {
            "booster": "gblinear",
            "gamma": 0,
            "learning_rate": None,
            "max_depth": 3,
            "min_child_weight": 2,
            "n_estimators": 10
        }
    },
    "SVM_D3_1": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf"
        }
    },
    "SVM_D3_2": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf"
        }
    },
    "SVM_D4_2": {
        "model_class": SVC,
        "params": {
            "C": 100,
            "gamma": 0.001,
            "kernel": "rbf"
        }
    },
    "XGBOOST_D4_2": {
        "model_class": XGBClassifier,
        "params": {
            "booster": "gblinear",
            "gamma": 0,
            "learning_rate": 1,
            "max_depth": 3,
            "min_child_weight": 2,
            "n_estimators": 50
        }
    }
}

#set the parameters  based on the conf, about the normalization, feature selection and pca
def parse_pipeline_string(pipeline_str):
    #parameters to set
    result = {
        "Normalization": None,
        "Feature_selection": None,
        "k" : None,
        "PCA": None,
        "n_components": None
    }
    if pipeline_str == "XGBOOST_D3_1":
        result["Normalization"] = "None"
        result["Feature_selection"] = "None"
        result["k"] = 0
        result["PCA"] = "Yes"
        result["n_components"] = 180

    elif pipeline_str == "SVM_D3_1":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 180
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "SVM_D4_2":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 160
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "SVM_D3_2":
        result["Normalization"] = "StandardScaler"
        result["Feature_selection"] = "Yes"
        result["k"] = 180
        result["PCA"] = "None"
        result["n_components"] = 0
    elif pipeline_str == "XGBOOST_D4_2":
        result["Normalization"] = "None"
        result["Feature_selection"] = "None"
        result["k"] = 0
        result["PCA"] = "Yes"
        result["n_components"] = 100
    return result


#"main" does cross validation for each configuration
def run_crossval_and_save_predictions(conf_list):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for conf in conf_list:
        print(f"Running config: {conf}")

        #recover the right configuration
        conf_diz = parse_pipeline_string(conf)

        # Load dataset based on config
        if conf == "XGBOOST_D3_1" or conf == "SVM_D3_1":
            df_all = pd.read_csv("All_Alex_back.csv")
        elif conf == "XGBOOST_D4_2" or conf == "SVM_D4_2":
            df_all = pd.read_csv("All_Gabor_back_den.csv")
        elif conf == "SVM_D3_2":
            df_all = pd.read_csv("All_Alex_back_den.csv")
        else:
            raise ValueError(f"Dataset not defined for configuration: {conf}")

        data = df_all.drop(columns=["label"])
        target_dataset = df_all["label"]
        label_encoder = LabelEncoder()
        target = label_encoder.fit_transform(target_dataset)
        feature_names = data.columns
        conf_pred=[]
        all_preds = np.zeros(len(target), dtype=object)

        for fold, (train_idx, test_idx) in enumerate(cv.split(data, target)):
            X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
            y_train, y_test = target[train_idx], target[test_idx]

            #Apply the conditions found the in the conf

            # Normalization
            if conf_diz["Normalization"] == "StandardScaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # Feature selection
            if conf_diz["Feature_selection"] == "Yes":
                selector = SelectKBest(score_func=f_classif, k=conf_diz["k"])
                X_train = selector.fit_transform(X_train, y_train)
                X_test = selector.transform(X_test)

            # PCA
            if conf_diz["PCA"] == "Yes":
                pca = PCA(n_components=conf_diz["n_components"])
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Model
            if conf.startswith("SVM"):
                model = SVC(**model_params[conf]["params"])
            elif conf.startswith("XGBOOST"):
                model = XGBClassifier(**model_params[conf]["params"])
            else:
                raise ValueError("Unsupported model type.")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_pred.append(accuracy)
            all_preds[test_idx] = y_pred

        # Salva solo le predizioni
        df_pred = pd.DataFrame({"predicted_label": all_preds})
        df_pred.to_csv(f"predictions_{conf}.csv", index=False)
        print(f"Saved predictions to predictions_{conf}.csv")
        total=0
        for acc in conf_pred:
            total+=acc
        acc_score=total/len(conf_pred)
        print(conf +f" accuracy:{acc_score:.4f}\n")

#best 5 config
conf_list = ["SVM_D3_1", "SVM_D3_2", "SVM_D4_2", "XGBOOST_D3_1", "XGBOOST_D4_2"]
#run the crossval for all configuration
run_crossval_and_save_predictions(conf_list)

preds_df = pd.DataFrame()

#build the csv
for conf in conf_list:
    preds = pd.read_csv(f"predictions_{conf}.csv")["predicted_label"]
    preds_df[conf] = preds


# Majority voting (for each sample (each leaf) will find the most common predition)
final_preds, _ = mode(preds_df.values, axis=1, keepdims=False)
final_preds = final_preds.flatten()

# load the dataset in order to take the true label
df_all = pd.read_csv("All_Alex_back.csv")
true_labels = df_all["label"]
label_encoder = LabelEncoder()
true_labels_encoded = label_encoder.fit_transform(true_labels)

# Accuracy finale: true label vs final preds (majority voting)
accuracy = accuracy_score(true_labels_encoded, final_preds)
print(f"Ensemble Majority Voting Accuracy: {accuracy:.4f}")
