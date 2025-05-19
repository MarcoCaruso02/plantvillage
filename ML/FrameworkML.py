import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import sys

#models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

#parameters


csv_path = "features.csv"        # Nome del file CSV
target_column = "label"
result_file ="XGBOOST-D0-A1-B.txt"

normalization = "none" ##possible options are min-max; std; none
feature_selection = False
num_feature = 10 #k-best for example
pca = False
num_principal_components = 15 #pca


model_name = "xgboost" #possible options are: random_forest,svm, xgboost
#kernel_type = "linear" #svm kernel: linear, poly, rbf, sigmoid.
cross_validation = True
cv_folds = 5

test_size = 0.2
train_size = 1 - test_size


#MAIN



df = pd.read_csv(csv_path)
data = df.drop(columns = [target_column])
target=df[target_column]

#encoding (no categorical feature, maybe just add a check for the prof). Maybe is needed encoding of the target
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

#normalization
if normalization == 'min-max':
    scaler = MinMaxScaler()
elif normalization == 'std':
    scaler = StandardScaler()

# no normalization
if normalization == 'none':
    data_scaled = data
#apply normalization
else:
    data_scaled = scaler.fit_transform(data)



if feature_selection:
    #k-best
    selector = SelectKBest(score_func=f_classif, k=num_feature)
    train_features_selected = selector.fit_transform(data, target)
    selected_feature_names = data.columns[selector.get_support()]
    print("Selected features: ", selected_feature_names)
    data_scaled = pd.DataFrame(train_features_selected,
                        columns=selected_feature_names,
                        index=data.index)

if pca:
    #TO ADD
    pca = PCA(n_components=num_principal_components)
    data_scaled = pca.fit_transform(data_scaled)


#model, + check if needed grid search
if model_name == "random_forest":
    # TO DO model =
    model = RandomForestClassifier(n_estimators=100,
                                      random_state=42)
    print("Model: Random Forest")
elif model_name == "svm":
    #Maybe to add a grid search to look for the best kernel and hyperpar.
    param_grid = [
        {'kernel': ['linear'], 'C': [0.01, 0.1, 1, 10, 100]},
        {'kernel': ['rbf'], 'C': [0.01, 0.1, 1, 10, 100], 'gamma': [1e-3, 1e-2, 0.1, 'scale', 'auto']},
        {'kernel': ['poly'], 'C': [0.1, 1], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
    ]

    model = SVC()
    model = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')


elif model_name == "xgboost":
    param_grid = {
        "n_estimators": [10, 50],
        "learning_rate": [None, 0.1, 1],
        "booster": ["gbtree", "gblinear"],
        "max_depth": [3, 6, 12],
        "min_child_weight": [0.5, 1, 2],
        "gamma": [0, 1, 10]
    }

    # Modello base
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # GridSearchCV
    model = GridSearchCV(estimator=model, param_grid=param_grid,
                         cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    print("Model: XGBoost")

#TO ADD GRID SEARCH

scores = []
if cross_validation:
    # TO ADD
    print("cross-validation")
    scores = cross_val_score(model, data_scaled, target_encoded, cv=cv_folds)

    for i in range(cv_folds):
        print("Fold-{}: {:.3f}".format(i + 1, scores[i]))
         # calculate and print the average performance
        avg_performance_cv = scores.mean()
        print("\nAccuracy: {:.3f}".format(avg_performance_cv))



print("Splitting")
# split the data into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data_scaled,
                                                                    target_encoded,
                                                                    test_size=test_size,
                                                                    random_state=42,
                                                                    stratify=target_encoded)
# display training set information
print("TRAINING SET:")
print("Size: {} %\n"
      "Data: {}\n"
      "Target: {}\n".format(train_size*100,
                            len(data_train),
                            len(target_train)))

# display training set information
print("TEST SET:")
print("Size: {} %\n"
      "Data: {}\n"
      "Target: {}\n".format(test_size*100,
                            len(data_test),
                            len(target_test)))
#training
model.fit(data_train, target_train)
#prediction
# TO DO
predictions_test = model.predict(data_test)
# computes accuracy of predictions
accuracy = accuracy_score(target_test, predictions_test)
# display accuracy
print("Accuracy: {:.3f}".format(accuracy))

#SAVE INTO A FILE
with open(result_file, 'w') as f:
    f.write("=== MACHINE LEARNING CLASSIFICATION REPORT ===\n\n")

    # Parametri usati
    f.write("PARAMETRI:\n")
    f.write(f"CSV path: {csv_path}\n")
    f.write(f"Target column: {target_column}\n")
    f.write(f"Normalizzazione: {normalization}\n")
    f.write(f"Feature selection: {feature_selection} ({num_feature} features)\n")
    f.write(f"PCA: {pca} ({num_principal_components} componenti)\n")
    f.write(f"Modello: {model_name}\n")
    if model_name == "svm":
        f.write(f"Best par: {model.best_params_}")
    if model_name == "xgboost":
        f.write(f"Best par: {model.best_params_}")
    f.write(f"Cross-validation: {cross_validation} ({cv_folds} folds)\n")
    f.write(f"Train size: {train_size}\n")
    f.write(f"Test size: {test_size}\n\n")

    # Feature selezionate
    if feature_selection:
        f.write("Feature selezionate:\n")
        for feat in selected_feature_names:
            f.write(f"- {feat}\n")
        f.write("\n")

    # Risultati
    f.write("RISULTATI:\n")
    f.write(f"Accuracy-oneshot: {accuracy:.3f}\n")
    f.write("\nClassification Report-oneshot:\n")
    f.write(classification_report(target_test, predictions_test))
    f.write(f"Result cv-scores: {scores}\n")
    f.write(f"Result cv-average: {avg_performance_cv:.3f}\n")



