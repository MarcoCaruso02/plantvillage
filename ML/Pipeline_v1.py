from scipy.cluster.hierarchy import weighted
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, make_scorer, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


def pipeline_creation(model_name):
    pipeline = Pipeline([
    ])
    if(model_name=="RF"):
        pipeline = Pipeline([
            ('normalization', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('dim_reduction', 'passthrough'),
            ('classifier', RandomForestClassifier()),
        ])
    elif(model_name=="SVM"):
        pipeline = Pipeline([
            ('normalization', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('dim_reduction', 'passthrough'),
            ('classifier', SVC()),
        ])
    elif (model_name == "XGBOOST"):
        pipeline = Pipeline([
            ('normalization', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('dim_reduction', 'passthrough'),
            ('classifier', XGBClassifier()),
        ])
    elif (model_name == "KNN"):
        pipeline = Pipeline([
            ('normalization', 'passthrough'),
            ('feature_selection', 'passthrough'),
            ('dim_reduction', 'passthrough'),
            ('classifier', KNeighborsClassifier()),
        ])
    return pipeline

def parameters_span(dataset_name):
    par_span=[]
    if(dataset_name=="LBP_8_(D0_1).csv" or dataset_name=="LBP_8_den(D0_2).csv"):
        par_span=[5,10,15]
    elif(dataset_name=="LBP_8_12_16(D1_1).csv" or dataset_name=="LBP_8_12_16_den(D1_2).csv"):
        par_span = [10, 30, 50, 70]
    elif(dataset_name=="LBP_max_GLCM(D2_1).csv" or dataset_name=="LBP_max_GLCM(D2_2).csv"):
        par_span=[30,50,70,90]
    return par_span


def param_grid_creation(dataset_name, model_name):
    param_grid=[]
    if(model_name=="RF"):
        param_grid = [
            # Only feature selection, no PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': [SelectKBest(f_classif)],
                'feature_selection__k': parameters_span(dataset_name),
                'dim_reduction': ['passthrough'],
                'classifier': [RandomForestClassifier()]
            },
            # Only PCA, no feature selection
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': [PCA()],
                'dim_reduction__n_components': parameters_span(dataset_name),
                'classifier': [RandomForestClassifier()]
            },
            # No feature selection or PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': ['passthrough'],
                'classifier': [RandomForestClassifier()]
            }
        ]
    elif(model_name=="SVM"):
        param_grid = [
            # Only feature selection, no PCA
            *[
                {
                    'normalization': [StandardScaler(), MinMaxScaler()],
                    'feature_selection': [SelectKBest(f_classif)],
                    'feature_selection__k': parameters_span(dataset_name),
                    'dim_reduction': ['passthrough'],
                    'classifier': [SVC()],
                    'classifier__kernel': [kernel],
                    **kernel_params
                }
                for kernel, kernel_params in {
                    'linear': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
                    'rbf': {'classifier__C': [0.01, 0.1, 1, 10, 100],
                            'classifier__gamma': [1e-3, 1e-2, 0.1, 'scale', 'auto']},
                    'poly': {'classifier__C': [0.1, 1], 'classifier__degree': [2, 3],
                             'classifier__gamma': ['scale', 'auto']}
                }.items()
            ],

            # Only PCA, no feature selection
            *[
                {
                    'normalization': [StandardScaler(), MinMaxScaler()],
                    'feature_selection': ['passthrough'],
                    'dim_reduction': [PCA()],
                    'dim_reduction__n_components': parameters_span(dataset_name),
                    'classifier': [SVC()],
                    'classifier__kernel': [kernel],
                    **kernel_params
                }
                for kernel, kernel_params in {
                    'linear': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
                    'rbf': {'classifier__C': [0.01, 0.1, 1, 10, 100],
                            'classifier__gamma': [1e-3, 1e-2, 0.1, 'scale', 'auto']},
                    'poly': {'classifier__C': [0.1, 1], 'classifier__degree': [2, 3],
                             'classifier__gamma': ['scale', 'auto']}
                }.items()
            ],

            # No feature selection or PCA
            *[
                {
                    'normalization': [StandardScaler(), MinMaxScaler()],
                    'feature_selection': ['passthrough'],
                    'dim_reduction': ['passthrough'],
                    'classifier': [SVC()],
                    'classifier__kernel': [kernel],
                    **kernel_params
                }
                for kernel, kernel_params in {
                    'linear': {'classifier__C': [0.01, 0.1, 1, 10, 100]},
                    'rbf': {'classifier__C': [0.01, 0.1, 1, 10, 100],
                            'classifier__gamma': [1e-3, 1e-2, 0.1, 'scale', 'auto']},
                    'poly': {'classifier__C': [0.1, 1], 'classifier__degree': [2, 3],
                             'classifier__gamma': ['scale', 'auto']}
                }.items()
            ]
        ]
    elif(model_name=="XGBOOST"):
        param_grid = [
            # Only feature selection, no PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': [SelectKBest(f_classif)],
                'feature_selection__k': parameters_span(dataset_name),
                'dim_reduction': ['passthrough'],
                'classifier': [XGBClassifier()],
                "classifier__n_estimators": [10, 50],
                "classifier__learning_rate": [None, 0.1, 1],
                "classifier__booster": ["gbtree", "gblinear"],
                "classifier__max_depth": [3, 6, 12],
                "classifier__min_child_weight": [0.5, 1, 2],
                "classifier__gamma": [0, 1, 10]
            },
            # Only PCA, no feature selection
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': [PCA()],
                'dim_reduction__n_components': parameters_span(dataset_name),
                'classifier': [XGBClassifier()],
                "classifier__n_estimators": [10, 50],
                "classifier__learning_rate": [None, 0.1, 1],
                "classifier__booster": ["gbtree", "gblinear"],
                "classifier__max_depth": [3, 6, 12],
                "classifier__min_child_weight": [0.5, 1, 2],
                "classifier__gamma": [0, 1, 10]
            },
            # No feature selection or PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': ['passthrough'],
                'classifier': [XGBClassifier()],
                "classifier__n_estimators": [10, 50],
                "classifier__learning_rate": [None, 0.1, 1],
                "classifier__booster": ["gbtree", "gblinear"],
                "classifier__max_depth": [3, 6, 12],
                "classifier__min_child_weight": [0.5, 1, 2],
                "classifier__gamma": [0, 1, 10]
            }
        ]
    elif (model_name == "KNN"):
        param_grid = [
            # Only feature selection, no PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': [SelectKBest(f_classif)],
                'feature_selection__k': parameters_span(dataset_name),
                'dim_reduction': ['passthrough'],
                'classifier': [KNeighborsClassifier()],
                "classifier__n_neighbors": [5,10,15,20],
            },
            # Only PCA, no feature selection
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': [PCA()],
                'dim_reduction__n_components': parameters_span(dataset_name),
                'classifier': [KNeighborsClassifier()],
                "classifier__n_neighbors": [5,10,15,20],
            },
            # No feature selection or PCA
            {
                'normalization': [StandardScaler(), MinMaxScaler(), 'passthrough'],
                'feature_selection': ['passthrough'],
                'dim_reduction': ['passthrough'],
                'classifier': [KNeighborsClassifier()],
                "classifier__n_neighbors": [5,10,15,20],
            }
        ]
    return param_grid


print("\nDataset Loading...\n")

dataset_name="LBP&GLCM.csv"
model_name="KNN"
result_file="provaKNN.txt"
#DATASET LOADING

df_all = pd.read_csv(dataset_name)
target_column = "label"


data = df_all.drop(columns=[target_column])
print("Missing values:", data.isnull().sum().sum(), " | Infinite values:", np.isinf(data).sum().sum())

target = df_all[target_column]
#encoding (no categorical feature, maybe just add a check for the prof). Maybe is needed encoding of the target
label_encoder = LabelEncoder()
target = pd.Series(label_encoder.fit_transform(target))

print("Dataset has been loaded correctly\n")

#PIPELINE CREATION

print("Pipeline creating...\n")

pipeline = pipeline_creation(model_name)
print("Pipeline created correctly\n")

print("Grid Search parameters creating...\n")

#GRID SEARCH PARAMETERS

param_grid = param_grid_creation(dataset_name,model_name)

print("Grid Search parameters created correctly\n")

scorer = make_scorer(accuracy_score)

best_params_list = []  # gather grid_search.best_params_ from each fold

# Outer CV: 5 folds
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
test_accuracies = []
test_precision=[]
test_recall=[]
test_f1score=[]


all_y_true = []
all_y_pred = []
fold_conf_matrices = []
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(data, target), 1):
    print(f"\n===== Outer Fold {fold_idx} =====\n")

    X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    # Inner CV inside GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,  # inner 5-fold CV for hyperparameter tuning
        scoring=scorer,
        verbose=1,
        n_jobs=-1
    )

    print("Grid Search object created successfully\n")

    grid_search.fit(X_train, y_train)

    best_params_list.append(grid_search.best_params_)
    print("Best parameters: ", grid_search.best_params_)
    print("Best CV score (inner): ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    # Evaluate on the outer test fold
    y_pred = best_model.predict(X_test)

    #in order to compute the confusion matrix for each fold e for the aggregated version
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    cm_fold = confusion_matrix(y_test, y_pred)
    fold_conf_matrices.append(cm_fold)

    accuracy = accuracy_score(y_test, y_pred)
    f1=f1_score(y_test,y_pred,average='weighted')
    precision= precision_score(y_test,y_pred,average='weighted')
    recall=recall_score(y_test,y_pred,average='weighted')

    print(f"Test accuracy on outer fold {fold_idx}: {accuracy:.4f}")
    print(f"Test f1 score on outer fold {fold_idx}: {f1:.4f}")
    print(f"Test precision on outer fold {fold_idx}: {precision:.4f}")
    print(f"Test recall on outer fold {fold_idx}: {recall:.4f}")
    test_accuracies.append(accuracy)
    test_f1score.append(f1)
    test_recall.append(recall)
    test_precision.append(precision)


cm_agg = confusion_matrix(all_y_true, all_y_pred)

mean_accuracy = np.mean(test_accuracies)
std_accuracy = np.std(test_accuracies)
mean_f1=np.mean(test_f1score)
std_f1=np.std(test_f1score)
mean_precision=np.mean(test_precision)
std_precision=np.std(test_precision)
mean_recall=np.mean(test_recall)
std_recall=np.std(test_recall)

#AVERAGE PERFORMANCE THROUGH ALL EXPERIMENTS

print("\n===== Summary over Outer Folds =====")
print(f"Average Test Accuracy: {mean_accuracy:.4f}")
print(f"Standard Deviation:    {std_accuracy:.4f}")
from collections import Counter

#Best parameters print (majority)

most_common_params = Counter(tuple(sorted(p.items())) for p in best_params_list).most_common(1)
print("Most common best parameters:\n")
for key, value in most_common_params:
        print(f"  {key}: {value}")

#Save results on file

with open(result_file, 'w') as f:
    f.write("=== MACHINE LEARNING CLASSIFICATION REPORT ===\n\n")
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"Model: {model_name}\n")
    f.write("Best configuration:\n")
    for key, value in most_common_params:
        f.write(f"  {key}: {value}\n")
    f.write(f"Average accuracy: {mean_accuracy*100}%\n")
    f.write(f"Average deviation: {std_accuracy*100}%\n")
    f.write(f"Average f1 score: {mean_f1 * 100}%\n")
    f.write(f"Average f1 score: {std_f1 * 100}%\n")
    f.write(f"Average precision: {mean_precision * 100}%\n")
    f.write(f"Average precision: {std_precision * 100}%\n")
    f.write(f"Average recall: {mean_recall * 100}%\n")
    f.write(f"Average recall: {std_recall * 100}%\n")
    f.write("\nConfusion Matrices per Fold:\n")
    for i, cm in enumerate(fold_conf_matrices, 1):
        f.write(f"\nFold {i} Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n")

    f.write("\nAggregated Confusion Matrix:\n")
    f.write(np.array2string(cm_agg))
