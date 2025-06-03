import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#primo grafico
def plot_model_accuracies(data: pd.DataFrame, title="Accuracy dei modelli sui dataset"):
    #Plot a grouped bar chart showing the accuracy of different models on multiple datasets.
    models = ['KNN', 'RandomForest', 'SVM', 'XGBoost']  # Ordered as requested
    bar_width = 0.2
    x = np.arange(len(data['Dataset']))

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        if model in data.columns:
            plt.bar(x + i * bar_width, data[model], width=bar_width, label=model)

    plt.xticks(x + bar_width * (len(models) - 1) / 2, data['Dataset'])
    plt.ylim(0, 100)
    plt.xlabel("Dataset")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend(title="Modello")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#dataset img senza denoised, primo grafico
data1 = pd.DataFrame({
    'Dataset': ['D0_1', 'D1_1', 'D2_1', 'D3_1', 'D4_1'],
    'KNN': [67.18, 68.51, 70.78, 84.37, 73.12],
    'RandomForest': [75.46, 79.76, 83.9, 90.15,	82.89],
    'SVM': [79.14, 86.4, 87.65,	95.31, 88.51],
    'XGBoost': [74.29, 81.78, 88.43, 96.17,	88.28]
})
#dataset img con denoised, primo grafico
data2 = pd.DataFrame({
    'Dataset': ['D0_2', 'D1_2', 'D2_2', 'D3_2', 'D4_2'],
    'KNN': [66.87, 67.57, 73.82, 78.36,	80.55],
    'RandomForest': [75.39,	79.68,	84.84,	83.75,	87.81],
    'SVM': [80.31,	83.82,	86.95,	91.8,	93.05],
    'XGBoost': [75,81.71, 88.82, 90.15,	91.72]
})

#secondo grafico, confusion matrix
def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix", normalize=False, figsize=(8, 6), cmap='Blues'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=class_labels, yticklabels=class_labels, cbar=True)

    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()
#Matrici confusione (aggregata) top 5 esperimenti, secondo grafico.
cm1 = np.array([
    [153, 1, 1, 2, 1, 1, 0, 1],
    [0, 150, 1, 0, 1, 8, 0, 0],
    [1, 0, 153, 0, 0, 4, 0, 2],
    [1, 0, 1, 157, 0, 0, 1, 0],
    [0, 0, 1, 2, 156, 1, 0, 0],
    [0, 1, 2, 4, 5, 148, 0, 0],
    [0, 0, 0, 0, 0, 0, 160, 0],
    [2, 1, 2, 0, 0, 1, 0, 154]
])
cm2 = np.array([
    [156, 0, 3, 0, 0, 0, 0, 1],
    [0, 158, 0, 0, 0, 2, 0, 0],
    [2, 0, 154, 1, 1, 1, 0, 1],
    [1, 1, 2, 155, 0, 0, 0, 1],
    [0, 0, 3, 7, 139, 7, 4, 0],
    [0, 4, 4, 3, 3, 145, 1, 0],
    [0, 0, 2, 0, 2, 0, 156, 0],
    [2, 0, 0, 1, 0, 0, 0, 157]
])
cm3 = np.array([
    [153, 2, 3, 0, 0, 0, 0, 2],
    [0, 155, 0, 0, 0, 5, 0, 0],
    [5, 0, 145, 3, 2, 2, 0, 3],
    [0, 0, 3, 147, 2, 5, 2, 1],
    [0, 1, 3, 4, 138, 7, 7, 0],
    [1, 4, 2, 3, 4, 144, 1, 1],
    [0, 0, 2, 4, 1, 1, 152, 0],
    [1, 0, 0, 0, 1, 1, 0, 157]
])
cm4 = np.array([
    [155, 0, 0, 5, 0, 0, 0, 2],
    [0, 152, 0, 0, 0, 8, 0, 0],
    [3, 0, 148, 3, 0, 4, 1, 1],
    [7, 0, 9, 135, 2, 4, 0, 3],
    [0, 1, 2, 1, 140, 14, 1, 1],
    [2, 6, 3, 4, 11, 134, 0, 0],
    [0, 0, 3, 1, 0, 0, 156, 0],
    [1, 0, 1, 1, 0, 0, 2, 155]
])
cm5 = np.array([
    [150, 0, 0, 7, 1, 1, 0, 1],
    [0, 149, 1, 0, 1, 9, 0, 0],
    [3, 0, 150, 1, 3, 3, 0, 0],
    [2, 2, 9, 139, 2, 4, 2, 0],
    [0, 0, 1, 4, 146, 6, 2, 1],
    [3, 1, 3, 5, 8, 136, 0, 4],
    [0, 0, 1, 2, 1, 2, 154, 0],
    [2, 0, 3, 1, 1, 1, 2, 150]
])
labels = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']


#terzo grafico, acc per fold
def plot_fold_accuracies(accuracies, model_name="Modello"):
    folds = [f"FOLD{i}" for i in range(len(accuracies))]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(folds, accuracies, color="skyblue")
    plt.ylim(0, 100)
    plt.title(f"Accuracy per Fold – {model_name}")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Fold")
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Annotazioni dentro le barre
    for bar, acc in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            acc - 4,  # posizionamento dentro la barra
            f"{acc:.1f}%",
            ha='center',
            va='top',
            color='black',
            fontsize=10
        )

    plt.tight_layout()
    plt.show()
accfold1 = [96.48, 98.44, 96.48, 96.88, 96.88]
accfold2 = [95.31, 97.27, 96.88, 93.36, 93.75]
accfold3 = [94.92, 94.53, 92.97, 93.75, 91.8]
accfold4 = [93.75, 93.36, 91.41, 92.58, 89.45]
accfold5 = [91.41, 92.97, 91.8, 91.02, 93.36]


#quarto grafico: acc best ofthe best
def plot_summary_accuracy(data: pd.DataFrame, title="Accuracy media dei metodi"):
    plt.figure(figsize=(7, 6))  # Dimensioni aumentate per chiarezza
    bars = plt.bar(data['Metodo'], data['Accuracy'], color=['steelblue', 'darkorange', 'seagreen'])

    max_acc = data['Accuracy'].max()
    y_max = max_acc + (2 if max_acc > 1 else 0.02)
    plt.ylim(0, y_max)

    plt.ylabel("Accuracy" + (" (%)" if max_acc > 1 else ""))
    plt.title(title)

    # Etichette dentro le barre
    for bar, acc in zip(bars, data['Accuracy']):
        label = f"{acc:.2f}%" if acc > 1 else f"{acc:.2%}"
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() * 0.5,  # A metà barra
                 label, ha='center', va='center', fontsize=12, color='white', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
#dati per il plot4
databestofbest =  pd.DataFrame({
    'Metodo': ['Best Model', 'Majority Voting', 'Soft Voting'],
    'Accuracy': [96.17, 97.42, 97.27 ]
})

#quinto grafico: conteggio scelte pipeline
def plot_config_frequencies(frequencies, labels, title="Frequenze delle Tecniche Scelte"):
    plt.figure(figsize=(12, 6))

    # Colori diversi per i due gruppi
    colors = ['cornflowerblue'] * 3 + ['mediumseagreen'] * 3

    bars = plt.bar(labels, frequencies, color=colors)
    plt.ylabel("Conteggio")
    plt.xticks(rotation=30, ha='right')
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Scrive i valori DENTRO le barre
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height - 0.5,
                 f"{int(freq)}", ha='center', va='top', color='white', fontsize=10)

    plt.tight_layout()
    plt.show()

frequenze = [5, 26, 9, 14, 16, 10]
etichette5 = ["No feature Reduction", "PCA", "Feature Selection",
             "No normalization", "StandardScaler norm.", "MinMax norm"]


#sesto
frequenze6_1 = [4, 0, 6, 0, 5, 5]
frequenze6_2 = [0, 9, 1, 7, 3, 0]

#primo grafico, bar plot
#plot_model_accuracies(data1, "Confronto prestazioni medie dei 5 dataset (senza denoised)")
#plot_model_accuracies(data2, "Confronto prestazioni medie dei 5 dataset (con denoised)")


#secondo grafico, heatmap, confusion matrix, best 5 experiment
#plot_confusion_matrix(cm1, class_labels=labels, title="Confusion Matrix - XGBOOST-D3_1", normalize=False)
#plot_confusion_matrix(cm2, class_labels=labels, title="Confusion Matrix - SVM-D3_1", normalize=False)
#plot_confusion_matrix(cm3, class_labels=labels, title="Confusion Matrix - SVM-D4_2", normalize=False)
#plot_confusion_matrix(cm4, class_labels=labels, title="Confusion Matrix - SVM-D3_2", normalize=False)
#plot_confusion_matrix(cm5, class_labels=labels, title="Confusion Matrix - XGBOOST-D4_2", normalize=False)

#terzo grafico, acc for fold
#plot_fold_accuracies(accfold1, "Accuracy per fold - XGBOOST-D3_1")
#plot_fold_accuracies(accfold2, "Accuracy per fold - SVM-D3_1")
#plot_fold_accuracies(accfold3, "Accuracy per fold - SVM-D4_2")
#plot_fold_accuracies(accfold4, "Accuracy per fold - SVM-D3_2")
#plot_fold_accuracies(accfold5, "Accuracy per fold - XGBOOST-D4_2")


#quarto grafico, confronto tra miglior modello, ensemble
#plot_summary_accuracy(databestofbest, "Confronto prestazioni best model ed ensemble")

#quinto grafico, conteggio scelte pipeline
#plot_config_frequencies(frequenze, etichette5)

#sesto grafico, per svm e xgboost cosa sceglie di solito
#plot_config_frequencies(frequenze6_1, etichette5, "Frequenze delle Tecniche Scelte SVM")
#plot_config_frequencies(frequenze6_2, etichette5, "Frequenze delle Tecniche Scelte XGBOOST")

