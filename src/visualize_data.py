# external packages
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def get_confusion_matrix(lbls, preds, class_dict, norm = "true"):
    
    # compute confusion matrix
    cm = (confusion_matrix(lbls, preds, normalize = norm) * 100).round(1)
    
    # generate labels
    labels = [f"{key}: {value[:10]}" for key, value in class_dict.items()]

    # plot confusion matrix
    _, ax = plt.subplots(figsize=(4,3))
    cm = ConfusionMatrixDisplay(cm, display_labels = labels)
    cm.plot(ax = ax, xticks_rotation = 'vertical', cmap = plt.cm.Blues)
    ax.set_xticklabels([f"{key}" for key in list(class_dict.keys())])
    ax.tick_params(axis='x', labelrotation = 0)
    ax.set_title("Confusion Matrix")
    plt.show()