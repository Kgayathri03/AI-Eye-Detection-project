import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Simulated ground truth and predictions for three classes: DR, Glaucoma, Cataracts
np.random.seed(42)
y_true = np.random.choice([0, 1, 2], size=300, p=[0.33, 0.33, 0.34])  # True labels
y_pred = y_true.copy()
y_pred[np.random.choice(range(300), size=30, replace=False)] = np.random.choice([0, 1, 2], size=30)  # Introduce errors

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["DR", "Glaucoma", "Cataracts"], yticklabels=["DR", "Glaucoma", "Cataracts"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve & AUC
n_classes = 3
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
y_pred_bin = label_binarize(y_pred, classes=[0, 1, 2])
plt.figure(figsize=(7,5))
colors = ['blue', 'green', 'red']
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class Classification")
plt.legend()
plt.show()

# Grad-CAM Simulation
image = np.ones((224, 224, 3), dtype=np.uint8) * 180  # Simulated retinal image
heatmap = np.random.rand(224, 224) * 255  # Randomized attention heatmap
heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
plt.figure(figsize=(5, 5))
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Simulated Grad-CAM Visualization")
plt.show()
