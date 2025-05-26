import zipfile
import os
import pathlib
import numpy as np
from skimage import io, exposure, filters, feature
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_curve, auc, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from collections import Counter
import random  

curPath = pathlib.Path(__file__).parent.resolve()

csv_file_path = f"{curPath}/dados_processados.csv"

def salvar_dados_em_csv(X, y, caminho):
    df = pd.DataFrame(X)
    df['target'] = y
    df.to_csv(caminho, index=False)
    print(f"\nDados salvos em {caminho}")

def carregar_dados_do_csv(caminho):
    return pd.read_csv(caminho)

def preprocess_image(image_path, background_tissue):
    img = io.imread(image_path)
    img = exposure.equalize_adapthist(img)
    if background_tissue == 'D':
        img = filters.frangi(img, sigmas=[1, 2])
    return img

def extract_features(image_path, background_tissue):
    try:
        img = preprocess_image(image_path, background_tissue)
        img_uint8 = (img * 255).astype(np.uint8)

        glcm = feature.graycomatrix(img_uint8, distances=[1, 2, 3], angles=[0, np.pi/4, np.pi/2], levels=256, symmetric=True, normed=True)
        contrast = feature.graycoprops(glcm, 'contrast').mean()
        dissimilarity = feature.graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = feature.graycoprops(glcm, 'homogeneity').mean()
        asm = feature.graycoprops(glcm, 'ASM').mean()
        correlation = feature.graycoprops(glcm, 'correlation').mean()

        features = [
            np.mean(img),
            np.std(img),
            np.sum(img > 0.8) / img.size,
            contrast,
            dissimilarity,
            homogeneity,
            asm,
            correlation
        ]
        return features
    except Exception as e:
        print(f"Erro ao processar {image_path}: {e}")
        return None

if os.path.exists(csv_file_path):
    print(f"\nCarregando dados de {csv_file_path}")
    df_features = carregar_dados_do_csv(csv_file_path)
    X = df_features.drop(columns=['target']).values
    y = df_features['target'].values
else:
    caminho = f"{curPath}/mias_dataset/all-mias"
    arquivos = os.listdir(caminho)
    print(f"{len(arquivos)} arquivos encontrados")
    print(arquivos[:5])

    metadata = []
    with open(f"{curPath}/mias_dataset/Info.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("#", "REFNUM", "=")):
                continue
            parts = line.split()
            entry = {
                'ref_num': parts[0],
                'background_tissue': parts[1],
                'class_label': parts[2],
                'severity': parts[3] if len(parts) > 3 else None,
                'x': int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None,
                'y': int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None,
                'radius': int(parts[6]) if len(parts) > 6 and parts[6].isdigit() else None
            }
            metadata.append(entry)

    X, y, failed = [], [], []
    for entry in tqdm(metadata, desc="Processando"):
        img_path = f"{curPath}/mias_dataset/all-mias/{entry['ref_num']}.pgm"
        if os.path.exists(img_path):
            features = extract_features(img_path, entry['background_tissue'])
            if features:
                X.append(features)
                y.append(0 if entry['class_label'] == 'NORM' else 1)
        else:
            failed.append(entry['ref_num'])

    print(f"\nSucesso: {len(X)} | Falhas: {len(failed)}")
    if failed:
        print(f"Falhas: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    salvar_dados_em_csv(X, y, csv_file_path)

print("\nDistribuição das classes:", Counter(y))

modelo = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200, max_depth=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision_normal = precision_score(y_test, y_pred, pos_label=0)
precision_anomalia = precision_score(y_test, y_pred, pos_label=1)
recall_normal = recall_score(y_test, y_pred, pos_label=0)
recall_anomalia = recall_score(y_test, y_pred, pos_label=1)

print("\n=== Métricas por Classe ===")
print(f"Acurácia Total: {accuracy:.2f}")
print(f"\nNormal:")
print(f" - Precisão: {precision_normal:.2f}")
print(f" - Recall: {recall_normal:.2f}")
print(f"\nAnomalia:")
print(f" - Precisão: {precision_anomalia:.2f}")
print(f" - Recall: {recall_anomalia:.2f}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=['Normal', 'Anomalia'],
                                        cmap='Blues', ax=plt.gca())
plt.title("Matriz de Confusão")

plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title("Curva ROC")
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== Interpretação ===")
print("• AUC ~0.50: Modelo equivalente a chute aleatório.")
print("• Recall Anomalia 0%: Não detectou NENHUMA anomalia.")
print("• Alta precisão em Normal: Apenas classifica tudo como 'normal'.")

class ImageTest:
    def __init__(self, ax):
        self.ax = ax
        self.image_name = None
        self.img = None
        self.pred = None
        self.proba = None

    def test_random_image(self):
        random_index = random.randint(1, 329)
        image_name = f"{curPath}/mias_dataset/all-mias/mdb{random_index:03d}.pgm"
        self.image_name = image_name

        features = extract_features(image_name, background_tissue='F')
        if features:
            self.pred = modelo.predict([features])[0]
            self.proba = modelo.predict_proba([features])[0][1]

            self.img = io.imread(image_name)
            self.ax.imshow(self.img, cmap='gray')
            self.ax.set_title(f"Classe: {'Anomalia' if self.pred == 1 else 'Normal'}\nProbabilidade: {self.proba:.2f}")
            self.ax.axis('off')  
            plt.draw() 
        else:
            print(f"Erro ao processar {image_name}")

    def button_callback(self, event):
        self.test_random_image()

fig, ax = plt.subplots()
image_test = ImageTest(ax)

ax_button = plt.axes([0.7, 0.01, 0.2, 0.075])
button = Button(ax_button, 'Testar Outra Imagem')
button.on_clicked(image_test.button_callback)

image_test.test_random_image()

plt.show()
