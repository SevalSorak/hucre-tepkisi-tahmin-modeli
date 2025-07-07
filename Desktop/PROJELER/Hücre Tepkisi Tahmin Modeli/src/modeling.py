import pandas as pd
import numpy as np
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

def train_classification_model(X, Y, model_type='RandomForest', test_size=0.3, random_state=42):
    """
    Belirtilen veri seti üzerinde bir sınıflandırma modeli eğitir ve değerlendirir.

    Args:
        X (pd.DataFrame): İşlenmiş gen ekspresyon matrisi (örnekler satır, genler sütun).
        Y (np.array): Sayısal olarak kodlanmış pertürbasyon etiketleri.
        model_type (str): Kullanılacak modelin tipi ('RandomForest' veya 'LogisticRegression').
        test_size (float): Test setinin oranı.
        random_state (int): Tekrarlanabilir sonuçlar için random seed.

    Returns:
        tuple: (model, X_test, Y_test, Y_pred, Y_pred_proba, feature_importances_df)
            model: Eğitilmiş makine öğrenimi modeli.
            X_test (pd.DataFrame): Test seti X verisi.
            Y_test (np.array): Test seti gerçek etiketleri.
            Y_pred (np.array): Test seti tahmin edilen etiketleri.
            Y_pred_proba (np.array): Test seti için tahmin edilen olasılıklar.
            feature_importances_df (pd.DataFrame): Özellik önem düzeyleri (RandomForest için,
                                                  LogisticRegression için katsayılar).
    """
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=Y
    )

    # Model seçimi ve eğitimi
    if model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=random_state, max_iter=1000, solver='liblinear') # 'liblinear' küçük datasetler için iyi
    else:
        raise ValueError("Desteklenmeyen model_type. 'RandomForest' veya 'LogisticRegression' seçin.")

    model.fit(X_train, Y_train)

    # Tahminler ve olasılıklar
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)

    # Performans metrikleri
    accuracy = accuracy_score(Y_test, Y_pred)

    try:
        roc_auc = roc_auc_score(Y_test, Y_pred_proba, multi_class='ovr')
    except ValueError:
        roc_auc = np.nan 

    print(f"\nModel Doğruluğu (Accuracy): {accuracy:.4f}")
    print(f"Ortalama ROC AUC Skoru (One-vs-Rest): {roc_auc:.4f}")
    
    report_text = classification_report(Y_test, Y_pred, output_dict=False) 

    # Özellik önem düzeyleri
    feature_importances_df = pd.DataFrame()
    if hasattr(model, 'feature_importances_'):
        feature_importances_df = pd.DataFrame({
            'Gene_ID': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif hasattr(model, 'coef_'): 
        avg_coef_importance = np.mean(np.abs(model.coef_), axis=0)
        feature_importances_df = pd.DataFrame({
            'Gene_ID': X.columns,
            'Importance': avg_coef_importance
        }).sort_values(by='Importance', ascending=False)

    return model, X_test, Y_test, Y_pred, Y_pred_proba, feature_importances_df, report_text

if __name__ == '__main__':
    
    # data_preprocessing.py'yi import etmeden önce, ana klasörü sys.path'e eklemeliyiz.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir) 

    from preprocessing import load_and_preprocess_data

    data_path_ge = os.path.join(parent_dir, 'data', 'processed_gene_expression_matrix.csv')
    data_path_meta = os.path.join(parent_dir, 'data', 'sample_metadata.csv')

    X_proc, Y_enc, label_enc = load_and_preprocess_data(
        gene_expression_filepath=data_path_ge,
        metadata_filepath=data_path_meta
    )

    if X_proc is not None:
        print("\nModel eğitimi:")
        model_rf, X_test_rf, Y_test_rf, Y_pred_rf, Y_pred_proba_rf, feature_imp_rf, report_rf = \
            train_classification_model(X_proc, Y_enc, model_type='RandomForest')
        
        print("\nRandom Forest Sınıflandırma Raporu:\n", report_rf)
    else:
        print("Veri ön işleme hatası nedeniyle test modeli eğitilemedi.")

