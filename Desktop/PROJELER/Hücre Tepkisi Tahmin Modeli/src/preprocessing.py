import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import re

def _load_data(gene_expression_filepath, metadata_filepath):
    """
    Yardımcı fonksiyon: Gen ekspresyon matrisini ve örnek metadata'sını CSV dosyalarından yükler.
    """
    print("Veri dosyaları yükleniyor...")
    try:
        gene_expression_df = pd.read_csv(gene_expression_filepath, index_col='ID_REF')
        sample_metadata_df = pd.read_csv(metadata_filepath)
        print(f"Gen Ekspresyon Matrisi Boyutu (Yüklendi): {gene_expression_df.shape}")
        print(f"Sample Metadata Boyutu (Yüklendi): {sample_metadata_df.shape}")
        return gene_expression_df, sample_metadata_df
    except FileNotFoundError as e:
        print(f"Hata: Dosya bulunamadı - {e.filename}. Lütfen dosya yollarını kontrol edin.")
        return None, None
    except Exception as e:
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        return None, None

def _clean_and_filter_dataframes(gene_expression_df, sample_metadata_df):
    """
    Yardımcı fonksiyon: DataFrame'lerdeki sütun ve indeks adlarındaki boşlukları temizler
    ve metadata'yı gen ekspresyon matrisindeki ilgili örnek ID'lerine göre filtreler.
    """
    gene_expression_df.columns = gene_expression_df.columns.str.strip()
    sample_metadata_df['sample_id'] = sample_metadata_df['sample_id'].str.strip()

    relevant_sample_ids = gene_expression_df.columns.tolist()
    metadata_filtered = sample_metadata_df[sample_metadata_df['sample_id'].isin(relevant_sample_ids)].copy()

    metadata_filtered.set_index('sample_id', inplace=True)

    if 'Sample_title' in metadata_filtered.columns:
        metadata_filtered['Sample_title'] = metadata_filtered['Sample_title'].str.strip()
    
    return gene_expression_df, metadata_filtered


def _extract_and_encode_labels(gene_expression_columns, metadata_filtered_df):
    """
    Yardımcı fonksiyon: Metadata'dan pertürbasyon etiketlerini çıkarır ve sayısal olarak kodlar.
    """
    print("Pertürbasyon etiketleri çıkarılıyor...")
    perturbation_labels = []

    metadata_filtered_df['Sample_title_Processed'] = metadata_filtered_df['Sample_title'].astype(str).str.strip().str.upper()
    metadata_filtered_df['Sample_description_Processed'] = metadata_filtered_df['Sample_description'].astype(str).str.strip().str.upper()

    for sample_id in gene_expression_columns:
        row = metadata_filtered_df.loc[sample_id]
        description_processed = row['Sample_description_Processed']
        characteristics = row['Sample_characteristics_ch1']
        sample_title_processed = row['Sample_title_Processed']

        treatment_type = "UNKNOWN_NP_TYPE" # Varsayılan olarak "BİLİNMEYEN" NP tipi

        # Sadece 3 ana sınıf: SM30_9nm_NP, AS30_18nm_NP (3h), AS30_18nm_NP (22h)
        if "SM30" in sample_title_processed:
            treatment_type = "SM30_9nm_NP"
        elif "AS30" in sample_title_processed:
            treatment_type = "AS30_18nm_NP"
        else:
            print(f"Uyarı: '{sample_id}' için nanopartikül tipi belirlenemedi, varsayılan atandı: {treatment_type}")

        # Zaman bilgisi (3hrs veya 22hrs)
        time_match = re.search(r'recovery time: (\d+)\s*hrs', characteristics)
        time = f"{time_match.group(1)}hrs" if time_match else "UNKNOWN_TIME_ERROR"

        final_label = f"{treatment_type}_{time}"
        perturbation_labels.append(final_label)

    metadata_filtered_df.drop(columns=['Sample_title_Processed', 'Sample_description_Processed'], inplace=True, errors='ignore')

    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(perturbation_labels)
    Y_labels_decoded = label_encoder.inverse_transform(Y_encoded)
    print("Oluşturulan Pertürbasyon Etiketleri (İlk 5):", Y_labels_decoded[:5])
    print("Etiketlerin Dağılımı:\n", pd.Series(Y_labels_decoded).value_counts())

    metadata_filtered_df['perturbation_label'] = perturbation_labels

    return Y_encoded, label_encoder, metadata_filtered_df

def _normalize_expression_data(gene_expression_df, metadata_filtered_df):
    """
    Yardımcı fonksiyon: Gen ekspresyon matrisini transpoze eder,
    log2 dönüşümü ve standart ölçeklendirme ile normalize eder.
    """
    print("Gen ekspresyon verisi normalize ediliyor...")

    X = gene_expression_df.T
    print(f"Gen Ekspresyon Matrisi Boyutu (Transpoze Edilmiş): {X.shape}")

    X = X.loc[metadata_filtered_df.index]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.fillna(X.mean())

    X_log2 = np.log2(X + 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log2)
    
    X_processed_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
    X_processed_df = X_processed_df.infer_objects(copy=False)

    print("Veri ön işleme tamamlandı.")
    return X_processed_df


def load_and_preprocess_data(gene_expression_filepath=r'data\processed_gene_expression_matrix.csv',
                             metadata_filepath=r'data\sample_metadata.csv',
                             soft_file_path=r'data\GSE53700_family.soft'):
    """
    Gen ekspresyon matrisini ve örnek metadata'sını yükler, işler ve makine öğrenimi için hazırlar.

    Adımlar:
    1. CSV dosyalarını Pandas DataFrame'lerine yükler.
    2. Metadata'dan pertürbasyon etiketlerini (Control/SM30_NP ve 3hrs/22hrs) çıkarır.
    3. Gen ekspresyon matrisini transpoze eder (örnekler satır, genler sütun).
    4. Gen ekspresyon değerlerini log2 dönüşümü ve standart ölçeklendirme ile normalize eder.
    5. Birleştirilmiş pertürbasyon etiketlerini ve işlenmiş gen ekspresyon matrisini döndürür.

    Args:
        gene_expression_filepath (str): İşlenmiş gen ekspresyon matrisinin CSV yolu.
        metadata_filepath (str): Örnek metadata CSV yolu.

    Returns:
        tuple: (X_processed_df, Y_labels, label_encoder_obj)
            X_processed_df (pd.DataFrame): Normalize edilmiş ve transpoze edilmiş gen ekspresyon matrisi.
                                           Satırlar örnekler (sample_id), sütunlar genler (ID_REF).
            Y_labels (np.array): Sayısal olarak kodlanmış pertürbasyon etiketleri.
            label_encoder_obj (sklearn.preprocessing.LabelEncoder): Etiketleri tekrar çözmek için kullanılan LabelEncoder objesi.
    """
    print("Veri yükleniyor ve ön işleme tabi tutuluyor...")

    from read_soft_file import process_soft_file
    platform_df, _, _ = process_soft_file(soft_file_path)
    if platform_df is None:
        return None, None, None, None

    gene_expression_df, sample_metadata_df = _load_data(gene_expression_filepath, metadata_filepath)
    if gene_expression_df is None or sample_metadata_df is None:
        return None, None, None, platform_df

    gene_expression_df, metadata_filtered_df = _clean_and_filter_dataframes(gene_expression_df, sample_metadata_df)

    Y_encoded, label_encoder, metadata_filtered_df = _extract_and_encode_labels(
        gene_expression_df.columns, metadata_filtered_df
    )

    X_processed_df = _normalize_expression_data(gene_expression_df, metadata_filtered_df)

    print("Veri ön işleme tamamlandı.")
    return X_processed_df, Y_encoded, label_encoder, platform_df