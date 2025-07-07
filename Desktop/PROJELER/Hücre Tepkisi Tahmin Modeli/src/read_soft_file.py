import pandas as pd
import re

def read_platform_table(lines):
    """
    Dosya satırlarından platform tablosunu okur ve DataFrame olarak döndürür.
    """
    platform_table = []
    platform_headers = None
    in_platform = False

    for i, line in enumerate(lines):
        if line.startswith('!platform_table_begin'):
            in_platform = True
            platform_headers = lines[i+1].strip().split('\t')
            continue
        if line.startswith('!platform_table_end'):
            in_platform = False
            break
        if in_platform and platform_headers is not None and i > 0:
            if not line.startswith('!platform_table_begin') and not line.startswith('!platform_table_end'):
                platform_table.append(line.strip().split('\t'))
    
    if platform_headers and platform_table:
        return pd.DataFrame(platform_table, columns=platform_headers)
    return pd.DataFrame()

def read_sample_data(lines):
    """
    Dosya satırlarından sample tablolarını ve metadata'yı okur.
    Sample DataFrame'leri ve metadata sözlüklerini döndürür.
    """
    sample_tables = {}
    sample_metadata = {}
    total_lines = len(lines)

    for i, line in enumerate(lines):
        if line.startswith('!Sample_title = ') or line.startswith('!Sample_geo_accession = '):
            meta = {}
            j = i
            # Metadata'yı oku
            while j < total_lines and not lines[j].startswith('!sample_table_begin'):
                if lines[j].startswith('!Sample_'):
                    keyval = lines[j][1:].split(' = ', 1)
                    if len(keyval) == 2:
                        key, val = keyval
                        meta[key] = val
                j += 1
            
            # Sample tablosunu oku (eğer varsa)
            if j < total_lines and lines[j].startswith('!sample_table_begin'):
                table_start = j
                header_line_index = table_start + 1
                
                # Başlık satırının geçerli olduğundan emin ol
                if header_line_index < total_lines:
                    header = lines[header_line_index].strip().split('\t')
                    data = []
                    k = table_start + 2
                    while k < total_lines and not lines[k].startswith('!sample_table_end'):
                        data.append(lines[k].strip().split('\t'))
                        k += 1
                    
                    sample_name = meta.get('Sample_geo_accession', meta.get('Sample_title', f'sample_{i}')).strip()
                    if header and data: # Başlık ve veri varsa DataFrame oluştur
                        sample_tables[sample_name] = pd.DataFrame(data, columns=header)
                        sample_metadata[sample_name] = meta
                    elif header: # Sadece başlık varsa boş DataFrame oluştur
                         sample_tables[sample_name] = pd.DataFrame(columns=header)
                         sample_metadata[sample_name] = meta
    return sample_tables, sample_metadata

def merge_sample_expression_data(sample_tables, platform_df):
    """
    Tüm sample tablolarını birleştirir ve gen ekspresyon matrisini oluşturur.
    Platform tablosundaki geçerli prob ID'leri ile filtreler.
    """
    merged_df = None
    probe_col = None

    for sample, df in sample_tables.items():
        if df.empty:
            continue # Boş DataFrame'leri atla

        if probe_col is None and not df.columns.empty:
            probe_col = df.columns[0]
        
        if probe_col is None: # Eğer hiç probe sütunu belirlenemezse devam etme
            continue

        # Ekspresyon değer sütunu (genellikle 2. sütun veya 'VALUE' gibi)
        expr_col = None
        if 'VALUE' in df.columns:
            expr_col = 'VALUE'
        elif len(df.columns) > 1:
            expr_col = df.columns[1]
        elif len(df.columns) == 1: # Sadece bir sütun varsa, hem probe hem ekspresyon olabilir
             expr_col = df.columns[0]

        if expr_col is None: # Ekspresyon sütunu bulunamazsa atla
            continue

        sub = df[[probe_col, expr_col]].copy()
        sub = sub.rename(columns={expr_col: sample.strip()}) # Sample adını temizle

        if merged_df is None:
            merged_df = sub
        else:
            merged_df = pd.merge(merged_df, sub, on=probe_col, how='outer')

    if merged_df is not None:
        merged_df = merged_df.set_index(probe_col)
        
        # Sadece platform tablosundaki gerçek prob ID'lerini tut
        # 'ID' sütunu olmayan platform_df durumunu ele al
        if 'ID' in platform_df.columns:
            # Sadece 'A_' ile başlayan ID'leri filtrele (tipik Affymetrix/Agilent prob formatı)
            valid_probes = platform_df[platform_df['ID'].astype(str).str.startswith('A_')]['ID'].tolist()
            merged_df = merged_df[merged_df.index.isin(valid_probes)]
        else:
            print("Uyarı: Platform tablosunda 'ID' sütunu bulunamadı. Tüm problar dahil edilecek.")

        # Tüm sütunları sayısal tiplere dönüştür, dönüştürülemeyenleri NaN yap
        for col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        merged_df = merged_df.dropna(how='all') # Tüm değerleri NaN olan satırları düşür
        merged_df = merged_df.dropna(axis=1, how='all') # Tüm değerleri NaN olan sütunları düşür

    return merged_df

def process_soft_file(file_path):
    """
    Bir SOFT dosyasını okur, platform ve sample verilerini çıkarır,
    gen ekspresyon matrisini birleştirir ve metadata'yı düzenler.
    """
    print(f"'{file_path}' dosyası okunuyor...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = list(f)
    except FileNotFoundError:
        print(f"Hata: '{file_path}' dosyası bulunamadı.")
        return None, None, None
    except Exception as e:
        print(f"Dosya okunurken bir hata oluştu: {e}")
        return None, None, None

    platform_df = read_platform_table(lines)
    sample_tables, sample_metadata = read_sample_data(lines)
    merged_expression_df = merge_sample_expression_data(sample_tables, platform_df)

    # Sample metadata'yı DataFrame'e dönüştür ve temizle
    metadata_list = []
    for sample_id, meta in sample_metadata.items():
        cleaned_meta = {k.strip(): v.strip() for k, v in meta.items()}
        cleaned_meta['sample_id'] = sample_id.strip()
        metadata_list.append(cleaned_meta)
    
    sample_metadata_df = pd.DataFrame(metadata_list)
    
    # Yaygın metadata sütunlarını temizle
    for col in ['Sample_geo_accession', 'Sample_characteristics_ch1', 'Sample_title', 'Sample_source_name_ch1']:
        if col in sample_metadata_df.columns:
            sample_metadata_df[col] = sample_metadata_df[col].astype(str).str.replace('\n', '', regex=False).str.strip()
    
    return platform_df, merged_expression_df, sample_metadata_df

def main():
    file_path = 'data/GSE53700_family.soft' # Dosya yolunuzu güncelleyin

    platform_df, merged_expression_df, sample_metadata_df = process_soft_file(file_path)

    if platform_df is not None:
        print('\nPlatform tablosu (ilk 5 satır):')
        print(platform_df.head())
        print('\nToplam probe (Platformda):', platform_df.shape[0])

    if merged_expression_df is not None:
        print('\nTüm sample ekspresyon matrisi (ilk 5 satır):')
        print(merged_expression_df.head())
        print('\nBirleşik matris boyutu:', merged_expression_df.shape)

        output_expression_file = 'processed_gene_expression_matrix.csv'
        merged_expression_df.to_csv(output_expression_file, index=True)
        print(f"Gen ekspresyon matrisi '{output_expression_file}' olarak kaydedildi.")
    else:
        print('Gen ekspresyon matrisi oluşturulamadı.')

    if sample_metadata_df is not None and not sample_metadata_df.empty:
        print('\nSample metadata (ilk 5 satır):')
        print(sample_metadata_df.head())
        print('\nSample metadata boyutu:', sample_metadata_df.shape)

        output_metadata_file = 'sample_metadata.csv'
        sample_metadata_df.to_csv(output_metadata_file, index=False)
        print(f"Sample metadata '{output_metadata_file}' olarak kaydedildi.")
    else:
        print('Sample metadata oluşturulamadı veya boş.')

if __name__ == '__main__':
    main()