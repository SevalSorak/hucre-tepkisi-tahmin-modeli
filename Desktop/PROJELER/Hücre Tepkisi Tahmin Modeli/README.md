# Yapay Zeka Destekli Hücre Tepkisi Tahmin Modeli: Silika Nanopartikül Pertürbasyonu

## Proje Amacı

Bu proje, `GSE53700` veri setini kullanarak A549 insan akciğer hücrelerinin silika nanopartikül (SM30 ve AS30 NPs) maruziyetine verdiği gen ekspresyon tepkilerini analiz etmeyi ve bu tepkileri tahmin eden bir makine öğrenimi modeli geliştirmeyi amaçlar. Model, farklı nanopartikül tipleri ve maruziyet sürelerine göre hücresel yanıtları sınıflandırır ve biyolojik olarak anlamlı genleri belirler.

## Klasör ve Dosya Yapısı

```
.
├── data/
│   ├── GSE53700_family.soft                # Orijinal SOFT formatında ham veri
│   ├── processed_gene_expression_matrix.csv # İşlenmiş gen ekspresyon matrisi
│   └── sample_metadata.csv                 # Örnek metadata
├── notebooks/
│   └── 01_model_training.ipynb             # Model eğitimi ve analiz adımlarını içeren Jupyter Notebook
├── src/
│   ├── modeling.py                         # Model eğitimi ve değerlendirme fonksiyonları
│   ├── preprocessing.py                    # Veri yükleme, ön işleme ve etiket çıkarımı
│   └── read_soft_file.py                   # SOFT dosyasından veri çıkarımı ve işlenmesi
```

## Veri Akışı ve Ana Bileşenler

1. **Ham Veri (data/GSE53700_family.soft):**
   - GEO'dan indirilen, gen ekspresyonu ve örnek bilgilerini içeren SOFT formatında dosya.

2. **Veri İşleme (src/read_soft_file.py):**
   - SOFT dosyasından gen ekspresyon matrisi ve örnek metadata çıkarılır.
   - Çıktılar: `processed_gene_expression_matrix.csv` ve `sample_metadata.csv`

3. **Ön İşleme (src/preprocessing.py):**
   - Gen ekspresyon matrisi ve metadata yüklenir.
   - Pertürbasyon etiketleri (NP tipi ve süre) çıkarılır ve sayısal olarak kodlanır.
   - Gen ekspresyon verisi log2 dönüşümü ve standart ölçeklendirme ile normalize edilir.

4. **Model Eğitimi ve Değerlendirme (src/modeling.py, notebooks/01_model_training.ipynb):**
   - Random Forest ve Logistic Regression sınıflandırıcıları ile model eğitimi yapılır.
   - Modelin doğruluk, ROC AUC, sınıflandırma raporu ve özellik önem düzeyleri hesaplanır.
   - Sonuçlar görselleştirilir ve biyolojik olarak yorumlanır.

## Kullanım

### 1. Ortam Kurulumu

Gerekli Python paketleri:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- shap (isteğe bağlı, model açıklanabilirliği için)
- jupyter

Kurulum için:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap jupyter
```

### 2. Veri Hazırlama

Ham SOFT dosyasından işlenmiş veri üretmek için:
```bash
python src/read_soft_file.py
```
Bu adım, `data/` klasöründe gerekli CSV dosyalarını oluşturur.

### 3. Model Eğitimi ve Analiz

Jupyter Notebook ile adım adım analiz ve model eğitimi için:
```bash
jupyter notebook notebooks/01_model_training.ipynb
```
veya doğrudan Python dosyası ile:
```bash
python src/modeling.py
```

### 4. Sonuçların Yorumlanması

- Model, farklı nanopartikül tipleri ve maruziyet sürelerini gen ekspresyon profillerinden yüksek doğrulukla ayırt edebilir.
- Özellik önem düzeyleri, biyolojik olarak anlamlı genleri ortaya çıkarır.
- Sonuçlar, sanal hücre modellemeleriyle entegre edilebilir ve yeni biyolojik hipotezler için temel oluşturabilir.

## Notlar ve Geliştirme Önerileri

- Veri seti küçük olduğu için modelin genelleme kapasitesi sınırlı olabilir. Daha büyük veri setleriyle test edilmesi önerilir.
- Farklı dozlar ve ek biyolojik koşullar için modelin genişletilmesi mümkündür.
- Derin öğrenme ve biyolojik yol analizi gibi ileri yöntemlerle entegrasyon potansiyeli vardır.

## Kaynaklar

- [GSE53700 - GEO DataSet](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE53700)
- Proje kapsamında kullanılan makale: *Altered Gene Transcription in Human Cells Treated with Ludox® Silica Nanoparticles* 