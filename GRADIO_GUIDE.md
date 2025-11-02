# Gradio UI Kullanım Kılavuzu

## 🎨 Gradio Arayüzü

ePlateReader için kullanıcı dostu web arayüzü.

## 🚀 Kurulum

```bash
# Gradio ve bağımlılıkları yükle
pip install gradio Pillow
```

## 📖 Kullanım

### 1. LLM Servisini Başlat (Zorunlu)

Gradio UI, LLM servisine bağlı çalışır. Önce servisi başlatın:

```bash
# Terminal 1
python run_service.py
```

### 2. Gradio UI'yi Başlat

```bash
# Terminal 2
python run_gradio.py
```

Tarayıcınızda otomatik açılacak: **http://localhost:7860**

## 🎯 Özellikler

### 🚗 Plaka Tanıma Sekmesi

**İşlem Akışı:**
1. Araç görselini yükleyin
2. "🔍 Plaka Tanı" butonuna tıklayın
3. Sistem otomatik olarak:
   - Plakayı tespit eder (YOLO)
   - Plakayı kırpar
   - Plakayı düzeltir (deskew)
   - LLM ile okur (Qwen3-VL)

**Çıktılar:**
- Kırpılmış plaka görseli
- Düzeltilmiş (deskewed) plaka görseli
- Tanınan plaka numarası
- Güven skoru ve işlem süresi

**Örnek:**
```
Input: Araç görseli
Output: 
  - Kırpılmış: [plaka_cropped.jpg]
  - Düzeltilmiş: [plaka_deskewed.jpg]
  - Plaka: 34ABC123
  - Güven: 95%
  - Süre: 2.5s
```

### 💬 Genel LLM Sorgusu Sekmesi

**İşlem Akışı:**
1. Herhangi bir görsel yükleyin
2. Prompt (soru/talimat) girin
3. Maksimum token sayısını ayarlayın (50-500)
4. "🤖 Gönder" butonuna tıklayın

**Örnek Kullanımlar:**

#### Örnek 1: Görsel Açıklama
```
Görsel: araç.jpg
Prompt: Bu görüntüde ne görüyorsun? Detaylı açıkla.
Max Tokens: 200
```

#### Örnek 2: Belge Okuma
```
Görsel: ehliyet.jpg
Prompt: Bu sürücü belgesinin 4a satırındaki veriliş tarihini oku.
Max Tokens: 50
```

#### Örnek 3: Nesne Sayma
```
Görsel: otopark.jpg
Prompt: Bu otoparkta kaç araç var ve ne renkteler?
Max Tokens: 100
```

## 🔧 Mimari

```
┌─────────────────────────────────────────────────────┐
│                  Gradio UI                          │
│              (http://localhost:7860)                │
│                                                     │
│  ┌─────────────────┐    ┌─────────────────┐       │
│  │  Plaka Tanıma   │    │  Genel Sorgu    │       │
│  │  (Tab 1)        │    │  (Tab 2)        │       │
│  └────────┬────────┘    └────────┬────────┘       │
│           │                      │                 │
└───────────┼──────────────────────┼─────────────────┘
            │                      │
            ▼                      ▼
    ┌───────────────────────────────────────┐
    │         LLM Service (API)             │
    │     (http://localhost:8000)           │
    │                                       │
    │  - /api/v1/recognize/plate           │
    │  - /api/v1/query                     │
    └───────────────────────────────────────┘
```

## 🎨 Kullanım Senaryoları

### Senaryo 1: Plaka Tanıma Testi

```bash
# 1. Servisi başlat
python run_service.py

# 2. Gradio'yu başlat
python run_gradio.py

# 3. Tarayıcıda:
#    - "Plaka Tanıma" sekmesine git
#    - Araç görselini yükle
#    - "Plaka Tanı" butonuna tıkla
#    - Sonuçları gör
```

### Senaryo 2: Belge Okuma

```bash
# 1. Servisi başlat (eğer çalışmıyorsa)
python run_service.py

# 2. Gradio'yu başlat
python run_gradio.py

# 3. Tarayıcıda:
#    - "Genel LLM Sorgusu" sekmesine git
#    - Belge görselini yükle
#    - Prompt gir: "Bu belgeden X bilgisini çıkar"
#    - "Gönder" butonuna tıkla
```

### Senaryo 3: API Kullanımı (Gradio Olmadan)

Gradio'yu başlatmadan sadece API kullanmak için:

```python
import requests
import base64

# Görsel yükle
with open("plate.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# Plaka tanı
response = requests.post(
    "http://localhost:8000/api/v1/recognize/plate",
    json={"image_base64": image_base64}
)

print(response.json())
```

## 🔒 Port Ayarları

### Varsayılan Portlar:
- **LLM Service:** 8000
- **Gradio UI:** 7860

### Farklı Port Kullanma:

```bash
# LLM Service için farklı port
export LLM_SERVICE_API_PORT=8080
python run_service.py

# Gradio için farklı port (kod içinde değiştir)
# run_gradio.py dosyasında server_port=7860 değiştir
```

## 🐛 Sorun Giderme

### Gradio başlamıyor

```bash
# Gradio yüklü mü kontrol et
pip list | grep gradio

# Yükle
pip install gradio
```

### "LLM Servisi çalışmıyor" hatası

```bash
# Servis çalışıyor mu kontrol et
curl http://localhost:8000/health

# Çalışmıyorsa başlat
python run_service.py
```

### Port zaten kullanımda

```bash
# Portu kullanan process'i bul
lsof -i :7860

# Durdur
kill <PID>
```

## 📊 Performans

- **Plaka Tanıma:** ~3-5 saniye
  - YOLO tespit: ~0.5s
  - Preprocessing: ~0.2s
  - LLM OCR: ~2-3s

- **Genel Sorgu:** ~2-10 saniye
  - Görsel boyutuna bağlı
  - Prompt karmaşıklığına bağlı
  - Max token sayısına bağlı

## 🎯 İpuçları

1. **Görsel Kalitesi:** Yüksek çözünürlüklü görseller daha iyi sonuç verir
2. **Prompt Yazımı:** Net ve spesifik promptlar daha iyi yanıtlar alır
3. **Max Token:** Kısa cevaplar için 50-100, detaylı için 200-500
4. **Batch İşlem:** Birden fazla görsel için API kullanın (daha hızlı)

## 📚 Daha Fazla Bilgi

- **API Dokümantasyonu:** http://localhost:8000/docs
- **Gradio Dokümantasyonu:** https://gradio.app/docs
- **Proje GitHub:** https://github.com/mustafasenel/ePlateReader

## 🤝 Katkıda Bulunma

Gradio UI'yi geliştirmek için:

1. `eplatereader/ui/app.py` dosyasını düzenle
2. Yeni özellikler ekle
3. Test et: `python run_gradio.py`
4. Pull request aç

## 📄 Lisans

MIT License
