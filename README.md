# ePlateReader - Türk Plaka Tanıma Sistemi

**Qwen3-VL** tabanlı akıllı plaka tanıma ve görsel analiz sistemi.

## 🎨 Kullanım Yöntemleri

1. **🖥️ Gradio UI** - Web arayüzü (Önerilen)
2. **💻 CLI** - Komut satırı
3. **🔌 API** - REST API

## 🚀 Hızlı Başlangıç

### Yöntem 1: Gradio UI (Önerilen)

```bash
# 1. Servisi başlat
python run_service.py

# 2. Gradio UI'yi başlat (başka terminalde)
python run_gradio.py

# 3. Tarayıcıda aç: http://localhost:7860
```

**Özellikler:**
- 🚗 Plaka tanıma (görsel → tespit → kırpma → deskew → OCR)
- 💬 Genel LLM sorgusu (görsel + prompt → yanıt)
- 📊 Görsel sonuç gösterimi
- 🎯 Kullanıcı dostu arayüz

### Yöntem 2: CLI

### 1. Servisi Başlat (Bir Kez)

LLM modelini yüklemek ve servisi başlatmak için:

```bash
# Terminal 1
python run_service.py
```

Servis başladığında göreceksiniz:
```
╔══════════════════════════════════════════════════════════╗
║          LLM Vision Service                              ║
║          Qwen3-VL License Plate Recognition              ║
╚══════════════════════════════════════════════════════════╝

Starting service on 0.0.0.0:8000
Model: Qwen/Qwen3-VL-2B-Instruct
Device: mps

API Documentation: http://localhost:8000/docs
```

**Önemli:** Servis bir kez başlatıldıktan sonra arka planda çalışmaya devam eder. Model bellekte kalır ve her seferinde yeniden yüklenmez!

### 2. Plaka Tanıma (CLI)

Servis çalışırken, başka bir terminalde:

```bash
# Terminal 2
python main.py testImages/1.png --debug
```

**Avantajlar:**
- ✅ Model sadece bir kez yüklenir (serviste)
- ✅ Her CLI çağrısı hızlıdır (model yeniden yüklenmez)
- ✅ Bellek tasarrufu
- ✅ Aynı model birden fazla istemci tarafından kullanılabilir

## 📋 Kullanım Senaryoları

### Senaryo 1: Tek Plaka Tanıma

```bash
# Servis çalışıyor olmalı
python main.py testImages/1.png
```

### Senaryo 2: Birden Fazla Plaka Tanıma

```bash
# Servis bir kez başlatılır
python run_service.py &

# Birden fazla görüntü işlenir (model yeniden yüklenmez!)
python main.py testImages/1.png
python main.py testImages/2.png
python main.py testImages/3.png
```

### Senaryo 3: API ile Kullanım (Harici Projeler)

```python
import requests
import base64

# Görüntüyü base64'e çevir
with open("plate.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# API'ye gönder
response = requests.post(
    "http://localhost:8000/api/v1/recognize/plate",
    json={"image_base64": image_base64}
)

result = response.json()
print(f"Plaka: {result['plate_text']}")
print(f"Güven: {result['confidence']:.2%}")
```

### Senaryo 4: Genel Amaçlı Görüntü Analizi

```python
import requests
import base64

with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "image_base64": image_base64,
        "prompt": "Bu görüntüde ne görüyorsun?",
        "max_tokens": 200
    }
)

result = response.json()
print(result['response'])
```

## 🔧 Mimari

```
┌─────────────────────────────────────────────────────┐
│                  LLM Service                        │
│              (python run_service.py)                │
│                                                     │
│  ┌──────────────────────────────────────────┐     │
│  │   Qwen3-VL Model (Bellekte - Tek Kopya) │     │
│  └──────────────────────────────────────────┘     │
│                      ▲                             │
│                      │ HTTP API                    │
└──────────────────────┼─────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
   │  CLI    │    │  API    │   │  Web    │
   │ main.py │    │ Client  │   │  App    │
   └─────────┘    └─────────┘   └─────────┘
```

## 📊 Performans

### İlk Başlatma (Servis)
- Model yükleme: ~5-10 saniye
- Bellek kullanımı: ~4-6 GB

### Sonraki İstekler (CLI/API)
- İstek süresi: ~2-3 saniye
- Bellek kullanımı: Minimal (sadece istek/cevap)
- Model yeniden yüklenmez ✅

## 🛠️ Konfigürasyon

Environment variables ile özelleştirme:

```bash
# Farklı port kullan
export LLM_SERVICE_API_PORT=8080
python run_service.py

# CLI'dan farklı URL kullan
python main.py testImages/1.png --service-url http://localhost:8080
```

## ❓ Sık Sorulan Sorular

### Servis çalışmıyor hatası alıyorum?

```bash
⚠️  Warning: LLM Service not available at http://localhost:8000
   Please start the service with: python run_service.py
```

**Çözüm:** Önce servisi başlatın:
```bash
python run_service.py
```

### Model her seferinde yeniden yükleniyor mu?

**Hayır!** Model sadece servis başlatıldığında bir kez yüklenir. Sonraki tüm istekler aynı model instance'ını kullanır.

### Servisi arka planda nasıl çalıştırırım?

```bash
# macOS/Linux
nohup python run_service.py > service.log 2>&1 &

# Veya screen kullan
screen -S llm-service
python run_service.py
# CTRL+A, D ile detach
```

### Servisi nasıl durdururum?

```bash
# Process ID'yi bul
ps aux | grep run_service.py

# Durdur
kill <PID>

# Veya
pkill -f run_service.py
```

## 🧪 Test

```bash
# Gradio UI testi
python run_gradio.py
# Tarayıcıda: http://localhost:7860

# Servis testi
python test_service.py testImages/1.png

# CLI testi
python main.py testImages/1.png --debug
```

## 📚 Daha Fazla Bilgi

- **Gradio UI Kılavuzu:** [GRADIO_GUIDE.md](GRADIO_GUIDE.md)
- **API Dokümantasyonu:** http://localhost:8000/docs
- **Kullanım Detayları:** [USAGE.md](USAGE.md)

## 🎯 Özet

### Gradio UI ile:
1. **Servisi başlat:** `python run_service.py`
2. **UI'yi başlat:** `python run_gradio.py`
3. **Tarayıcıda kullan:** http://localhost:7860

### CLI ile:
1. **Servisi başlat:** `python run_service.py`
2. **Plaka tanı:** `python main.py <image>`
3. **Model bellekte kalır** - Her seferinde yeniden yüklenmez!

### API ile:
```python
import requests, base64
with open("plate.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode('utf-8')
response = requests.post("http://localhost:8000/api/v1/recognize/plate", 
                        json={"image_base64": img_b64})
print(response.json()["plate_text"])
```
