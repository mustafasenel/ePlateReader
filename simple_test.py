#!/usr/bin/env python3
"""Basit test scripti - Görsel ve prompt ile LLM'e sorgu gönder."""

import requests
import base64
from pathlib import Path
from PIL import Image
import io


# ==================== BURAYA GİR ====================
IMAGE_PATH = "/Users/senel/Downloads/ehliyet.jpeg"  # Görsel yolu
PROMPT = "Bu sürücü belgesinin 4a satırında yer alan veriliş tarihi nedir?"  # Sorun
SERVICE_URL = "http://localhost:8000"  # Servis URL'i
# ====================================================


def resize_image(image_path: str, max_size: int = 1024) -> str:
    """Görseli aspect ratio koruyarak küçült ve base64'e çevir.
    
    Args:
        image_path: Görsel dosya yolu
        max_size: Maksimum genişlik veya yükseklik (piksel)
    
    Returns:
        Base64 encoded resized image
    """
    # Görseli aç
    img = Image.open(image_path)
    
    # Orijinal boyutları al
    original_width, original_height = img.size
    print(f"   Orijinal boyut: {original_width}x{original_height}")
    
    # Aspect ratio'yu koruyarak yeni boyutları hesapla
    if original_width > max_size or original_height > max_size:
        if original_width > original_height:
            new_width = max_size
            new_height = int(original_height * (max_size / original_width))
        else:
            new_height = max_size
            new_width = int(original_width * (max_size / original_height))
        
        # Görseli yeniden boyutlandır (LANCZOS en kaliteli)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        print(f"   Yeni boyut: {new_width}x{new_height}")
    else:
        print(f"   Görsel zaten küçük, yeniden boyutlandırma yapılmadı")
    
    # RGB'ye çevir (RGBA ise)
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # JPEG olarak buffer'a kaydet
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85, optimize=True)
    buffer.seek(0)
    
    # Base64'e çevir
    return base64.b64encode(buffer.read()).decode('utf-8')


def query_llm(image_path: str, prompt: str, service_url: str = "http://localhost:8000"):
    """LLM'e görsel ve prompt gönder."""
    
    # Görsel kontrolü
    if not Path(image_path).exists():
        print(f"❌ Hata: Görsel bulunamadı: {image_path}")
        return None
    
    print(f"📸 Görsel: {image_path}")
    print(f"💬 Prompt: {prompt}")
    print(f"🌐 Servis: {service_url}")
    print("-" * 60)
    
    try:
        # Görseli küçült ve base64'e çevir
        print("🔄 Görsel yeniden boyutlandırılıyor ve base64'e çevriliyor...")
        image_base64 = resize_image(image_path, max_size=1024)
        
        # API'ye gönder
        print("📤 LLM servisine gönderiliyor...")
        response = requests.post(
            f"{service_url}/api/v1/query",
            json={
                "image_base64": image_base64,
                "prompt": prompt,
                "max_tokens": 200
            },
            timeout=120  # 2 dakika timeout
        )
        
        # Sonuç
        if response.status_code != 200:
            print(f"❌ Hata: HTTP {response.status_code}")
            print(response.text)
            return None
        
        result = response.json()
        
        if result.get("success"):
            print("\n" + "=" * 60)
            print("✅ SONUÇ:")
            print("=" * 60)
            print(result['response'])
            print("=" * 60)
            print(f"⏱️  İşlem süresi: {result['processing_time']:.2f} saniye")
            return result['response']
        else:
            print(f"❌ Hata: {result.get('error')}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Hata: Servis çalışmıyor!")
        print("   Lütfen önce servisi başlatın: python run_service.py")
        return None
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🤖 LLM Vision Test - Basit Sorgu")
    print("=" * 60 + "\n")
    
    # Sorgu gönder
    result = query_llm(IMAGE_PATH, PROMPT, SERVICE_URL)
    
    if result:
        print("\n✅ Test başarılı!")
    else:
        print("\n❌ Test başarısız!")
