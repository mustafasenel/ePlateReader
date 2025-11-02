#!/usr/bin/env python3
"""Basit test scripti - Görsel ve prompt ile LLM'e sorgu gönder."""

import requests
import base64
from pathlib import Path
from PIL import Image
import io


# ==================== BURAYA GİR ====================
IMAGE_PATH = "/Users/senel/Downloads/1.jpeg"  # Görsel yolu
PROMPT = """
You are an expert in OCR post-processing and invoice normalization.

You will receive the raw OCR text extracted from an invoice.  
Invoices may come from different companies, and the table column order, names, or formats may vary.  
Your task is to extract and normalize all useful data into a structured JSON with three main parts:

1. invoice_meta: General information about the invoice:
   - Firma: the company name (seller/supplier) at the top of the invoice
   - Rechnungsnummer (Invoice number)
   - Rechnungsdatum (Invoice date)

2. invoice_data: Line items of the invoice table.  
   Each row should have:
   - ArtikelNumber: product code (usually numeric/alphanumeric, first or second column)
   - ArtikelBez: product description (free text, product name)
   - Kolli: number of packages (integer)
   - Inhalt: number of items per package (integer)
   - Menge: total quantity (Kolli × Inhalt)
   - Preis: price per unit (float)
   - Netto: total line amount (Menge × Preis)
   - MwSt: VAT rate for this line item (typically 7 or 19, as integer) - ONLY include if explicitly present in the invoice table. Do not add this field if the invoice does not have a VAT column.

3. invoice_summary: Extract financial totals from the invoice footer. This section may not be present on all pages, especially on first pages of multi-page invoices. It will typically be found on the LAST page of the invoice. If NO financial totals are found at all, return null for this entire section.
   
   When financial totals ARE found, you MUST extract and calculate these fields:
   
   REQUIRED FIELDS (must always be present or calculated):
   - total_vat: Total VAT amount (Gesamte MwSt / Gesamt-Steuer) - REQUIRED
   - total_net: Total net amount before VAT (Gesamtbetrag netto / Zwischensumme) - REQUIRED
   - total_gross: Final total gross amount including VAT (Gesamtbetrag brutto / Endbetrag) - REQUIRED
   
   OPTIONAL FIELDS (only include if explicitly present):
   - vat_7: 7% VAT amount (7% MwSt) - OPTIONAL, only if this rate is used
   - vat_19: 19% VAT amount (19% MwSt) - OPTIONAL, only if this rate is used
   
   CRITICAL CALCULATION RULES:
   - If you see total_net and total_gross, calculate: total_vat = total_gross - total_net
   - If you see vat_7 and vat_19, calculate: total_vat = vat_7 + vat_19
   - If you see total_vat and total_net, calculate: total_gross = total_net + total_vat
   - Always verify the equation: total_net + total_vat = total_gross
   - If explicit VAT rate lines are shown (e.g., "7% MwSt: 160,99" or "19% MwSt: 450,00"), include vat_7 and/or vat_19
   - If no separate VAT rates are shown, DO NOT include vat_7 or vat_19 in the output
   
   CALCULATION PRIORITY:
   1. First, look for explicit total amounts in the invoice footer
   2. If total_net and total_gross are found, calculate total_vat
   3. If separate VAT rates (7%, 19%) are explicitly shown, extract them as vat_7 and vat_19
   4. Always perform mathematical verification to ensure accuracy

### Important Rules & Data Validation:
- Your primary task is not just to extract, but to ensure the final JSON is logically correct.
- Common Sense Price & Number Validation: You are processing invoices for retail/grocery goods. A single unit price (Preis) or quantity will be a reasonable number, almost never in the thousands or millions. If you encounter an ambiguous number like 1,234, it is overwhelmingly likely to be 1.234 (one and a bit), NOT one thousand two hundred thirty-four. Use this context to correctly interpret decimal separators (',' or '.') based on the most logical value for the item.
- Handling OCR Zero-Padding Errors: OCR can produce numbers with excessive trailing zeros after a decimal separator, like 2,3900000 or 15,50000. You must correctly interpret these as 2.39 and 15.5 respectively. Do not interpret the trailing zeros as part of a larger number.
- CRITICAL VALIDATION: For every line item, you MUST perform these calculations:
  1. Calculate Menge: Menge must be the result of Kolli * Inhalt. If the OCR text shows a different Menge, ignore it and use your calculated value.
  2. Calculate Netto: Netto must be the result of your calculated Menge * Preis. If the OCR text shows a different Netto, ignore it and use your calculated value.
- Trust your calculations over the raw OCR text for Menge and Netto to correct potential OCR errors.
- Column headers may vary across companies, always map to the target fields above.
- Normalize numeric formats: use a dot . as decimal separator, remove currency signs. All currency values (Preis, Netto, totals) must be numbers with up to 3 decimal places.
- Normalize date formats: The invoice date Rechnungsdatum must always be converted to dd.MM.yyyy format (e.g., 24.10.2025).
- Output must always be valid JSON with exactly this structure:
  {
    "invoice_meta": { ... },
    "invoice_data": [ ... ],
    "invoice_summary": { 
      "vat_7": number (optional),
      "vat_19": number (optional),
      "total_vat": number (required),
      "total_net": number (required),
      "total_gross": number (required)
    } or null
  }
    
###CRITICAL INSTRUCTIONS FOR JSON FORMATTING:
- Your entire response must be ONLY the raw JSON object. Do not include any text, explanations, or markdown like json.
- The JSON must be perfectly valid. Pay close attention to syntax.
- CRITICAL: Do not use trailing commas. The last element in any array or object must NOT be followed by a comma. This is a common mistake you must avoid.
- Ensure all strings are enclosed in double quotes.

Your response must start with { and end with }.
"""  # Sorun
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
                "max_tokens": 2048
            },
            timeout=600 # 2 dakika timeout
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
