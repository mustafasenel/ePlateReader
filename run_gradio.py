#!/usr/bin/env python3
"""Run Gradio UI for ePlateReader."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eplatereader.ui.app import create_gradio_app


def main():
    """Run Gradio interface."""
    print("""
╔══════════════════════════════════════════════════════════╗
║          ePlateReader - Gradio UI                        ║
║          Türk Plaka Tanıma Sistemi                       ║
╚══════════════════════════════════════════════════════════╝

⚠️  ÖNEMLİ: LLM servisinin çalışıyor olması gerekir!
   Eğer servis çalışmıyorsa, başka bir terminalde:
   
   python run_service.py

Gradio UI başlatılıyor...
""")
    
    # Create and launch app
    app = create_gradio_app(service_url="http://localhost:8000")
    
    print("""
✅ Gradio UI hazır!

🌐 Tarayıcınızda açın: http://localhost:7860

Özellikler:
  - 🚗 Plaka Tanıma (Tespit → Kırpma → Deskew → OCR)
  - 💬 Genel LLM Sorgusu (Görsel + Prompt → Yanıt)

Press CTRL+C to stop
""")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
