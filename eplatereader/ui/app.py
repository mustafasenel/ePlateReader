"""Gradio UI for ePlateReader."""

import gradio as gr
import cv2
import numpy as np
import requests
import base64
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eplatereader.core.detector import PlateDetector
from eplatereader.core.preprocessor import PlatePreprocessor


class PlateReaderUI:
    """Gradio UI for license plate recognition."""
    
    def __init__(self, service_url: str = "http://localhost:8000"):
        """Initialize UI.
        
        Args:
            service_url: URL of the LLM service
        """
        self.service_url = service_url
        self.detector = None
        self.preprocessor = None
        
    def _init_components(self):
        """Lazy initialization of components."""
        if self.detector is None:
            print("Loading YOLO detector...")
            self.detector = PlateDetector()
            self.preprocessor = PlatePreprocessor(debug=False)
            print("Components loaded!")
    
    def _check_service(self) -> bool:
        """Check if LLM service is running."""
        try:
            response = requests.get(f"{self.service_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def recognize_plate(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, str]:
        """Process image and recognize plate.
        
        Args:
            image: Input image (RGB from Gradio)
            
        Returns:
            Tuple of (cropped_plate, deskewed_plate, plate_text, status_message)
        """
        if image is None:
            return None, None, "", "❌ Lütfen bir görsel yükleyin!"
        
        # Check service
        if not self._check_service():
            return None, None, "", f"❌ LLM Servisi çalışmıyor!\n\nLütfen servisi başlatın:\npython run_service.py"
        
        try:
            # Initialize components
            self._init_components()
            
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Step 1: Detect plates
            detections = self.detector.detect(image_bgr, use_fallback=True)
            
            if not detections:
                return None, None, "", "❌ Plaka tespit edilemedi!"
            
            # Get first detection
            detection = detections[0]
            x1, y1, x2, y2 = detection['bbox']
            plate_img = image_bgr[y1:y2, x1:x2]
            
            # Convert to RGB for display
            plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            
            # Step 2: Deskew
            variants = self.preprocessor.preprocess_multi(plate_img, debug_dir=None, just_deskew=True)
            
            if not variants:
                return plate_img_rgb, None, "", "❌ Ön işleme başarısız!"
            
            deskewed_img = variants[0][1]  # Get deskewed image
            deskewed_img_rgb = cv2.cvtColor(deskewed_img, cv2.COLOR_BGR2RGB)
            
            # Step 3: Recognize with LLM
            # Encode to base64
            success, buffer = cv2.imencode('.jpg', deskewed_img)
            if not success:
                return plate_img_rgb, deskewed_img_rgb, "", "❌ Görsel kodlama hatası!"
            
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to LLM service
            response = requests.post(
                f"{self.service_url}/api/v1/recognize/plate",
                json={"image_base64": image_base64},
                timeout=30
            )
            
            if response.status_code != 200:
                return plate_img_rgb, deskewed_img_rgb, "", f"❌ API Hatası: {response.status_code}"
            
            result = response.json()
            
            if result.get("success"):
                plate_text = result.get("plate_text", "")
                confidence = result.get("confidence", 0.0)
                processing_time = result.get("processing_time", 0.0)
                
                status = f"""✅ Başarılı!

🚗 Plaka: {plate_text}
📊 Güven: {confidence:.2%}
⏱️  Süre: {processing_time:.2f}s
🔍 Tespit: YOLO + Qwen3-VL
"""
                return plate_img_rgb, deskewed_img_rgb, plate_text, status
            else:
                error = result.get("error", "Bilinmeyen hata")
                return plate_img_rgb, deskewed_img_rgb, "", f"❌ Tanıma başarısız: {error}"
                
        except Exception as e:
            return None, None, "", f"❌ Hata: {str(e)}"
    
    def query_llm(self, image: np.ndarray, prompt: str, max_tokens: int) -> Tuple[str, str]:
        """Query LLM with image and prompt.
        
        Args:
            image: Input image (RGB from Gradio)
            prompt: User prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response, status_message)
        """
        if image is None:
            return "", "❌ Lütfen bir görsel yükleyin!"
        
        if not prompt or prompt.strip() == "":
            return "", "❌ Lütfen bir prompt girin!"
        
        # Check service
        if not self._check_service():
            return "", f"❌ LLM Servisi çalışmıyor!\n\nLütfen servisi başlatın:\npython run_service.py"
        
        try:
            # Convert RGB to BGR then to JPEG
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            success, buffer = cv2.imencode('.jpg', image_bgr)
            if not success:
                return "", "❌ Görsel kodlama hatası!"
            
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send to LLM service
            response = requests.post(
                f"{self.service_url}/api/v1/query",
                json={
                    "image_base64": image_base64,
                    "prompt": prompt,
                    "max_tokens": max_tokens
                },
                timeout=120
            )
            
            if response.status_code != 200:
                return "", f"❌ API Hatası: {response.status_code}"
            
            result = response.json()
            
            if result.get("success"):
                llm_response = result.get("response", "")
                processing_time = result.get("processing_time", 0.0)
                
                status = f"✅ Başarılı! (⏱️ {processing_time:.2f}s)"
                return llm_response, status
            else:
                error = result.get("error", "Bilinmeyen hata")
                return "", f"❌ Sorgu başarısız: {error}"
                
        except Exception as e:
            return "", f"❌ Hata: {str(e)}"


def create_gradio_app(service_url: str = "http://localhost:8000") -> gr.Blocks:
    """Create Gradio interface.
    
    Args:
        service_url: URL of the LLM service
        
    Returns:
        Gradio Blocks app
    """
    ui = PlateReaderUI(service_url=service_url)
    
    with gr.Blocks(title="ePlateReader - Plaka Tanıma Sistemi", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🚗 ePlateReader - Türk Plaka Tanıma Sistemi
        
        **Qwen3-VL** tabanlı akıllı plaka tanıma ve görsel analiz sistemi.
        
        ⚠️ **Not:** LLM servisinin çalışıyor olması gerekir: `python run_service.py`
        """)
        
        with gr.Tabs():
            # Tab 1: Plate Recognition
            with gr.Tab("🚗 Plaka Tanıma"):
                gr.Markdown("""
                ### Araç Plakası Tanıma
                
                Araç görselini yükleyin, sistem otomatik olarak:
                1. Plakayı tespit eder (YOLO)
                2. Plakayı kırpar ve düzeltir (Deskew)
                3. LLM ile okur (Qwen3-VL)
                """)
                
                with gr.Row():
                    with gr.Column():
                        plate_input = gr.Image(label="Araç Görseli", type="numpy")
                        plate_button = gr.Button("🔍 Plaka Tanı", variant="primary", size="lg")
                    
                    with gr.Column():
                        plate_cropped = gr.Image(label="Kırpılmış Plaka", type="numpy")
                        plate_deskewed = gr.Image(label="Düzeltilmiş Plaka", type="numpy")
                
                with gr.Row():
                    with gr.Column():
                        plate_result = gr.Textbox(
                            label="Tanınan Plaka",
                            placeholder="Plaka numarası burada görünecek...",
                            lines=1,
                            max_lines=1,
                            scale=2
                        )
                    with gr.Column():
                        plate_status = gr.Textbox(
                            label="Durum",
                            placeholder="İşlem durumu...",
                            lines=6,
                            scale=1
                        )
                
                plate_button.click(
                    fn=ui.recognize_plate,
                    inputs=[plate_input],
                    outputs=[plate_cropped, plate_deskewed, plate_result, plate_status]
                )
                
                gr.Examples(
                    examples=[
                        ["testImages/1.png"],
                        ["testImages/2.png"],
                        ["testImages/3.png"],
                    ],
                    inputs=plate_input,
                    label="Örnek Görseller"
                )
            
            # Tab 2: General LLM Query
            with gr.Tab("💬 Genel LLM Sorgusu"):
                gr.Markdown("""
                ### Görsel Analiz ve Soru-Cevap
                
                Herhangi bir görsel yükleyin ve LLM'e soru sorun.
                """)
                
                with gr.Row():
                    with gr.Column():
                        llm_input = gr.Image(label="Görsel", type="numpy")
                        llm_prompt = gr.Textbox(
                            label="Prompt (Soru/Talimat)",
                            placeholder="Örn: Bu görüntüde ne görüyorsun?",
                            lines=3
                        )
                        llm_max_tokens = gr.Slider(
                            minimum=50,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Maksimum Token"
                        )
                        llm_button = gr.Button("🤖 Gönder", variant="primary", size="lg")
                    
                    with gr.Column():
                        llm_response = gr.Textbox(
                            label="LLM Yanıtı",
                            placeholder="Yanıt burada görünecek...",
                            lines=15
                        )
                        llm_status = gr.Textbox(
                            label="Durum",
                            placeholder="İşlem durumu...",
                            lines=2
                        )
                
                llm_button.click(
                    fn=ui.query_llm,
                    inputs=[llm_input, llm_prompt, llm_max_tokens],
                    outputs=[llm_response, llm_status]
                )
                
                gr.Examples(
                    examples=[
                        ["testImages/1.png", "Bu araç hangi marka ve model?", 100],
                        ["testImages/2.png", "Bu görüntüde kaç araç var?", 50],
                        ["testImages/3.png", "Bu görüntüyü detaylı açıkla.", 200],
                    ],
                    inputs=[llm_input, llm_prompt, llm_max_tokens],
                    label="Örnek Sorgular"
                )
        
        gr.Markdown("""
        ---
        ### 📚 Kullanım Bilgileri
        
        - **Plaka Tanıma:** Araç görselini yükleyin, sistem otomatik olarak plakayı bulur ve okur.
        - **Genel Sorgu:** Herhangi bir görsel ve prompt ile LLM'e soru sorabilirsiniz.
        - **API Kullanımı:** Gradio olmadan da API endpoint'lerini kullanabilirsiniz.
        
        **API Dokümantasyonu:** [http://localhost:8000/docs](http://localhost:8000/docs)
        """)
    
    return app


if __name__ == "__main__":
    # Test için direkt çalıştırma
    app = create_gradio_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
