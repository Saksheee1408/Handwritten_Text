import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from transformers import pipeline
import io
import base64
from typing import Optional, Tuple
import logging
import os
import platform
import subprocess

# ================================
# TESSERACT PATH FIX - ADD THIS FIRST
# ================================

def setup_tesseract():
    """Automatically detect and set Tesseract path"""
    system = platform.system()
    
    if system == "Windows":
        # Common Windows Tesseract paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\tools\tesseract\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
            r'E:\Program Files\Tesseract-OCR\tesseract.exe',
        ]
        
        # Try to find Tesseract
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return path
        
        # Try using 'where' command
        try:
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout.strip():
                path = result.stdout.strip().split('\n')[0]
                pytesseract.pytesseract.tesseract_cmd = path
                return path
        except:
            pass
            
        return None
    
    return "tesseract"  # For Linux/macOS

# Setup Tesseract path
tesseract_path = setup_tesseract()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Handwritten Text OCR & Summarizer",
    page_icon="📝",
    layout="wide"
)

class HandwrittenTextProcessor:
    def __init__(self):
        """Initialize the text processor with OCR and summarization models."""
        self.summarizer = None
        self.setup_models()
    
    def setup_models(self):
        """Setup OCR and summarization models."""
        try:
            # Use a smaller, faster model for better performance
            st.write("🔄 Loading summarization model...")
            self.summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",  # Smaller, faster model
                device=-1  # Use CPU
            )
            st.success("✅ Summarization model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.error(f"Error loading models: {e}")
            self.summarizer = None
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess the image for better OCR results.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            
            # Handle different image modes
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
                else:  # RGB
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Resize if too large (memory optimization)
            height, width = gray.shape
            if height > 2000 or width > 2000:
                scale = min(2000/height, 2000/width)
                new_height, new_width = int(height*scale), int(width*scale)
                gray = cv2.resize(gray, (new_width, new_height))
            
            # Apply preprocessing techniques
            # 1. Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # 2. Contrast enhancement using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Thresholding for better text recognition
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 4. Morphological operations to clean up
            kernel = np.ones((1,1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            st.error(f"Error preprocessing image: {e}")
            return np.array(image)
    
    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from handwritten image using OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text as string
        """
        try:
            # Check Tesseract availability
            if not tesseract_path:
                return "❌ Tesseract OCR not found!\n\n🛠️ QUICK FIX:\n\n1. Download: https://github.com/UB-Mannheim/tesseract/wiki\n2. Install and restart this app\n3. Or manually set path in code:\n   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'"
            
            # Test Tesseract
            try:
                version = pytesseract.get_tesseract_version()
            except Exception as e:
                return f"❌ Tesseract error: {str(e)}\n\n🔧 Try setting path manually:\npytesseract.pytesseract.tesseract_cmd = r'YOUR_TESSERACT_PATH'"
            
            # Preprocess the image
            processed_img = self.preprocess_image(image)
            
            # Configure Tesseract for handwritten text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[] '
            
            # Extract text
            extracted_text = pytesseract.image_to_string(
                processed_img, 
                config=custom_config,
                lang='eng'
            )
            
            # Clean the text
            cleaned_text = self.clean_extracted_text(extracted_text)
            
            return cleaned_text
            
        except Exception as e:
            error_msg = str(e)
            if "tesseract is not installed" in error_msg.lower() or "not in your path" in error_msg.lower():
                return f"❌ Tesseract PATH issue!\n\n🔧 QUICK FIX - Add this line to your code:\npytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'\n\n📥 Download Tesseract: https://github.com/UB-Mannheim/tesseract/wiki"
            else:
                logger.error(f"Error extracting text: {e}")
                return f"❌ OCR Error: {str(e)}"
    
    def clean_extracted_text(self, text: str) -> str:
        """
        Clean and process the extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace and newlines
        cleaned = ' '.join(text.split())
        
        # Remove very short "words" that are likely OCR errors
        words = cleaned.split()
        filtered_words = [word for word in words if len(word) > 1 or word.isalnum()]
        
        return ' '.join(filtered_words)
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """
        Summarize the extracted text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text
        """
        try:
            if not self.summarizer:
                return "❌ Summarization model not available. Please restart the app to reload models."
            
            # Check if text is long enough to summarize
            word_count = len(text.split())
            if word_count < 15:
                return f"📝 Text too short to summarize effectively ({word_count} words). Original text: {text}"
            
            # Adjust parameters based on text length
            adjusted_max = min(max_length, word_count * 2)
            adjusted_min = min(min_length, max(10, word_count // 4))
            
            # Generate summary
            summary = self.summarizer(
                text,
                max_length=adjusted_max,
                min_length=adjusted_min,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return f"❌ Summarization error: {str(e)[:100]}...\n\nOriginal text: {text[:200]}..."

def show_tesseract_status():
    """Show Tesseract installation status"""
    st.sidebar.subheader("🔧 System Status")
    
    if tesseract_path:
        try:
            version = pytesseract.get_tesseract_version()
            st.sidebar.success(f"✅ Tesseract: v{version}")
            if platform.system() == "Windows":
                st.sidebar.info(f"📂 Path: {tesseract_path}")
        except Exception as e:
            st.sidebar.error(f"❌ Tesseract Error: {str(e)[:50]}...")
            show_manual_fix()
    else:
        st.sidebar.error("❌ Tesseract not found!")
        show_manual_fix()

def show_manual_fix():
    """Show manual Tesseract fix instructions"""
    with st.sidebar.expander("🛠️ Manual Fix"):
        st.code("""
# Add this line at the top of your code:
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
        """)
        st.markdown("**Download:** [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)")

def create_api_endpoint():
    """Create API endpoint for Next.js integration."""
    st.subheader("🔌 API Integration")
    
    st.code("""
# Example API call for Next.js integration
import requests

def call_streamlit_api(image_base64, summarize=True):
    url = "YOUR_STREAMLIT_APP_URL"
    
    payload = {
        "image": image_base64,
        "summarize": summarize
    }
    
    response = requests.post(f"{url}/api/process", json=payload)
    return response.json()

# Usage in Next.js
const handleImageUpload = async (imageFile) => {
    const reader = new FileReader();
    reader.onload = async (e) => {
        const base64 = e.target.result.split(',')[1];
        
        const response = await fetch('/api/ocr-summarize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                image: base64,
                summarize: true 
            })
        });
        
        const result = await response.json();
        console.log('Extracted text:', result.text);
        console.log('Summary:', result.summary);
    };
    reader.readAsDataURL(imageFile);
};
    """, language="javascript")

@st.cache_resource
def get_processor():
    """Get cached processor instance"""
    return HandwrittenTextProcessor()

def main():
    """Main Streamlit application."""
    st.title("📝 Handwritten Text OCR & Summarizer")
    st.markdown("Upload an image with handwritten text to extract and summarize the content.")
    
    # Show system status
    show_tesseract_status()
    
    # Initialize processor
    processor = get_processor()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # OCR Settings
    st.sidebar.subheader("🔍 OCR Settings")
    psm_mode = st.sidebar.selectbox(
        "Page Segmentation Mode",
        options=[6, 7, 8, 11, 13],
        index=0,
        help="6: Single block, 7: Single line, 8: Single word, 11: Sparse text, 13: Raw line"
    )
    
    show_preprocessed = st.sidebar.checkbox("Show Preprocessed Image", value=False)
    
    # Summary settings
    st.sidebar.subheader("📄 Summary Settings")
    max_summary_length = st.sidebar.slider(
        "Max Summary Length", 
        min_value=50, 
        max_value=300, 
        value=150,
        help="Maximum number of words in the summary"
    )
    
    min_summary_length = st.sidebar.slider(
        "Min Summary Length", 
        min_value=20, 
        max_value=100, 
        value=50,
        help="Minimum number of words in the summary"
    )
    
    enable_summary = st.sidebar.checkbox("Enable Summarization", value=True)
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload an image containing handwritten text"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.caption(f"📊 Size: {image.size[0]}x{image.size[1]} pixels | Mode: {image.mode} | Size: {uploaded_file.size/1024:.1f} KB")
            
            # Show preprocessed image if requested
            if show_preprocessed:
                with st.spinner("Preprocessing image..."):
                    try:
                        processed_img = processor.preprocess_image(image)
                        st.image(processed_img, caption="Preprocessed Image", use_column_width=True)
                    except Exception as e:
                        st.error(f"Error showing preprocessed image: {e}")
            
            # Process button
            if st.button("🔍 Extract & Process Text", type="primary"):
                with st.spinner("Processing image..."):
                    # Extract text
                    import time
                    start_time = time.time()
                    extracted_text = processor.extract_text_from_image(image)
                    ocr_time = time.time() - start_time
                    
                    if extracted_text.strip() and not extracted_text.startswith("❌"):
                        st.session_state.extracted_text = extracted_text
                        st.session_state.ocr_time = ocr_time
                        
                        # Generate summary if enabled
                        if enable_summary and processor.summarizer:
                            with st.spinner("Generating summary..."):
                                summary_start = time.time()
                                summary = processor.summarize_text(
                                    extracted_text,
                                    max_length=max_summary_length,
                                    min_length=min_summary_length
                                )
                                summary_time = time.time() - summary_start
                                st.session_state.summary = summary
                                st.session_state.summary_time = summary_time
                        
                        st.success(f"✅ Processing completed in {ocr_time:.2f}s!")
                    else:
                        # Show error message
                        if extracted_text.startswith("❌"):
                            st.error(extracted_text)
                        else:
                            st.error("❌ No text could be extracted from the image. Please try with a clearer image or check Tesseract installation.")
    
    with col2:
        st.subheader("📋 Results")
        
        if 'extracted_text' in st.session_state:
            # Display extracted text
            st.write("**Extracted Text:**")
            st.text_area(
                "Raw Text",
                value=st.session_state.extracted_text,
                height=200,
                key="extracted_text_display"
            )
            
            # Display summary if available
            if enable_summary and 'summary' in st.session_state:
                st.write("**Summary:**")
                st.info(st.session_state.summary)
            
            # Performance metrics
            if 'ocr_time' in st.session_state:
                col_perf1, col_perf2, col_perf3 = st.columns(3)
                with col_perf1:
                    st.metric("OCR Time", f"{st.session_state.ocr_time:.2f}s")
                with col_perf2:
                    if 'summary_time' in st.session_state:
                        st.metric("Summary Time", f"{st.session_state.summary_time:.2f}s")
                with col_perf3:
                    st.metric("Words Extracted", len(st.session_state.extracted_text.split()))
            
            # Download options
            st.subheader("💾 Download Results")
            
            # Prepare download data
            results = {
                "extracted_text": st.session_state.extracted_text,
                "summary": st.session_state.get('summary', 'Not generated'),
                "word_count": len(st.session_state.extracted_text.split()),
                "character_count": len(st.session_state.extracted_text),
                "processing_time": {
                    "ocr_time": st.session_state.get('ocr_time', 0),
                    "summary_time": st.session_state.get('summary_time', 0)
                }
            }
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                # Download as text
                st.download_button(
                    label="📄 Download Text",
                    data=st.session_state.extracted_text,
                    file_name="extracted_text.txt",
                    mime="text/plain"
                )
            
            with col_b:
                # Download as JSON
                import json
                json_data = json.dumps(results, indent=2)
                st.download_button(
                    label="📊 Download JSON",
                    data=json_data,
                    file_name="ocr_results.json",
                    mime="application/json"
                )
        else:
            st.info("👆 Upload and process an image to see results here")
    
    # API Integration section
    st.divider()
    create_api_endpoint()
    
    # Statistics
    if 'extracted_text' in st.session_state:
        st.divider()
        st.subheader("📊 Text Statistics")
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        
        with col_stats1:
            st.metric(
                "Word Count", 
                len(st.session_state.extracted_text.split())
            )
        
        with col_stats2:
            st.metric(
                "Character Count", 
                len(st.session_state.extracted_text)
            )
        
        with col_stats3:
            if 'summary' in st.session_state and not st.session_state.summary.startswith("❌"):
                compression_ratio = len(st.session_state.summary.split()) / len(st.session_state.extracted_text.split())
                st.metric(
                    "Compression Ratio", 
                    f"{compression_ratio:.2f}"
                )
            else:
                st.metric("Compression Ratio", "N/A")
        
        with col_stats4:
            if 'ocr_time' in st.session_state:
                words_per_sec = len(st.session_state.extracted_text.split()) / max(st.session_state.ocr_time, 0.1)
                st.metric(
                    "OCR Speed", 
                    f"{words_per_sec:.1f} w/s"
                )

# API endpoint for Next.js integration
if st.experimental_get_query_params().get('api'):
    @st.experimental_fragment
    def api_handler():
        """Handle API requests from Next.js"""
        if st.experimental_get_query_params().get('api') == ['process']:
            # This would be handled by a proper FastAPI endpoint
            # For now, show instructions
            st.json({
                "message": "API endpoint ready",
                "usage": "POST to /api/process with base64 image",
                "response": {
                    "extracted_text": "string",
                    "summary": "string",
                    "success": "boolean",
                    "processing_time": "float"
                }
            })
    
    api_handler()
else:
    # Run main app
    if __name__ == "__main__":
        main()