import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import requests
from transformers import pipeline
import io
import base64
from typing import Optional, Tuple, Dict, Any
import logging
import os
import platform
import subprocess
import json
import time
from datetime import datetime
import hashlib

# ================================
# ENHANCED TESSERACT SETUP
# ================================

def setup_tesseract():
    """Automatically detect and set Tesseract path with better error handling"""
    system = platform.system()
    
    if system == "Windows":
        # Expanded Windows Tesseract paths
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\tools\tesseract\tesseract.exe',
            r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
            r'D:\Program Files\Tesseract-OCR\tesseract.exe',
            r'E:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\msys64\mingw64\bin\tesseract.exe',  # MSYS2 installation
            r'C:\chocolatey\bin\tesseract.exe',      # Chocolatey installation
        ]
        
        # Try to find Tesseract
        for path in possible_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return path
        
        # Try using 'where' command with better error handling
        try:
            result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True, shell=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                paths = result.stdout.strip().split('\n')
                for path in paths:
                    if os.path.exists(path.strip()):
                        pytesseract.pytesseract.tesseract_cmd = path.strip()
                        return path.strip()
        except (subprocess.TimeoutExpired, Exception):
            pass
            
        return None
    
    elif system == "Darwin":  # macOS
        possible_paths = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
            '/usr/bin/tesseract'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    return "tesseract"  # Default for Linux

# Setup Tesseract path
tesseract_path = setup_tesseract()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        # Uncomment to add file logging:
        # logging.FileHandler('ocr_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Handwritten Text OCR & Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedHandwrittenTextProcessor:
    def __init__(self):
        """Initialize the enhanced text processor with OCR and summarization models."""
        self.summarizer = None
        self.history = []
        self.supported_languages = ['eng', 'fra', 'deu', 'spa', 'ita', 'por']
        self.setup_models()
    
    def setup_models(self):
        """Setup OCR and summarization models with better error handling."""
        try:
            st.write("🔄 Loading AI models...")
            
            # Try multiple summarization models in order of preference
            models_to_try = [
                "sshleifer/distilbart-cnn-12-6",  # Fast and reliable
                "facebook/bart-large-cnn",         # High quality but slower
                "t5-small",                        # Lightweight alternative
            ]
            
            for model_name in models_to_try:
                try:
                    self.summarizer = pipeline(
                        "summarization",
                        model=model_name,
                        device=-1,  # Use CPU
                        torch_dtype="float32"
                    )
                    st.success(f"✅ Loaded summarization model: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load {model_name}: {e}")
                    continue
                    
            if not self.summarizer:
                st.warning("⚠️ Could not load any summarization model. OCR will still work.")
                
        except Exception as e:
            logger.error(f"Error setting up models: {e}")
            st.error(f"Error setting up models: {e}")
    
    def enhanced_preprocess_image(self, image: Image.Image, preprocessing_level: str = "medium") -> np.ndarray:
        """
        Enhanced image preprocessing with multiple levels of processing.
        
        Args:
            image: PIL Image object
            preprocessing_level: "light", "medium", or "aggressive"
            
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
            
            # Dynamic resizing based on image size
            height, width = gray.shape
            if height > 3000 or width > 3000:
                scale = min(3000/height, 3000/width)
                new_height, new_width = int(height*scale), int(width*scale)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
            if preprocessing_level == "light":
                # Minimal processing
                denoised = cv2.medianBlur(gray, 3)
                _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            elif preprocessing_level == "medium":
                # Standard processing
                denoised = cv2.medianBlur(gray, 3)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                kernel = np.ones((1,1), np.uint8)
                processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
                
            else:  # aggressive
                # Advanced processing
                # 1. Advanced denoising
                denoised = cv2.fastNlMeansDenoising(gray)
                
                # 2. Adaptive histogram equalization
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                
                # 3. Multiple thresholding approaches
                _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                adaptive = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                               cv2.THRESH_BINARY, 11, 2)
                
                # Combine thresholding results
                processed = cv2.bitwise_and(otsu, adaptive)
                
                # 4. Advanced morphological operations
                kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_close)
                
                # 5. Remove small noise
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in enhanced preprocessing: {e}")
            return np.array(image.convert('L'))
    
    def extract_text_with_confidence(self, image: Image.Image, language: str = 'eng', 
                                   psm_mode: int = 6, preprocessing_level: str = "medium") -> Dict[str, Any]:
        """
        Extract text from image with confidence scores and multiple attempts.
        
        Args:
            image: PIL Image object
            language: Tesseract language code
            psm_mode: Page segmentation mode
            preprocessing_level: Level of image preprocessing
            
        Returns:
            Dictionary with text, confidence, and metadata
        """
        try:
            # Check Tesseract availability
            if not tesseract_path:
                return {
                    "text": "❌ Tesseract OCR not found!",
                    "confidence": 0,
                    "error": "Tesseract not installed"
                }
            
            # Test Tesseract
            try:
                version = pytesseract.get_tesseract_version()
            except Exception as e:
                return {
                    "text": f"❌ Tesseract error: {str(e)}",
                    "confidence": 0,
                    "error": str(e)
                }
            
            # Preprocess the image
            processed_img = self.enhanced_preprocess_image(image, preprocessing_level)
            
            # Multiple OCR attempts with different configurations
            configs = [
                f'--oem 3 --psm {psm_mode} -l {language}',
                f'--oem 3 --psm {psm_mode} -l {language} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;:()[] ',
                f'--oem 1 --psm {psm_mode} -l {language}',  # Neural network engine
            ]
            
            best_result = {"text": "", "confidence": 0}
            
            for config in configs:
                try:
                    # Get text with confidence
                    data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Calculate average confidence
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    # Extract text
                    text = pytesseract.image_to_string(processed_img, config=config)
                    cleaned_text = self.advanced_text_cleaning(text)
                    
                    if avg_confidence > best_result["confidence"] and len(cleaned_text.strip()) > 0:
                        best_result = {
                            "text": cleaned_text,
                            "confidence": avg_confidence,
                            "config": config,
                            "word_count": len(cleaned_text.split()),
                            "char_count": len(cleaned_text)
                        }
                        
                except Exception as e:
                    logger.warning(f"OCR attempt failed with config {config}: {e}")
                    continue
            
            if best_result["confidence"] == 0:
                return {
                    "text": "❌ No text could be extracted from the image.",
                    "confidence": 0,
                    "error": "OCR failed with all configurations"
                }
            
            return best_result
            
        except Exception as e:
            logger.error(f"Error in OCR extraction: {e}")
            return {
                "text": f"❌ OCR Error: {str(e)}",
                "confidence": 0,
                "error": str(e)
            }
    
    def advanced_text_cleaning(self, text: str) -> str:
        """
        Advanced text cleaning with better noise removal.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and formatted text
        """
        import re
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove single characters that are likely OCR errors (except common single letters)
        words = cleaned.split()
        valid_single_chars = {'I', 'a', 'A'}
        filtered_words = []
        
        for word in words:
            # Keep word if it's longer than 1 char, or if it's a valid single character
            if len(word) > 1 or word in valid_single_chars or word.isdigit():
                filtered_words.append(word)
        
        # Join and clean up punctuation
        result = ' '.join(filtered_words)
        
        # Fix common OCR mistakes
        replacements = {
            r'\s+([,.!?;:])': r'\1',  # Remove spaces before punctuation
            r'([,.!?;:])\s+': r'\1 ', # Normalize spaces after punctuation
            r'\s+': ' ',              # Normalize multiple spaces
        }
        
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result)
        
        return result.strip()
    
    def intelligent_summarize(self, text: str, target_ratio: float = 0.3, 
                            summary_style: str = "balanced") -> Dict[str, Any]:
        """
        Intelligent text summarization with multiple strategies.
        
        Args:
            text: Text to summarize
            target_ratio: Target compression ratio (0.1 to 0.8)
            summary_style: "extractive", "abstractive", or "balanced"
            
        Returns:
            Dictionary with summary and metadata
        """
        try:
            if not self.summarizer:
                return {
                    "summary": "❌ Summarization model not available.",
                    "method": "error",
                    "compression_ratio": 0
                }
            
            word_count = len(text.split())
            
            # Check if text is long enough
            if word_count < 20:
                return {
                    "summary": f"Text too short for summarization ({word_count} words). Original: {text}",
                    "method": "passthrough",
                    "compression_ratio": 1.0
                }
            
            # Calculate target lengths
            target_length = max(20, int(word_count * target_ratio))
            max_length = min(512, max(target_length + 20, int(word_count * 0.8)))
            min_length = max(10, min(target_length - 10, int(word_count * 0.1)))
            
            try:
                # Generate summary
                summary_result = self.summarizer(
                    text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False,
                    truncation=True
                )
                
                summary_text = summary_result[0]['summary_text']
                actual_ratio = len(summary_text.split()) / word_count
                
                return {
                    "summary": summary_text,
                    "method": "abstractive",
                    "compression_ratio": actual_ratio,
                    "original_words": word_count,
                    "summary_words": len(summary_text.split())
                }
                
            except Exception as e:
                # Fallback to extractive summarization
                logger.warning(f"Abstractive summarization failed: {e}")
                return self.extractive_summary(text, target_ratio)
                
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {
                "summary": f"❌ Summarization error: {str(e)[:100]}...",
                "method": "error",
                "compression_ratio": 0
            }
    
    def extractive_summary(self, text: str, target_ratio: float) -> Dict[str, Any]:
        """
        Simple extractive summarization as fallback.
        
        Args:
            text: Input text
            target_ratio: Target compression ratio
            
        Returns:
            Dictionary with summary and metadata
        """
        sentences = text.split('.')
        target_sentences = max(1, int(len(sentences) * target_ratio))
        
        # Simple scoring: prefer longer sentences
        scored_sentences = [(len(s.split()), s.strip()) for s in sentences if s.strip()]
        scored_sentences.sort(reverse=True)
        
        selected = [s[1] for s in scored_sentences[:target_sentences]]
        summary = '. '.join(selected)
        
        if not summary.endswith('.'):
            summary += '.'
        
        return {
            "summary": summary,
            "method": "extractive_fallback",
            "compression_ratio": len(summary.split()) / len(text.split()),
            "original_words": len(text.split()),
            "summary_words": len(summary.split())
        }
    
    def add_to_history(self, result: Dict[str, Any]):
        """Add processing result to history."""
        result['timestamp'] = datetime.now().isoformat()
        result['id'] = hashlib.md5(str(result).encode()).hexdigest()[:8]
        self.history.append(result)
        
        # Keep only last 10 results
        if len(self.history) > 10:
            self.history = self.history[-10:]

def create_enhanced_ui():
    """Create enhanced user interface."""
    
    # Custom header
    st.markdown("""
    <div class="main-header">
        <h1>📝 Advanced OCR & Text Summarizer</h1>
        <p>Extract and summarize handwritten text with AI-powered precision</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = EnhancedHandwrittenTextProcessor()
    
    processor = st.session_state.processor
    
    # Enhanced sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # System status
        st.subheader("🔧 System Status")
        show_enhanced_system_status()
        
        # OCR Settings
        st.subheader("🔍 OCR Settings")
        
        language = st.selectbox(
            "Language",
            options=['eng', 'fra', 'deu', 'spa', 'ita', 'por'],
            index=0,
            format_func=lambda x: {'eng': '🇺🇸 English', 'fra': '🇫🇷 French', 
                                  'deu': '🇩🇪 German', 'spa': '🇪🇸 Spanish',
                                  'ita': '🇮🇹 Italian', 'por': '🇵🇹 Portuguese'}[x]
        )
        
        psm_mode = st.selectbox(
            "Page Segmentation",
            options=[6, 7, 8, 11, 13],
            index=0,
            format_func=lambda x: {6: "Single block", 7: "Single line", 
                                  8: "Single word", 11: "Sparse text", 
                                  13: "Raw line"}[x]
        )
        
        preprocessing_level = st.selectbox(
            "Preprocessing Level",
            options=["light", "medium", "aggressive"],
            index=1,
            help="Higher levels may improve accuracy but take longer"
        )
        
        show_preprocessed = st.checkbox("Show Preprocessed Image", value=False)
        
        # Summary Settings
        st.subheader("📄 Summary Settings")
        
        enable_summary = st.checkbox("Enable Summarization", value=True)
        
        if enable_summary:
            target_ratio = st.slider(
                "Compression Ratio", 
                min_value=0.1, 
                max_value=0.8, 
                value=0.3,
                step=0.1,
                help="Lower values create shorter summaries"
            )
            
            summary_style = st.selectbox(
                "Summary Style",
                options=["balanced", "extractive", "abstractive"],
                index=0
            )
        
        # Performance Settings
        st.subheader("⚡ Performance")
        
        batch_processing = st.checkbox("Enable Batch Processing", value=False)
        save_history = st.checkbox("Save Processing History", value=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload & Process")
        
        # File uploader with drag and drop
        uploaded_files = st.file_uploader(
            "Choose image files",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            accept_multiple_files=batch_processing,
            help="Upload images containing handwritten or printed text"
        )
        
        if uploaded_files:
            files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
            
            for idx, uploaded_file in enumerate(files):
                st.write(f"**File {idx + 1}: {uploaded_file.name}**")
                
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded Image {idx + 1}", use_column_width=True)
                
                # Image info
                st.caption(f"📊 {image.size[0]}×{image.size[1]} | {image.mode} | {uploaded_file.size/1024:.1f} KB")
                
                # Show preprocessed image
                if show_preprocessed:
                    with st.spinner("Preprocessing..."):
                        processed_img = processor.enhanced_preprocess_image(image, preprocessing_level)
                        st.image(processed_img, caption=f"Preprocessed {idx + 1}", use_column_width=True)
                
                # Process button
                process_key = f"process_{idx}"
                if st.button(f"🔍 Process Image {idx + 1}", key=process_key, type="primary"):
                    process_single_image(image, processor, language, psm_mode, 
                                       preprocessing_level, enable_summary, 
                                       target_ratio if enable_summary else 0.3,
                                       summary_style if enable_summary else "balanced",
                                       idx, save_history)
    
    with col2:
        st.subheader("📋 Results")
        show_results_panel()
    
    # History section
    if save_history and processor.history:
        st.divider()
        show_processing_history(processor.history)

def process_single_image(image, processor, language, psm_mode, preprocessing_level, 
                        enable_summary, target_ratio, summary_style, idx, save_history):
    """Process a single image and store results."""
    
    with st.spinner(f"Processing image {idx + 1}..."):
        start_time = time.time()
        
        # Extract text with confidence
        ocr_result = processor.extract_text_with_confidence(
            image, language, psm_mode, preprocessing_level
        )
        
        ocr_time = time.time() - start_time
        
        if ocr_result.get("confidence", 0) > 0:
            result = {
                "image_idx": idx,
                "extracted_text": ocr_result["text"],
                "confidence": ocr_result["confidence"],
                "ocr_time": ocr_time,
                "word_count": ocr_result.get("word_count", 0),
                "char_count": ocr_result.get("char_count", 0)
            }
            
            # Generate summary if enabled
            if enable_summary and processor.summarizer and len(ocr_result["text"].split()) > 10:
                summary_start = time.time()
                summary_result = processor.intelligent_summarize(
                    ocr_result["text"], target_ratio, summary_style
                )
                summary_time = time.time() - summary_start
                
                result.update({
                    "summary": summary_result["summary"],
                    "summary_method": summary_result["method"],
                    "compression_ratio": summary_result["compression_ratio"],
                    "summary_time": summary_time
                })
            
            # Store in session state
            if f"results" not in st.session_state:
                st.session_state.results = {}
            
            st.session_state.results[idx] = result
            
            # Add to history
            if save_history:
                processor.add_to_history(result)
            
            st.success(f"✅ Image {idx + 1} processed successfully! "
                      f"Confidence: {ocr_result['confidence']:.1f}%")
        else:
            st.error(f"❌ Failed to process image {idx + 1}: {ocr_result.get('error', 'Unknown error')}")

def show_results_panel():
    """Display results panel with enhanced formatting."""
    
    if 'results' in st.session_state and st.session_state.results:
        # Results tabs for multiple images
        if len(st.session_state.results) > 1:
            tabs = st.tabs([f"Image {i+1}" for i in st.session_state.results.keys()])
            
            for tab, (idx, result) in zip(tabs, st.session_state.results.items()):
                with tab:
                    display_single_result(result, idx)
        else:
            # Single result
            result = list(st.session_state.results.values())[0]
            idx = list(st.session_state.results.keys())[0]
            display_single_result(result, idx)
    else:
        st.info("👆 Upload and process images to see results here")

def display_single_result(result, idx):
    """Display a single result with enhanced formatting."""
    
    # Extracted text
    st.write("**📝 Extracted Text:**")
    st.text_area(
        f"Text from Image {idx + 1}",
        value=result["extracted_text"],
        height=150,
        key=f"text_{idx}"
    )
    
    # Confidence and metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "🟢" if result["confidence"] > 70 else "🟡" if result["confidence"] > 40 else "🔴"
        st.metric("Confidence", f"{result['confidence']:.1f}%", delta=None)
        st.write(f"{confidence_color} Quality: {'High' if result['confidence'] > 70 else 'Medium' if result['confidence'] > 40 else 'Low'}")
    
    with col2:
        st.metric("Words", result["word_count"])
        st.metric("Characters", result["char_count"])
    
    with col3:
        st.metric("OCR Time", f"{result['ocr_time']:.2f}s")
        if 'summary_time' in result:
            st.metric("Summary Time", f"{result['summary_time']:.2f}s")
    
    # Summary if available
    if 'summary' in result and not result['summary'].startswith('❌'):
        st.write("**📄 Summary:**")
        st.info(result['summary'])
        
        if 'compression_ratio' in result:
            st.caption(f"Compression: {result['compression_ratio']:.2f} | "
                      f"Method: {result.get('summary_method', 'unknown')}")
    
    # Download options
    st.write("**💾 Download:**")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.download_button(
            "📄 Text (.txt)",
            data=result["extracted_text"],
            file_name=f"extracted_text_{idx+1}.txt",
            mime="text/plain",
            key=f"download_txt_{idx}"
        )
    
    with col_b:
        if 'summary' in result:
            st.download_button(
                "📋 Summary (.txt)",
                data=result["summary"],
                file_name=f"summary_{idx+1}.txt",
                mime="text/plain",
                key=f"download_summary_{idx}"
            )
    
    with col_c:
        # Full result as JSON
        json_data = json.dumps(result, indent=2, default=str)
        st.download_button(
            "📊 Full Result (.json)",
            data=json_data,
            file_name=f"ocr_result_{idx+1}.json",
            mime="application/json",
            key=f"download_json_{idx}"
        )

def show_enhanced_system_status():
    """Enhanced system status display."""
    
    if tesseract_path:
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract v{version}")
            
            if platform.system() == "Windows":
                st.caption(f"📂 {tesseract_path}")
            
            # Test OCR functionality
            test_img = np.ones((50, 200), dtype=np.uint8) * 255
            cv2.putText(test_img, "TEST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 0, 2)
            
            try:
                test_result = pytesseract.image_to_string(test_img)
                if "TEST" in test_result:
                    st.success("🧪 OCR Test: Passed")
                else:
                    st.warning("🧪 OCR Test: Partial")
            except Exception as e:
                st.error(f"🧪 OCR Test: Failed - {str(e)[:30]}...")
                
        except Exception as e:
            st.error(f"❌ Tesseract Error")
            show_installation_guide()
    else:
        st.error("❌ Tesseract not found!")
        show_installation_guide()

def show_installation_guide():
    """Show Tesseract installation guide."""
    
    with st.expander("🛠️ Installation Guide"):
        system = platform.system()
        
        if system == "Windows":
            st.markdown("""
            **Windows Installation:**
            
            1. **Download:** [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki)
            2. **Install** to default location
            3. **Add to code:**
            ```python
            import pytesseract
            pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
            ```
            4. **Or add to PATH** (System Properties → Environment Variables)
            """)
            
        elif system == "Darwin":  # macOS
            st.markdown("""
            **macOS Installation:**
            
            Using Homebrew:
            ```bash
            brew install tesseract
            ```
            
            Using MacPorts:
            ```bash
            sudo port install tesseract
            ```
            """)
            
        else:  # Linux
            st.markdown("""
            **Linux Installation:**
            
            Ubuntu/Debian:
            ```bash
            sudo apt update
            sudo apt install tesseract-ocr
            ```
            
            CentOS/RHEL:
            ```bash
            sudo yum install epel-release
            sudo yum install tesseract
            ```
            """)

def show_processing_history(history):
    """Display processing history with enhanced visualization."""
    
    st.subheader("📚 Processing History")
    
    if not history:
        st.info("No processing history available")
        return
    
    # History summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Processed", len(history))
    
    with col2:
        avg_confidence = np.mean([h.get('confidence', 0) for h in history])
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col3:
        total_words = sum([h.get('word_count', 0) for h in history])
        st.metric("Total Words", total_words)
    
    with col4:
        avg_time = np.mean([h.get('ocr_time', 0) for h in history])
        st.metric("Avg OCR Time", f"{avg_time:.2f}s")
    
    # Detailed history
    with st.expander("📋 View Detailed History", expanded=False):
        for i, item in enumerate(reversed(history[-5:])):  # Show last 5
            with st.container():
                col_a, col_b, col_c = st.columns([2, 1, 1])
                
                with col_a:
                    preview = item.get('extracted_text', '')[:100]
                    if len(preview) == 100:
                        preview += "..."
                    st.write(f"**{i+1}.** {preview}")
                
                with col_b:
                    st.caption(f"Confidence: {item.get('confidence', 0):.1f}%")
                    st.caption(f"Words: {item.get('word_count', 0)}")
                
                with col_c:
                    timestamp = item.get('timestamp', 'Unknown')
                    if timestamp != 'Unknown':
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            st.caption(f"Time: {dt.strftime('%H:%M:%S')}")
                        except:
                            st.caption(f"Time: {timestamp}")
                    
                    # Download individual result
                    if st.button(f"⬇️ Download", key=f"hist_download_{item.get('id', i)}"):
                        result_json = json.dumps(item, indent=2, default=str)
                        st.download_button(
                            "📄 Download Result",
                            data=result_json,
                            file_name=f"history_result_{item.get('id', i)}.json",
                            mime="application/json",
                            key=f"hist_dl_btn_{item.get('id', i)}"
                        )
                
                st.divider()
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        if 'processor' in st.session_state:
            st.session_state.processor.history.clear()
        st.success("History cleared!")
        st.experimental_rerun()

def create_api_documentation():
    """Create comprehensive API documentation."""
    
    st.subheader("🔌 API Integration")
    
    with st.expander("📖 API Documentation", expanded=False):
        st.markdown("""
        ### REST API Endpoints
        
        #### POST /api/ocr
        Extract text from image
        
        **Request:**
        ```json
        {
            "image": "base64_encoded_image",
            "language": "eng",
            "psm_mode": 6,
            "preprocessing_level": "medium"
        }
        ```
        
        **Response:**
        ```json
        {
            "success": true,
            "text": "extracted text",
            "confidence": 85.5,
            "word_count": 42,
            "processing_time": 1.23
        }
        ```
        
        #### POST /api/summarize
        Summarize extracted text
        
        **Request:**
        ```json
        {
            "text": "text to summarize",
            "target_ratio": 0.3,
            "style": "balanced"
        }
        ```
        
        **Response:**
        ```json
        {
            "success": true,
            "summary": "summarized text",
            "compression_ratio": 0.28,
            "method": "abstractive"
        }
        ```
        """)
        
        # JavaScript integration example
        st.code("""
        // Next.js integration example
        const processImage = async (imageFile) => {
            const formData = new FormData();
            formData.append('image', imageFile);
            formData.append('language', 'eng');
            formData.append('preprocessing_level', 'medium');
            
            const response = await fetch('/api/ocr-process', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.success) {
                console.log('Extracted text:', result.text);
                console.log('Confidence:', result.confidence);
                
                // Optional: Get summary
                if (result.text && result.word_count > 20) {
                    const summaryResponse = await fetch('/api/summarize', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            text: result.text,
                            target_ratio: 0.3
                        })
                    });
                    
                    const summaryResult = await summaryResponse.json();
                    console.log('Summary:', summaryResult.summary);
                }
            }
        };
        
        // React component example
        const OCRComponent = () => {
            const [result, setResult] = useState(null);
            const [loading, setLoading] = useState(false);
            
            const handleFileUpload = async (event) => {
                const file = event.target.files[0];
                if (!file) return;
                
                setLoading(true);
                try {
                    const result = await processImage(file);
                    setResult(result);
                } catch (error) {
                    console.error('OCR processing failed:', error);
                } finally {
                    setLoading(false);
                }
            };
            
            return (
                <div>
                    <input type="file" accept="image/*" onChange={handleFileUpload} />
                    {loading && <p>Processing...</p>}
                    {result && (
                        <div>
                            <h3>Extracted Text:</h3>
                            <p>{result.text}</p>
                            <p>Confidence: {result.confidence}%</p>
                        </div>
                    )}
                </div>
            );
        };
        """, language="javascript")

def create_performance_monitor():
    """Create performance monitoring dashboard."""
    
    if 'results' not in st.session_state or not st.session_state.results:
        return
    
    st.subheader("📊 Performance Analytics")
    
    results = list(st.session_state.results.values())
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_confidence = np.mean([r['confidence'] for r in results])
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")
    
    with col2:
        total_processing_time = sum([r['ocr_time'] for r in results])
        st.metric("Total Processing Time", f"{total_processing_time:.2f}s")
    
    with col3:
        avg_words_per_image = np.mean([r['word_count'] for r in results])
        st.metric("Avg Words/Image", f"{avg_words_per_image:.0f}")
    
    with col4:
        if any('summary_time' in r for r in results):
            summary_times = [r.get('summary_time', 0) for r in results if 'summary_time' in r]
            avg_summary_time = np.mean(summary_times) if summary_times else 0
            st.metric("Avg Summary Time", f"{avg_summary_time:.2f}s")
    
    # Performance visualization
    if len(results) > 1:
        try:
            import plotly.express as px
            import pandas as pd
            
            # Create performance dataframe
            df = pd.DataFrame([
                {
                    'Image': f"Image {i+1}",
                    'Confidence': r['confidence'],
                    'OCR Time': r['ocr_time'],
                    'Word Count': r['word_count']
                }
                for i, r in enumerate(results)
            ])
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig1 = px.bar(df, x='Image', y='Confidence', 
                             title='Confidence by Image',
                             color='Confidence',
                             color_continuous_scale='RdYlGn')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_chart2:
                fig2 = px.scatter(df, x='Word Count', y='OCR Time',
                                 size='Confidence', title='Processing Performance',
                                 hover_data=['Image'])
                st.plotly_chart(fig2, use_container_width=True)
                
        except ImportError:
            st.info("Install plotly for performance visualizations: `pip install plotly`")

def main():
    """Enhanced main function with comprehensive features."""
    
    try:
        # Initialize session state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.results = {}
        
        # Create main UI
        create_enhanced_ui()
        
        # Performance monitoring
        if st.session_state.results:
            st.divider()
            create_performance_monitor()
        
        # API documentation
        st.divider()
        create_api_documentation()
        
        # Footer with additional info
        st.divider()
        col_foot1, col_foot2, col_foot3 = st.columns(3)
        
        with col_foot1:
            st.markdown("**🔧 System Requirements:**")
            st.markdown("""
            - Python 3.7+
            - Tesseract OCR 4.0+
            - OpenCV 4.0+
            - Transformers library
            """)
        
        with col_foot2:
            st.markdown("**📚 Supported Formats:**")
            st.markdown("""
            - PNG, JPEG, TIFF
            - BMP, WebP
            - Multi-page documents
            - Handwritten & printed text
            """)
        
        with col_foot3:
            st.markdown("**🚀 Features:**")
            st.markdown("""
            - Multi-language OCR
            - AI-powered summarization
            - Batch processing
            - Performance analytics
            """)
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {e}", exc_info=True)
        
        # Recovery options
        st.subheader("🔧 Recovery Options")
        if st.button("Reset Application State"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

# Run the application
if __name__ == "__main__":
    main()