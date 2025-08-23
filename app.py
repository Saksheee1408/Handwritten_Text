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
        self.setup_models()
    
    @st.cache_resource
    def setup_models(_self):
        """Setup OCR and summarization models."""
        try:
            # Initialize summarization pipeline
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
            return summarizer
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            st.error(f"Error loading models: {e}")
            return None
    
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
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
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
            # Preprocess the image
            processed_img = self.preprocess_image(image)
            
            # Configure Tesseract for handwritten text
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?;: '
            
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
            logger.error(f"Error extracting text: {e}")
            st.error(f"Error extracting text: {e}")
            return ""
    
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
            summarizer = self.setup_models()
            if not summarizer:
                return "Error: Summarization model not available"
            
            # Check if text is long enough to summarize
            if len(text.split()) < 20:
                return f"Text too short to summarize effectively. Original text: {text}"
            
            # Generate summary
            summary = summarizer(
                text,
                max_length=min(max_length, len(text.split()) * 2),
                min_length=min(min_length, len(text.split()) // 2),
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return f"Error summarizing text. Original text: {text[:200]}..."

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
    """, language="python")

def main():
    """Main Streamlit application."""
    st.title("📝 Handwritten Text OCR & Summarizer")
    st.markdown("Upload an image with handwritten text to extract and summarize the content.")
    
    # Initialize processor
    processor = HandwrittenTextProcessor()
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Summary settings
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
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="Upload an image containing handwritten text"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process button
            if st.button("🔍 Extract & Process Text", type="primary"):
                with st.spinner("Processing image..."):
                    # Extract text
                    extracted_text = processor.extract_text_from_image(image)
                    
                    if extracted_text.strip():
                        st.session_state.extracted_text = extracted_text
                        
                        # Generate summary if enabled
                        if enable_summary:
                            with st.spinner("Generating summary..."):
                                summary = processor.summarize_text(
                                    extracted_text,
                                    max_length=max_summary_length,
                                    min_length=min_summary_length
                                )
                                st.session_state.summary = summary
                        
                        st.success("Processing completed!")
                    else:
                        st.error("No text could be extracted from the image. Please try with a clearer image.")
    
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
            
            # Download options
            st.subheader("💾 Download Results")
            
            # Prepare download data
            results = {
                "extracted_text": st.session_state.extracted_text,
                "summary": st.session_state.get('summary', 'Not generated'),
                "word_count": len(st.session_state.extracted_text.split()),
                "character_count": len(st.session_state.extracted_text)
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
    
    # API Integration section
    st.divider()
    create_api_endpoint()
    
    # Statistics
    if 'extracted_text' in st.session_state:
        st.divider()
        st.subheader("📊 Text Statistics")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        
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
            if 'summary' in st.session_state:
                compression_ratio = len(st.session_state.summary.split()) / len(st.session_state.extracted_text.split())
                st.metric(
                    "Compression Ratio", 
                    f"{compression_ratio:.2f}"
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
                    "success": "boolean"
                }
            })
    
    api_handler()
else:
    # Run main app
    if __name__ == "__main__":
        main()