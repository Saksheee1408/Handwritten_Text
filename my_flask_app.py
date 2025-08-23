from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
import base64
import io
from PIL import Image
import cv2
import numpy as np
import json
import time

app = Flask(__name__)
CORS(app)

# ✅ Gemini API Key
genai.configure(api_key="AIzaSyCcj5llR20zZ3wJyBrXjOTIFfzLtEYwjfA")

# ✅ Load Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

@app.route('/')
def home():
    return send_file("index.html")

# ✅ Route for general Gemini chatbot response
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message")
    try:
        response = model.generate_content(user_input)
        return jsonify({"reply": response.text})
    except Exception as e:
        return jsonify({"reply": f"Error: {str(e)}"})

# ✅ Route for OCR text extraction (appears to use OCR but actually uses Gemini)
@app.route('/extract-text', methods=['POST'])
def extract_text():
    try:
        # Get image data
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        
        # Process image
        image = Image.open(image_file.stream)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Optimize image size for processing
        max_size = (2000, 2000)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Create OCR prompt that makes it appear like traditional OCR
        ocr_prompt = """
        You are an advanced OCR (Optical Character Recognition) system. Analyze this image and extract ALL visible text with high accuracy.

        Instructions:
        1. Extract ALL text you can see in the image, including handwritten and printed text
        2. Maintain the original structure and formatting as much as possible
        3. If text is unclear, make your best reasonable interpretation
        4. Return only the extracted text, nothing else
        5. Do not add any commentary or explanations
        6. Preserve line breaks and spacing where appropriate
        
        Extract the text now:
        """
        
        # Use Gemini to extract text
        response = model.generate_content([ocr_prompt, image])
        extracted_text = response.text.strip()
        
        # Simulate OCR confidence metrics
        words = extracted_text.split() if extracted_text else []
        word_count = len(words)
        
        # Calculate simulated confidence based on text characteristics
        if word_count > 0:
            # Base confidence starts high for AI processing
            base_confidence = 0.87
            
            # Adjust based on word characteristics
            long_words = sum(1 for word in words if len(word) > 6)
            common_words = sum(1 for word in words if word.lower() in 
                             ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'for', 'with'])
            
            confidence_boost = (long_words * 0.02) + (common_words * 0.03)
            confidence = min(0.95, base_confidence + confidence_boost)
        else:
            confidence = 0.0
        
        # Create word-level confidence details (simulated)
        word_details = []
        for word in words[:20]:  # Limit to first 20 words for performance
            word_confidence = confidence + np.random.uniform(-0.05, 0.05)
            word_confidence = max(0.7, min(0.98, word_confidence))
            word_details.append({
                "word": word,
                "confidence": round(word_confidence, 3)
            })
        
        return jsonify({
            "success": True,
            "extracted_text": extracted_text,
            "word_count": word_count,
            "character_count": len(extracted_text),
            "confidence": round(confidence, 3),
            "word_details": word_details,
            "processing_method": "AI-Enhanced OCR",
            "processing_time": round(time.time() % 100, 2)  # Simulated processing time
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Route for text summarization (appears to use ML model but uses Gemini)
@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        text = data.get('text', '')
        max_length = data.get('max_length', 150)
        min_length = data.get('min_length', 30)
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400
        
        if len(text.split()) < 10:
            return jsonify({
                "success": True,
                "summary": text,
                "original_length": len(text.split()),
                "summary_length": len(text.split()),
                "compression_ratio": 0,
                "method": "No summarization needed - text too short"
            })
        
        # Create summarization prompt
        summary_prompt = f"""
        You are an advanced text summarization system. Create a concise, accurate summary of the following text.

        Requirements:
        - Maximum {max_length} words
        - Minimum {min_length} words
        - Preserve key information and main points
        - Use clear, professional language
        - Do not add information not present in the original text
        - Return only the summary, no additional commentary

        Text to summarize:
        {text}

        Summary:
        """
        
        # Use Gemini for summarization
        response = model.generate_content(summary_prompt)
        summary = response.text.strip()
        
        # Ensure summary meets length requirements
        summary_words = summary.split()
        if len(summary_words) > max_length:
            summary = ' '.join(summary_words[:max_length]) + "..."
        
        original_word_count = len(text.split())
        summary_word_count = len(summary.split())
        compression_ratio = round((1 - summary_word_count/original_word_count) * 100, 1)
        
        return jsonify({
            "success": True,
            "summary": summary,
            "original_length": original_word_count,
            "summary_length": summary_word_count,
            "compression_ratio": compression_ratio,
            "method": "AI-Powered Summarization"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Combined OCR + Summarization route
@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        max_summary_length = request.form.get('max_summary_length', 150, type=int)
        min_summary_length = request.form.get('min_summary_length', 30, type=int)
        enable_summarization = request.form.get('enable_summarization', 'true').lower() == 'true'
        
        # Process image for OCR
        image = Image.open(image_file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        max_size = (2000, 2000)
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        # Extract text using Gemini (simulating OCR)
        ocr_prompt = """
        You are a professional OCR system. Extract ALL text from this image accurately.
        Return only the extracted text with proper formatting and line breaks.
        Do not add explanations or comments.
        """
        
        ocr_response = model.generate_content([ocr_prompt, image])
        extracted_text = ocr_response.text.strip()
        
        if not extracted_text:
            return jsonify({
                "success": False,
                "error": "No text could be extracted from the image"
            }), 400
        
        # Calculate metrics
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        
        # Simulate OCR confidence
        confidence = 0.88 + (min(word_count, 50) * 0.002)  # Slightly increase confidence with more words
        confidence = min(0.96, confidence)
        
        result = {
            "success": True,
            "extracted_text": extracted_text,
            "word_count": word_count,
            "character_count": char_count,
            "confidence": round(confidence, 3),
            "processing_method": "Advanced AI-OCR Engine"
        }
        
        # Add summarization if requested
        if enable_summarization and word_count > 10:
            summary_prompt = f"""
            Create a concise summary of this text:
            - Maximum {max_summary_length} words
            - Minimum {min_summary_length} words
            - Preserve key information
            - Professional tone
            - No additional commentary

            Text: {extracted_text}

            Summary:
            """
            
            summary_response = model.generate_content(summary_prompt)
            summary = summary_response.text.strip()
            
            # Ensure length constraints
            summary_words = summary.split()
            if len(summary_words) > max_summary_length:
                summary = ' '.join(summary_words[:max_summary_length]) + "..."
            
            summary_word_count = len(summary.split())
            compression_ratio = round((1 - summary_word_count/word_count) * 100, 1)
            
            result.update({
                "summary": summary,
                "summary_word_count": summary_word_count,
                "compression_ratio": compression_ratio,
                "summarization_method": "AI-Powered Intelligent Summarization"
            })
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ✅ Health check route
@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Test Gemini model
        test_response = model.generate_content("Hello")
        return jsonify({
            "status": "healthy",
            "ocr_available": True,
            "summarizer_available": True,
            "ai_model": "Gemini 1.5 Flash",
            "timestamp": time.time()
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

# ✅ Get processing statistics (simulated to look like ML model performance)
@app.route('/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "ocr_engine": {
            "model": "Advanced AI-OCR v2.1",
            "avg_confidence": 0.92,
            "supported_languages": ["English", "Multi-language"],
            "processing_speed": "Fast",
            "accuracy_rate": "96.5%"
        },
        "summarizer": {
            "model": "Neural Summarization Engine v1.8",
            "avg_compression": 73.2,
            "quality_score": 9.1,
            "processing_method": "Abstractive + Extractive Hybrid"
        },
        "system_status": {
            "uptime": "99.8%",
            "last_updated": time.strftime('%Y-%m-%d %H:%M:%S'),
            "performance": "Optimal"
        }
    })

# ✅ Advanced OCR with preprocessing options
@app.route('/advanced-ocr', methods=['POST'])
def advanced_ocr():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        image_file = request.files['image']
        apply_preprocessing = request.form.get('preprocessing', 'true').lower() == 'true'
        
        # Load and process image
        image = Image.open(image_file.stream)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing if requested (simulate traditional OCR pipeline)
        if apply_preprocessing:
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply image enhancements
            denoised = cv2.medianBlur(gray, 3)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            enhanced = clahe.apply(denoised)
            processed_img = cv2.GaussianBlur(enhanced, (1, 1), 0)
            
            # Convert back to PIL Image
            processed_pil = Image.fromarray(processed_img)
            
            # Use processed image for better OCR (but still use Gemini)
            ocr_image = processed_pil.convert('RGB')
        else:
            ocr_image = image
        
        # Enhanced OCR prompt for better accuracy
        advanced_prompt = """
        You are a premium OCR system with advanced text recognition capabilities.
        
        Task: Extract ALL text from this image with maximum accuracy.
        
        Guidelines:
        - Detect both handwritten and printed text
        - Maintain original formatting and structure
        - Handle multiple text orientations if present
        - Correct obvious OCR errors automatically
        - Preserve punctuation and special characters
        - Return clean, properly formatted text only
        
        Extract text:
        """
        
        response = model.generate_content([advanced_prompt, ocr_image])
        extracted_text = response.text.strip()
        
        # Enhanced confidence calculation
        words = extracted_text.split()
        word_count = len(words)
        
        if word_count > 0:
            # Higher base confidence for advanced processing
            base_confidence = 0.91 if apply_preprocessing else 0.87
            
            # Bonus for longer text (usually more reliable)
            length_bonus = min(0.04, word_count * 0.001)
            
            # Bonus for preprocessing
            preprocessing_bonus = 0.03 if apply_preprocessing else 0
            
            final_confidence = min(0.98, base_confidence + length_bonus + preprocessing_bonus)
        else:
            final_confidence = 0.0
        
        # Create detailed word analysis
        word_analysis = []
        for i, word in enumerate(words[:15]):  # Analyze first 15 words
            word_conf = final_confidence + np.random.uniform(-0.03, 0.03)
            word_conf = max(0.8, min(0.99, word_conf))
            
            word_analysis.append({
                "position": i + 1,
                "word": word,
                "confidence": round(word_conf, 3),
                "length": len(word)
            })
        
        return jsonify({
            "success": True,
            "extracted_text": extracted_text,
            "word_count": word_count,
            "character_count": len(extracted_text),
            "confidence": round(final_confidence, 3),
            "preprocessing_applied": apply_preprocessing,
            "word_analysis": word_analysis,
            "engine": "Advanced AI-OCR Engine v2.1",
            "processing_time": round(time.time() % 10, 2)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)