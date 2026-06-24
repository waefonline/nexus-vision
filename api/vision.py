"""
Nexus Vision API - Proxy for Gemini Vision
Extracts trading signals from images using Google Gemini Vision AI.

Updated to use the new google.genai SDK (December 2024)
Replaces deprecated google.generativeai library
"""

import os
import json
import base64
from http.server import BaseHTTPRequestHandler

# Prompt for signal extraction
SIGNAL_EXTRACTION_PROMPT = """
Analyze this trading signal image and extract the following information if present:
- Symbol (e.g., XAUUSD, EURUSD, GOLD, etc.)
- Action (BUY or SELL)
- Entry price or entry zone (range)
- Stop Loss (SL)
- Take Profit levels (TP1, TP2, TP3, etc.)

Return ONLY the extracted text in a simple format like:
SYMBOL ACTION @ ENTRY
SL: [value]
TP1: [value]
TP2: [value]
...

If no trading signal is found in the image, return: NO_SIGNAL_FOUND

Be concise. Return only the signal data, no explanations.
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {
            "status": "ok",
            "service": "nexus-vision-api",
            "version": "2.0.0",  # Updated version for new SDK
            "sdk": "google-genai"
        }
        self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """Handle POST request with image data."""
        try:
            # Import the new Google GenAI SDK
            from google import genai
            
            # Get API key from environment
            api_key = os.environ.get("_API_KEY")
            if not api_key:
                self._send_error(500, "_API_KEY not configured")
                return
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self._send_error(400, "No image data provided")
                return
            
            body = self.rfile.read(content_length)
            
            # Parse JSON body
            try:
                data = json.loads(body)
                image_base64 = data.get("image")
                if not image_base64:
                    self._send_error(400, "Missing 'image' field in request")
                    return
            except json.JSONDecodeError:
                image_base64 = base64.b64encode(body).decode('utf-8')
            
            # Create client with API key
            client = genai.Client(api_key=api_key)
            
            # Decode image
            image_bytes = base64.b64decode(image_base64)
            
            # Create image part for Gemini (new format)
            image_part = {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": image_base64  # Already base64 encoded
                }
            }
            
            # Generate content using new SDK structure
            response = client.models.generate_content(
                model='gemini-flash-latest',
                contents=[
                    SIGNAL_EXTRACTION_PROMPT,
                    image_part
                ]
            )
            extracted_text = response.text.strip()
            
            # Send success response
            self._send_json(200, {
                "success": True,
                "extracted_text": extracted_text,
                "is_signal": "NO_SIGNAL_FOUND" not in extracted_text
            })
            
        except Exception as e:
            self._send_error(500, f"Error processing image: {str(e)}")
    
    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, status_code, message):
        self._send_json(status_code, {"success": False, "error": message})
