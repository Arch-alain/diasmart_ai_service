import os
import json
import PIL.Image
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Fetch the API key from the environment
API_KEY = os.getenv("LLM_API_KEY")

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if not API_KEY:
        return jsonify({"error": "LLM_API_KEY is not set on the server"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "Missing image file in request"}), 400
        
    image_file = request.files['image']
    
    try:
        # 1. Read the image for Gemini
        image_file.seek(0)
        food_image = PIL.Image.open(image_file)
        
        # 2. Initialize Gemini
        client = genai.Client(api_key=API_KEY)
        
        # 3. Structure the prompt for strict JSON
        prompt = (
            "Analyze this image and describe the food shown. "
            "1. Identify each visible ingredient.\n"
            "2. Estimate the portion size in grams for each item.\n"
            "3. Provide an estimated calorie count for each item based on those sizes.\n"
            "4. Calculate the total estimated calories.\n\n"
            "IMPORTANT: You must return ONLY a valid JSON object matching this exact structure:\n"
            "{\n"
            '  "ingredients": [\n'
            '    {"name": "Cassava Leaves", "estimated_grams": 150, "calories": 120},\n'
            '    {"name": "Beans", "estimated_grams": 100, "calories": 130}\n'
            "  ],\n"
            '  "total_calories": 250.0\n'
            "}"
        )
        
        # 4. Call the LLM
        response = client.models.generate_content(
            model='gemini-3.6-flash', 
            contents=[food_image, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # 5. Return the JSON to the user
        return jsonify(json.loads(response.text)), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Start the local development server
if __name__ == '__main__':
    app.run(debug=True, port=8000)