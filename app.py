import os
import cv2
import pytesseract
import json
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from flask import Flask,Response, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
import logging
import pandas as pd
import io

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key in production

# Set upload and output directories
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'gif'}

# Set the path to your Tesseract executable if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# Load the MahaNER-BERT model and tokenizer
model_name = "l3cube-pune/marathi-ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Create a NER pipeline
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Step 1: Preprocess the image for better OCR detection
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Smoothing with a larger kernel for better text clarity
    # Adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    return thresh

# Step 2: Detect text boxes (main target boxes)
def detect_boxes(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    # Sort contours by position (top-to-bottom, left-to-right)
    contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 50 and h > 20:  # Adjust size threshold based on expected box sizes
            boxes.append((x, y, w, h))
    
    return boxes

# Step 3: Extract text from each detected box
def extract_text_from_box(image, lang='eng+mar'):
    config = '--psm 6 --oem 3'  # Use OEM 3 for both legacy and LSTM OCR engine for better accuracy
    return pytesseract.image_to_string(image, lang=lang, config=config)

def post_processing(extracted_text):
    corrected_text = extracted_text
    # 2. Custom Replacement Rules
    custom_replacements = {
        'aren': 'आशा',
        'RO': 'SRO',
        '85१': 'SRO',
        '810': 'SRO',
        '880': 'SRO',
        '5२0': 'SRO',
        '380': 'SRO',
        '8१0': 'SRO',
        'SSRO': 'SRO',
        '890': 'SRO',
        '800': 'SRO',
        '88३0': 'SRO',
        'SAO': 'SRO',
        '8२0': 'SRO',
        '350': 'SRO',
        '3१०': 'SRO',
        '580': 'SRO',
        '310': 'SRO',
        '590': 'SRO',
        '580': 'SRO',
        'R0': 'SRO',
        '3२0': 'SRO',
        '8४0': 'SRO',
        '80': 'SRO',
        '850': 'SRO',
        '४४४': 'JVW',
        'uvw': 'JVW',
        'JVvW': 'JVW',
        'JVwW': 'JVW',
        'JVw': 'JVW',
        'Jvw': 'JVW',
        'uvwo': 'JVW0',
        'JVWO': 'JVW0',
        'JWWOR': 'JVW',
        'JSVW': 'JVW',
    }

    for wrong, right in custom_replacements.items():
        corrected_text = corrected_text.replace(wrong, right)

    # 3. Pattern Enforcement for Alphanumeric Strings
    corrected_text = re.sub(r'\b([a-z])\b', lambda x: x.group(1).upper(), corrected_text)  # Capitalize single letters

    # 4. Removing Non-Alphanumeric Characters (with Devanagari support)
    corrected_text = re.sub(r'[^\u0900-\u097F A-Za-z0-9\s:]', '', corrected_text)  # Remove unwanted characters

    return corrected_text

def clean_input(text):
    text = text.replace('*', '')
    text = text.replace(';', ':')

    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.replace("छायाचित्र", "").strip()
        line = line.replace("उपलब्ध", "").strip()

        if ':' in line:
            parts = line.split(':')
            for part in parts:
                if part.strip():  # Ensure non-empty strings
                    cleaned_lines.append(part.strip() + ":")
        elif line:
            cleaned_lines.append(line.strip())

    cleaned_text = '\n'.join(cleaned_lines)
    return cleaned_text

def extract_entities(text):
    cleaned_text = clean_input(text)
    results = ner_pipeline(cleaned_text)

    entities = {
        "Booth Number": None,
        "नाव": None,
        "पतीचे नाव": None,
        "वय": None,
        "लिंग": None
    }

    # Custom regex to capture age (वय)
    age_regex = r"(?:वय|चय)\s*[:;\-]?\s*(\d+)"
    age_match = re.search(age_regex, cleaned_text)
    if age_match:
        entities["वय"] = age_match.group(1)

    # First regex to capture booth number pattern with JVW, SRO, SAO
    booth_number_pattern_1 = r"\b(JVW\S+|Jvw\S+|SRO\S+|SAO\S+|JVWO\S+|JWWOR\S+)\b"
    booth_number_match = re.search(booth_number_pattern_1, text)

    # If the first pattern doesn't match, apply the second one
    if not booth_number_match:
        booth_number_pattern_2 = r'[A-Z]{3,5}[A-Z0-9]{1,2}[ ?]?\d{4,7}'
        booth_number_match = re.search(booth_number_pattern_2, text)

    # Assign the booth number if a match is found
    if booth_number_match:
        entities["Booth Number"] = booth_number_match.group(0)

    current_person_name = ''
    for entity in results:
        if entity['entity'].lower() == 'person':
            current_person_name += entity['word'].replace("##", "").strip()
        elif current_person_name:
            if entities["नाव"] is None:
                entities["नाव"] = current_person_name
            elif entities["पतीचे नाव"] is None:
                entities["पतीचे नाव"] = current_person_name
            current_person_name = ''

        if entity['entity'].lower() == 'measure':
            if entities["वय"] is None and entity['word'].isdigit():
                entities["वय"] = entity['word']

        if entity['entity'].lower() == 'other':
            if entities["लिंग"] is None and entity['word'] in ["महिला", "पुरुष"]:
                entities["लिंग"] = entity['word']

    # Capture any remaining name after the loop
    if current_person_name:
        if entities["नाव"] is None:
            entities["नाव"] = current_person_name
        elif entities["पतीचे नाव"] is None:
            entities["पतीचे नाव"] = current_person_name

    # Remove only the fields 
    if entities["नाव"] is None and entities["पतीचे नाव"] is None:
        entities.pop("नाव", None)
        entities.pop("पतीचे नाव", None)
        entities.pop("वय", None)
        entities.pop("लिंग", None)
        entities.pop("Booth Number", None)
    return entities

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Unable to read image at {image_path}")
        return None
    
    preprocessed_image = preprocess_image(image)
    
    boxes = detect_boxes(preprocessed_image)
    text_data_array = []
    extracted_entities_array = []

    if not boxes:
        logging.warning(f"No text boxes detected in image: {image_path}")
    
    # Process each detected box
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cropped_box = preprocessed_image[y:y+h, x:x+w]
        input_text = extract_text_from_box(cropped_box)
        
        # Apply post-processing to improve accuracy
        processed_text = post_processing(input_text)
        text_data_array.append(processed_text)
        
        logging.info(f"Image: {os.path.basename(image_path)} | Box {idx} - Processed Text:\n{processed_text}\n")
    
        # Extract entities from the post-processed text
        extracted_entities = extract_entities(processed_text)
    
        # Only append if extracted_entities is not empty
        if extracted_entities:
            extracted_entities_array.append(extracted_entities)
        else:
            logging.info(f"Image: {os.path.basename(image_path)} | Box {idx} - No entities extracted.")
    
    # Return the extracted entities (without serial numbers)
    return extracted_entities_array if extracted_entities_array else None

def assign_serial_numbers(entities):
    """
    Assign serial numbers to a list of entities sequentially.
    """
    for i, entity in enumerate(entities, start=1):
        entity['serial_number'] = i
    return entities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the files part
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('files[]')
        if not files or files[0].filename == '':
            flash('No selected files')
            return redirect(request.url)
        
        saved_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(save_path)
                saved_files.append(save_path)
            else:
                flash(f'File {file.filename} is not an allowed image type.')
        
        if not saved_files:
            flash('No valid images uploaded.')
            return redirect(request.url)
        
        all_extracted_data = []
        
        # Process each uploaded image
        for image_path in saved_files:
            logging.info(f"Processing image: {os.path.basename(image_path)}")
            extracted_data = process_image(image_path)
            if extracted_data:
                # Include the image name in each entity for reference
                for entity in extracted_data:
                    entity['image'] = os.path.basename(image_path)
                all_extracted_data.extend(extracted_data)
        
        if all_extracted_data:
            # Assign serial numbers
            all_extracted_data = assign_serial_numbers(all_extracted_data)
            
            # Save to JSON
            output_json_path = os.path.join(OUTPUT_FOLDER, 'extracted_data.json')
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_extracted_data, f, ensure_ascii=False, indent=4)
            
            # Optionally, remove uploaded files after processing
            for file_path in saved_files:
                os.remove(file_path)
            
            return render_template('index.html', tables=[all_extracted_data], titles=['Extracted Entities'], json_available=True)
        else:
            flash('No entities extracted from the uploaded images.')
            return redirect(request.url)
    
    return render_template('index.html', tables=None, titles=None, json_available=False)

@app.route('/download', methods=['GET'])
def download():
    output_json_path = os.path.join(OUTPUT_FOLDER, 'extracted_data.json')
    if os.path.exists(output_json_path):
        return send_file(output_json_path, as_attachment=True)
    else:
        flash('No JSON file available for download.')
        return redirect(url_for('index'))
@app.route('/download_entities_csv')
def download_entities_csv():
    # Path to the JSON file
    output_json_path = os.path.join(OUTPUT_FOLDER, 'extracted_data.json')
    
    if os.path.exists(output_json_path):
        # Load the JSON data using pandas
        try:
            # Read the JSON file into a DataFrame
            df = pd.read_json(output_json_path, encoding='utf-8')
        except ValueError:
            flash('Error reading JSON file. Please check the file format.')
            return redirect(url_for('index'))
        except UnicodeDecodeError:
            # Handle the case where the file is not in UTF-8 encoding
            flash('Error reading JSON file. Please check the file encoding.')
            return redirect(url_for('index'))

        # Create a CSV response with the appropriate mimetype
        output = io.StringIO()  # Create an in-memory buffer
        df.to_csv(output, index=False, encoding='utf-8-sig')  # Save DataFrame to the buffer as CSV
        
        # Move cursor back to the beginning of the stream
        output.seek(0)

        return Response(
            output.getvalue(),  # Get the value of the buffer
            mimetype='text/csv',
            headers={"Content-Disposition": "attachment;filename=extracted_entities.csv"}
        )
    else:
        # If JSON doesn't exist, show an error
        flash('No JSON file available for download.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
