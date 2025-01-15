Image Entities Extraction with Flask
Overview
This project is a Flask web application that allows users to upload multiple images from a folder, extract specific entities using Optical Character Recognition (OCR), and display the results in a table format. Users can also download the extracted data in JSON format.
Features
•	Upload multiple images for processing.
•	Extract entities such as Booth Number, Name, Age, and Gender from the images.
•	Display extracted data in a user-friendly table.
•	Download the extracted data as a JSON file.
•	Loader to indicate processing status.
Requirements
•	Python 3.x
•	Flask
•	Tesseract OCR
•	Pillow
•	pytesseract
Installation
1.	Clone the repository:
bash
Copy code
git clone <repository-url>
cd <repository-folder>
2.	Create a virtual environment (optional but recommended):
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3.	Install the required packages:
bash
Copy code
pip install Flask pytesseract Pillow
4.	Install Tesseract OCR:
o	Download and install Tesseract OCR from Tesseract OCR.
o	Make sure to add the Tesseract executable path to your system's environment variables.
5.	Create an uploads folder: Create a folder named uploads in the same directory as app.py to store uploaded images.
Usage
1.	Run the application:
bash
Copy code
python app.py
2.	Access the application: Open your web browser and go to http://127.0.0.1:5000/.
3.	Upload Images:
o	Select multiple images from your folder and click the submit button to extract entities.
4.	Download Output:
o	Click the download button to get the extracted data in JSON format.
Customization
•	Modify the extract_entities function in app.py to customize the entity extraction logic based on your specific needs.
•	Adjust regex patterns according to the format of the text you expect from the images.


