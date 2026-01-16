
import cv2
import pytesseract

# Set tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Tesseract-OCR\tesseract.exe"




def extract_text_from_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve text clarity
    gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]

    # Extract text
    text = pytesseract.image_to_string(gray)

    return text


