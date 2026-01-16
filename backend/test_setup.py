import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import cv2
import pytesseract
from PIL import Image
import flask
import joblib

print("All libraries imported successfully!")

# Test NLP
sample = ["This is a fake job posting "]
vec = TfidfVectorizer()
X = vec.fit_transform(sample)
print("NLP Vectorization OK")

# Test OpenCV
img = np.zeros((100,100,3), dtype='uint8')
cv2.imwrite("test.png", img)
print("OpenCV OK")

# Test Pillow
Image.open("test.png")
print("Pillow OK")

# Test Flask object
app = flask.Flask(__name__)
print("Flask OK")

print("✔️ Setup Successful!")
