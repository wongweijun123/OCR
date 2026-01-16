import streamlit as st
import pytesseract
from PIL import Image, ImageOps
from pdf2image import convert_from_path
import os
import sys
import cv2
import numpy as np
import re
import platform
import requests
from io import BytesIO

# --- CLOUD COMPATIBILITY FIX ---
IS_WINDOWS = platform.system() == "Windows"
if IS_WINDOWS:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    POPPLER_PATH = r"C:\Program Files\poppler-25.12.0\Library\bin"
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    POPPLER_PATH = None 

# --- PROCESSING FUNCTIONS ---
def fix_orientation(pil_image):
    def get_rotation(img_to_check):
        try:
            osd_result = pytesseract.image_to_osd(img_to_check, config='--psm 0 -c min_characters_to_try=3')
            return osd_result
        except Exception:
            return None
    try:
        osd = get_rotation(pil_image)
        if not osd:
            gray = ImageOps.grayscale(pil_image)
            binary_temp = gray.point(lambda x: 0 if x < 128 else 255, '1')
            osd = get_rotation(binary_temp)
        if osd:
            rotation_match = re.search(r'Rotate: (\d+)', osd)
            if rotation_match:
                rotation = int(rotation_match.group(1))
                if rotation in [90, 180, 270]:
                    return pil_image.rotate(-rotation, expand=True)
        return pil_image
    except Exception as e:
        return pil_image

def deskew_image(open_cv_image):
    try:
        gray = open_cv_image
        if len(open_cv_image.shape) == 3:
             gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) < 50: return open_cv_image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45: angle = -(90 + angle)
        else: angle = -angle
        if abs(angle) > 0.5:
            (h, w) = open_cv_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
            bound_w, bound_h = int(h * abs_sin + w * abs_cos), int(h * abs_cos + w * abs_sin)
            M[0, 2] += bound_w / 2 - center[0]
            M[1, 2] += bound_h / 2 - center[1]
            return cv2.warpAffine(open_cv_image, M, (bound_w, bound_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return open_cv_image
    except Exception:
        return open_cv_image

def remove_noise(binary_image):
    try:
        inverted = cv2.bitwise_not(binary_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        return cv2.bitwise_not(opened)
    except Exception:
        return binary_image

def preprocess_for_ocr(pil_image):
    img = np.array(pil_image)
    if len(img.shape) == 3:
        img = img[:, :, ::-1].copy() 
    
    pil_temp = Image.fromarray(img[:, :, ::-1]) if len(img.shape) == 3 else Image.fromarray(img)
    pil_temp = fix_orientation(pil_temp)
    img = np.array(pil_temp)
    if len(img.shape) == 3: img = img[:, :, ::-1].copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = deskew_image(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 10)
    binary = remove_noise(binary)
    padding = 50
    padded = cv2.copyMakeBorder(binary, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return Image.fromarray(padded)

def ocr_processing_logic(file_bytes, is_pdf=False, file_name="file"):
    extracted_text = ""
    try:
        if is_pdf:
            # PDF Processing
            with open("temp_processing.pdf", "wb") as f:
                f.write(file_bytes)
            
            pages = convert_from_path("temp_processing.pdf", dpi=300, poppler_path=POPPLER_PATH)
            for i, page in enumerate(pages):
                processed = preprocess_for_ocr(page)
                page_text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3')
                extracted_text += f"--- Page {i+1} ---\n{page_text}\n"
        else:
            # Image Processing
            image = Image.open(BytesIO(file_bytes))
            processed = preprocess_for_ocr(image)
            extracted_text = pytesseract.image_to_string(processed, config='--oem 3 --psm 3')
            
        return extracted_text
    except Exception as e:
        return f"Error: {str(e)}"

# --- STREAMLIT UI ---
st.set_page_config(page_title="Pro OCR Tool", layout="wide")
st.title("📄 Professional OCR Tool")

# Tabs for Input Method
tab1, tab2 = st.tabs(["📤 Upload File", "🔗 Paste URL"])

final_text = None

# --- TAB 1: UPLOAD ---
with tab1:
    uploaded_file = st.file_uploader("Upload PDF or Image", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None and st.button("Start Extraction (Upload)"):
        with st.spinner("Processing Upload..."):
            is_pdf = uploaded_file.name.lower().endswith(".pdf")
            final_text = ocr_processing_logic(uploaded_file.getbuffer(), is_pdf, uploaded_file.name)

# --- TAB 2: URL ---
with tab2:
    url_input = st.text_input("Enter Image or PDF URL")
    if url_input and st.button("Start Extraction (URL)"):
        with st.spinner("Downloading & Processing..."):
            try:
                response = requests.get(url_input)
                if response.status_code == 200:
                    is_pdf = url_input.lower().endswith(".pdf")
                    final_text = ocr_processing_logic(response.content, is_pdf, "url_file")
                else:
                    st.error("Could not retrieve file from URL.")
            except Exception as e:
                st.error(f"Error fetching URL: {e}")

# --- RESULTS DISPLAY ---
if final_text:
    st.success("Extraction Complete!")
    st.text_area("Results", final_text, height=400)
    st.download_button("Download Text", final_text, file_name="extracted_text.txt")