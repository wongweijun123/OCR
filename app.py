import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError
import os
import cv2
import numpy as np
import re
import platform
from io import BytesIO
import google.generativeai as genai

# --- PAGE CONFIG (Must be first) ---
st.set_page_config(page_title="OCR Tool", layout="wide", page_icon="🔍")

# --- SESSION STATE INITIALIZATION ---
if 'ocr_results' not in st.session_state:
    st.session_state['ocr_results'] = None
if 'ocr_run_id' not in st.session_state:
    st.session_state['ocr_run_id'] = 0

# --- GEMINI CONFIGURATION ---
api_key = None
try:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, Exception):
    pass

# --- CLOUD/LOCAL CONFIGURATION ---
IS_WINDOWS = platform.system() == "Windows"
POPPLER_PATH = None

if IS_WINDOWS:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Expanded list of common Poppler paths
    possible_poppler_paths = [
        r"C:\Program Files\poppler-25.12.0\Library\bin",
        r"C:\Program Files\poppler-24.02.0\Library\bin",
        r"C:\Program Files\poppler-0.68.0\bin",
        os.path.join(os.getcwd(), "poppler", "bin"),
        os.path.join(os.getcwd(), "Poppler", "Library", "bin")
    ]
    
    for path in possible_poppler_paths:
        if os.path.exists(path):
            POPPLER_PATH = path
            break
            
    if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
        st.warning("⚠️ Tesseract not found. Please install Tesseract-OCR.")
else:
    pytesseract.pytesseract.tesseract_cmd = 'tesseract'
    POPPLER_PATH = None 

# --- HELPER FUNCTIONS ---

def reset_state():
    """Clears results when a new file is uploaded to prevent stale data."""
    st.session_state['ocr_results'] = None
    if 'summary_text' in st.session_state:
        del st.session_state['summary_text']

def gemini_summarize(text, key, detail_level="Standard"):
    """
    Summarizes text using Gemini.
    - Uses specific models available to your API key.
    - detail_level controls verbosity.
    - Increased max_output_tokens for longer generation.
    """
    if not key:
        return "⚠️ Error: API Key is missing. Please enter it in the sidebar."

    # Define prompts based on detail level
    prompts = {
        "Concise": "Please provide a very short, concise summary (bullet points) of the following extracted text:",
        "Standard": "Please provide a clear and well-structured summary of the following extracted text:",
        "Detailed": "Please provide a comprehensive, detailed summary of the following extracted text. Include all key sections, data points, and specific details. Do not be brief."
    }
    base_prompt = prompts.get(detail_level, prompts["Standard"])

    try:
        genai.configure(api_key=key)
        
        # Priority list
        model_names = [
            'gemini-2.5-flash',
            'gemini-2.5-pro',
            'gemini-2.0-flash',
            'gemini-2.0-flash-lite',
            'gemini-flash-latest',
            'gemini-1.5-flash'
        ]
        
        errors = []
        
        for m_name in model_names:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content(
                    f"{base_prompt}\n\n{text}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=8192
                    )
                )
                
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    return f"⚠️ Summary blocked by AI safety filters. Reason: {response.prompt_feedback.block_reason}"
                
                return response.text

            except Exception as e:
                errors.append(f"{m_name}: {str(e)}")
                continue
        
        # If all fail, list what IS available
        try:
            available = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available.append(m.name)
            available_str = ", ".join(available) if available else "None"
            debug_info = f"\n\n🔍 Debug Info - Available Models:\n{available_str}"
        except Exception as list_err:
            debug_info = f"\n\n(Could not list available models: {list_err})"

        error_details = "\n".join(errors)
        return f"❌ AI Summary Failed. Details:\n{error_details}{debug_info}"

    except Exception as e:
        return f"❌ Initialization Error: {str(e)}"

def normalize_lighting(image):
    try:
        rgb_planes = cv2.split(image)
        result_planes = []
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(norm_img)
        return cv2.merge(result_planes)
    except:
        return image

def upscale_image(cv_image, scale=2):
    h, w = cv_image.shape[:2]
    return cv2.resize(cv_image, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

def fix_orientation(pil_image):
    try:
        img_copy = pil_image.copy()
        osd = pytesseract.image_to_osd(img_copy, config='--psm 0 -c min_characters_to_try=5')
        rotation_match = re.search(r'Rotate: (\d+)', osd)
        if rotation_match:
            rotation = int(rotation_match.group(1))
            if rotation in [90, 180, 270]:
                return pil_image.rotate(-rotation, expand=True)
    except Exception:
        pass 
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
        if 0.5 < abs(angle) < 45:
            (h, w) = open_cv_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(open_cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return open_cv_image
    except Exception:
        return open_cv_image

def pipeline_processing(pil_image, params):
    img = np.array(pil_image)
    if len(img.shape) == 3: img = img[:, :, ::-1].copy()
    
    if params['remove_shadows']:
        img = normalize_lighting(img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    if params['upscale']:
        gray = upscale_image(gray, scale=2)
    
    if params['deskew']:
        gray = deskew_image(gray)
    
    if params['denoise']:
        gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    
    binary = gray
    if params['threshold_type'] == 'Adaptive Gaussian':
        block_size = params['thresh_block'] if params['thresh_block'] % 2 != 0 else params['thresh_block'] + 1
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10)
    elif params['threshold_type'] == 'Simple Otsu':
        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
    if params['morphology'] == 'Erosion (Thin)':
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.erode(binary, kernel, iterations=1)
    elif params['morphology'] == 'Dilation (Thick)':
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=1)
        
    padding = 20
    padded = cv2.copyMakeBorder(binary, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return Image.fromarray(padded)

def run_ocr(file_bytes, is_pdf, params):
    extracted_data = []
    
    # Use the poppler path determined at startup
    current_poppler_path = params.get('poppler_path')
    
    try:
        if is_pdf:
            with open("temp_ocr_file.pdf", "wb") as f: f.write(file_bytes)
            
            pages = convert_from_path("temp_ocr_file.pdf", dpi=300, poppler_path=current_poppler_path)
            
            for i, page in enumerate(pages):
                if params['auto_rotate']:
                    page = fix_orientation(page)
                
                processed_img = pipeline_processing(page, params)
                
                custom_config = f"--oem 3 --psm {params['psm_mode']}"
                if params['whitelist']:
                    custom_config += f" -c tessedit_char_whitelist={params['whitelist']}"
                
                text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')
                extracted_data.append((processed_img, f"Page {i+1}", text))
                
            if os.path.exists("temp_ocr_file.pdf"):
                os.remove("temp_ocr_file.pdf")
        else:
            image = Image.open(BytesIO(file_bytes))
            if params['auto_rotate']:
                image = fix_orientation(image)
            
            processed_img = pipeline_processing(image, params)
            
            custom_config = f"--oem 3 --psm {params['psm_mode']}"
            if params['whitelist']:
                custom_config += f" -c tessedit_char_whitelist={params['whitelist']}"
            
            text = pytesseract.image_to_string(processed_img, config=custom_config, lang='eng')
            extracted_data.append((processed_img, "Image", text))
            
        return extracted_data, None
    except PDFInfoNotInstalledError:
        return None, (
            "❌ Poppler not installed or not in PATH.\n"
            "• **Windows**: Install 'Release-24.02.0-0' from [github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases), "
            "extract it, and paste the `.../bin` path in the code or add to system PATH.\n"
            "• **Mac**: Run `brew install poppler`\n"
            "• **Linux**: Run `sudo apt-get install poppler-utils`"
        )
    except Exception as e:
        return None, str(e)

# --- UI ---
st.markdown("<style>.stTextArea textarea {font-family: 'Courier New', monospace;}</style>", unsafe_allow_html=True)
st.title("🔍 Pro OCR Tool (Enhanced Accuracy)")

# Sidebar
st.sidebar.header("⚙️ Advanced Settings")

# Removed Windows System Config Section (Poppler Manual Override) as requested

if not api_key:
    with st.sidebar.expander("🔑 AI Configuration", expanded=True):
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Paste key to enable summaries")
        st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com)")

with st.sidebar.expander("Preprocessing", expanded=True):
    param_upscale = st.checkbox("Upscale Image (2x)", value=True)
    param_shadows = st.checkbox("Remove Shadows", value=True)
    param_denoise = st.checkbox("Denoise", value=False)
    param_deskew = st.checkbox("Deskew", value=True)
    param_rotate = st.checkbox("Auto-Rotate (90°)", value=True)

with st.sidebar.expander("Thresholding & Filters", expanded=False):
    param_thresh = st.selectbox("Threshold Method", ["Adaptive Gaussian", "Simple Otsu", "None (Grayscale)"])
    param_block = st.slider("Block Size", 3, 101, 15, step=2)
    param_morph = st.selectbox("Morphology", ["None", "Erosion (Thin)", "Dilation (Thick)"])

with st.sidebar.expander("Engine Configuration", expanded=False):
    param_psm = st.selectbox("PSM Mode", [3, 4, 6, 11], index=2, help="3=Default, 4=Column, 6=Block, 11=Sparse")
    param_whitelist = st.text_input("Whitelist Chars", placeholder="e.g. 0123456789")

params = {
    'upscale': param_upscale, 'remove_shadows': param_shadows, 'denoise': param_denoise,
    'deskew': param_deskew, 'auto_rotate': param_rotate, 'threshold_type': param_thresh,
    'thresh_block': param_block, 'morphology': param_morph, 'psm_mode': param_psm,
    'whitelist': param_whitelist,
    'poppler_path': POPPLER_PATH # Using the auto-detected path directly
}

# Main Content - File Uploader Only (Tabs Removed)
# on_change=reset_state clears the old results instantly when you browse a new file
uploaded_file = st.file_uploader(
    "Upload PDF or Image", 
    type=["png", "jpg", "jpeg", "pdf", "tif", "bmp"],
    on_change=reset_state
)
if uploaded_file and st.button("Extract (Upload)"):
    with st.spinner("Processing..."):
        is_pdf = uploaded_file.name.lower().endswith(".pdf")
        results, error_msg = run_ocr(uploaded_file.getvalue(), is_pdf, params)
        if results:
            st.session_state['ocr_results'] = results
            st.session_state['ocr_run_id'] += 1  # Increment to force UI refresh
            # Clear previous summary
            if "summary_text" in st.session_state: del st.session_state["summary_text"]
        if error_msg: st.error(error_msg)

# Display Results
if st.session_state['ocr_results']:
    results = st.session_state['ocr_results']
    current_run_id = st.session_state['ocr_run_id']
    
    full_text_concat = ""
    for _, title, text in results:
        full_text_concat += f"--- {title} ---\n{text}\n\n"

    page_tabs = st.tabs([title for _, title, _ in results])
    for i, (processed_img, title, text) in enumerate(results):
        with page_tabs[i]:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(processed_img, caption="Processed Image", use_container_width=True)
            with col2:
                # Key now includes run_id to ensure a fresh widget on every new extraction
                st.text_area("Extracted Text", text, height=600, key=f"text_{current_run_id}_{i}")

    st.download_button("Download All Text", full_text_concat, file_name="ocr_result.txt")
    
    st.divider()
    st.subheader("🧠 Gemini AI Summary")
    
    # Added detail level slider
    col_sum1, col_sum2 = st.columns([2, 1])
    with col_sum1:
        detail_level = st.select_slider(
            "Summary Detail Level", 
            options=["Concise", "Standard", "Detailed"],
            value="Standard",
            help="Control how much detail the AI includes in the summary."
        )

    if not api_key:
        st.warning("⚠️ No Gemini API Key provided. Please enter it in the sidebar to enable summarization.")
    else:
        if st.button("Generate Summary", type="primary"):
            if len(full_text_concat.strip()) < 50:
                st.warning("Text is too short to summarize (must be at least 50 characters).")
            else:
                with st.spinner("🤖 Gemini is analyzing..."):
                    # Input limit increased to 100k characters
                    summary = gemini_summarize(full_text_concat[:100000], api_key, detail_level)
                    st.session_state["summary_text"] = summary
    
    if "summary_text" in st.session_state:
        st.info(st.session_state["summary_text"])