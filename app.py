
"""
AgroVision AI
-------------
Smart Agriculture Crop Disease Diagnosis

Streamlit inference app for automated cassava and maize
leaf disease classification.

Required GitHub files:
    app.py
    requirements.txt
    class_names.json
    metadata.json

Hugging Face Hub:
    model.keras

Run locally:
    streamlit run app.py
"""

# ============================================================
# IMPORTS
# ============================================================

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent

CLASS_NAMES_PATH = APP_DIR / "class_names.json"
METADATA_PATH = APP_DIR / "metadata.json"

HF_REPO_ID = "Iroh-Emmanuel/CASSAVA-AND-MAIZE-DISEASES-DETECTION-MODEL"
HF_MODEL_FILENAME = "model.keras"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgroVision AI | Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL PREPROCESSING FUNCTIONS
# ============================================================

PREPROCESS_FUNCS = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
    ====================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(46, 125, 50, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(139, 195, 74, 0.08),
                transparent 25%
            ),
            #f6faf5;
    }

    .main .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #123d27 0%,
                #195c35 55%,
                #103b26 100%
            );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f4fff5 !important;
    }

    .sidebar-brand {
        text-align: center;
        padding: 15px 5px 25px 5px;
    }

    .sidebar-logo {
        width: 75px;
        height: 75px;
        margin: auto;
        border-radius: 50%;
        background: linear-gradient(135deg, #a5d66a, #43a047);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 38px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.20);
    }

    .sidebar-title {
        font-size: 22px;
        font-weight: 800;
        margin-top: 14px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        opacity: 0.75;
        line-height: 1.5;
    }

    .sidebar-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 15px;
        margin: 12px 0;
    }

    .sidebar-card-title {
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .sidebar-stat {
        display: flex;
        justify-content: space-between;
        margin: 7px 0;
        font-size: 13px;
    }

    .status-dot {
        display: inline-block;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background: #7CFC70;
        margin-right: 7px;
        box-shadow: 0 0 8px #7CFC70;
    }


    /* ======================================================
       HERO
    ====================================================== */

    .hero {
        position: relative;
        overflow: hidden;
        border-radius: 28px;
        padding: 45px 48px;
        margin-bottom: 25px;
        background:
            linear-gradient(
                120deg,
                rgba(10, 62, 35, 0.98),
                rgba(28, 105, 54, 0.96)
            );
        box-shadow: 0 20px 50px rgba(25, 85, 45, 0.18);
        color: white;
    }

    .hero::before {
        content: "🌾";
        position: absolute;
        right: 50px;
        top: -20px;
        font-size: 150px;
        opacity: 0.10;
        transform: rotate(-8deg);
    }

    .hero::after {
        content: "🌿";
        position: absolute;
        right: 190px;
        bottom: -45px;
        font-size: 130px;
        opacity: 0.08;
        transform: rotate(20deg);
    }

    .hero-badge {
        display: inline-block;
        background: rgba(255,255,255,0.13);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 30px;
        padding: 7px 14px;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 46px;
        line-height: 1.08;
        font-weight: 850;
        margin: 0;
        max-width: 720px;
        letter-spacing: -1.5px;
    }

    .hero-title span {
        color: #b8e986;
    }

    .hero-text {
        font-size: 17px;
        line-height: 1.65;
        max-width: 700px;
        margin-top: 17px;
        color: rgba(255,255,255,0.82);
    }

    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 9px;
        margin-top: 22px;
    }

    .hero-pill {
        background: rgba(255,255,255,0.10);
        border: 1px solid rgba(255,255,255,0.13);
        padding: 8px 13px;
        border-radius: 20px;
        font-size: 12px;
    }


    /* ======================================================
       SECTION HEADERS
    ====================================================== */

    .section-label {
        color: #338447;
        font-size: 12px;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 5px;
    }

    .section-title {
        color: #173d27;
        font-size: 27px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .section-description {
        color: #6b7b70;
        font-size: 14px;
        margin-bottom: 20px;
    }


    /* ======================================================
       CROP CARDS
    ====================================================== */

    .crop-card {
        border-radius: 18px;
        padding: 20px;
        background: white;
        border: 1px solid #e1ebe2;
        box-shadow: 0 8px 25px rgba(24,70,35,0.06);
        height: 100%;
    }

    .crop-icon {
        font-size: 38px;
        margin-bottom: 10px;
    }

    .crop-name {
        font-weight: 800;
        color: #1c4930;
        font-size: 18px;
    }

    .crop-description {
        color: #718076;
        font-size: 13px;
        line-height: 1.55;
        margin-top: 6px;
    }


    /* ======================================================
       UPLOAD AREA
    ====================================================== */

    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 22px;
        padding: 15px;
        border: 2px dashed #b9d9be;
        box-shadow: 0 10px 35px rgba(30,80,40,0.07);
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #43a047;
        background: #fbfffb;
    }


    /* ======================================================
       RESULT CARDS
    ====================================================== */

    .result-card {
        background: white;
        border-radius: 22px;
        padding: 26px;
        border: 1px solid #dfeae1;
        box-shadow: 0 12px 35px rgba(30,80,40,0.09);
        margin-top: 10px;
    }

    .result-label {
        font-size: 11px;
        color: #6b7d70;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 800;
    }

    .result-value {
        color: #173d27;
        font-size: 30px;
        font-weight: 850;
        margin-top: 3px;
    }

    .healthy-result {
        border-left: 6px solid #43a047;
    }

    .disease-result {
        border-left: 6px solid #e67e22;
    }


    /* ======================================================
       METRIC CARDS
    ====================================================== */

    .metric-card {
        background: white;
        border-radius: 18px;
        padding: 18px;
        border: 1px solid #e1ebe2;
        box-shadow: 0 8px 25px rgba(25,80,35,0.05);
        height: 100%;
    }

    .metric-icon {
        font-size: 25px;
    }

    .metric-title {
        color: #718076;
        font-size: 12px;
        margin-top: 8px;
    }

    .metric-value {
        color: #1c4930;
        font-size: 21px;
        font-weight: 800;
        margin-top: 2px;
    }


    /* ======================================================
       INFO BOX
    ====================================================== */

    .info-box {
        background: #edf7ee;
        border: 1px solid #d2e9d5;
        border-radius: 16px;
        padding: 16px 18px;
        color: #365c40;
        font-size: 13px;
        line-height: 1.6;
        margin-top: 18px;
    }


    /* ======================================================
       FOOTER
    ====================================================== */

    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #dfe8df;
        color: #7b8a7e;
        font-size: 12px;
    }

    .footer strong {
        color: #367447;
    }


    /* ======================================================
       STREAMLIT BUTTON
    ====================================================== */

    .stButton > button {
        border-radius: 12px;
        border: none;
        background: #267a40;
        color: white;
        font-weight: 700;
    }


    /* ======================================================
       MOBILE RESPONSIVENESS
    ====================================================== */

    @media (max-width: 768px) {

        .hero {
            padding: 30px 25px;
            border-radius: 20px;
        }

        .hero-title {
            font-size: 34px;
        }

        .hero-text {
            font-size: 15px;
        }

        .hero::before {
            font-size: 90px;
            right: 10px;
        }

        .hero::after {
            font-size: 80px;
            right: 90px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🌿</div>
            <div class="sidebar-title">AgroVision AI</div>
            <div class="sidebar-subtitle">
                Intelligent crop monitoring<br>
                powered by deep learning
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">⚡ System Status</div>

            <div class="sidebar-stat">
                <span>AI Engine</span>
                <span>
                    <span class="status-dot"></span>Ready
                </span>
            </div>

            <div class="sidebar-stat">
                <span>Crop Coverage</span>
                <strong>2 Crops</strong>
            </div>

            <div class="sidebar-stat">
                <span>Analysis</span>
                <strong>Automated</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🌾 Supported Crops</div>

            <div class="sidebar-stat">
                <span>🌿 Cassava</span>
                <span>✓</span>
            </div>

            <div class="sidebar-stat">
                <span>🌽 Maize</span>
                <span>✓</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-card-title">🤖 How It Works</div>

            <div style="
                font-size:13px;
                line-height:1.65;
                opacity:0.85;
            ">
                1. Upload a crop leaf image.<br>
                2. AI preprocesses the image.<br>
                3. Deep learning model analyzes the leaf.<br>
                4. Disease class is identified.<br>
                5. Confidence scores are displayed.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            margin-top:35px;
            font-size:11px;
            opacity:0.55;
        ">
            SMART AGRICULTURE<br>
            AI • COMPUTER VISION • FARMING
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HERO SECTION
# ============================================================

hero_html = """
<div class="hero">

    <div class="hero-badge">
        ✦ AI-POWERED SMART AGRICULTURE PLATFORM
    </div>

    <h1 class="hero-title">
        Protecting Crops with
        <span>Intelligent Vision</span>
    </h1>

    <div class="hero-text">
        Upload a cassava or maize leaf and let our automated
        deep-learning system analyze it for potential diseases.
        Designed to support faster crop monitoring, early detection,
        and smarter agricultural decisions.
    </div>

    <div class="hero-pills">
        <div class="hero-pill">🌿 Cassava Detection</div>
        <div class="hero-pill">🌽 Maize Detection</div>
        <div class="hero-pill">🧠 Deep Learning</div>
        <div class="hero-pill">⚡ Automated Analysis</div>
    </div>

</div>
"""

# Streamlit versions with st.html use the dedicated HTML renderer.
# Older versions fall back to st.markdown.
if hasattr(st, "html"):
    st.html(hero_html)
else:
    st.markdown(hero_html, unsafe_allow_html=True)


# ============================================================
# CROP OVERVIEW
# ============================================================

st.markdown(
    """
    <div class="section-label">SMART FARMING</div>

    <div class="section-title">
        What can AgroVision analyze?
    </div>

    <div class="section-description">
        The system is designed to identify diseases affecting important
        agricultural crops using leaf-image classification.
    </div>
    """,
    unsafe_allow_html=True,
)

crop_col1, crop_col2 = st.columns(2)

with crop_col1:

    st.markdown(
        """
        <div class="crop-card">

            <div class="crop-icon">🌿</div>

            <div class="crop-name">
                Cassava
            </div>

            <div class="crop-description">
                Analyze cassava leaves for visible disease patterns and
                automatically classify the detected condition using the
                trained computer-vision model.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with crop_col2:

    st.markdown(
        """
        <div class="crop-card">

            <div class="crop-icon">🌽</div>

            <div class="crop-name">
                Maize
            </div>

            <div class="crop-description">
                Screen maize leaf images for disease symptoms and receive
                an automated classification together with model confidence.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# LOAD MODEL AND METADATA
# ============================================================

@st.cache_resource(show_spinner="Loading AgroVision AI model...")
def load_everything():

    # --------------------------------------------------------
    # Check required local files
    # --------------------------------------------------------

    missing_files = []

    if not CLASS_NAMES_PATH.exists():
        missing_files.append("class_names.json")

    if not METADATA_PATH.exists():
        missing_files.append("metadata.json")

    if missing_files:
        raise FileNotFoundError(
            "The following required files are missing: "
            + ", ".join(missing_files)
            + ". Make sure they are in the same GitHub folder as app.py."
        )

    # --------------------------------------------------------
    # Read class names
    # --------------------------------------------------------

    try:

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            class_names = json.load(f)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"class_names.json contains invalid JSON: {e}"
        )


    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    try:

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(f)

    except json.JSONDecodeError as e:

        raise ValueError(
            f"metadata.json contains invalid JSON: {e}"
        )


    # --------------------------------------------------------
    # Validate class names
    # --------------------------------------------------------

    if not isinstance(class_names, list) or len(class_names) == 0:

        raise ValueError(
            "class_names.json must contain a non-empty list of class names."
        )


    # --------------------------------------------------------
    # Get model architecture
    # --------------------------------------------------------

    best_model_name = metadata.get("best_model")

    if not best_model_name:

        raise ValueError(
            "metadata.json does not contain the 'best_model' field."
        )


    if best_model_name not in PREPROCESS_FUNCS:

        raise ValueError(
            f"metadata.json says best_model='{best_model_name}', "
            f"but the app supports only: "
            f"{list(PREPROCESS_FUNCS.keys())}"
        )


    # --------------------------------------------------------
    # Get image size
    # --------------------------------------------------------

    image_size = metadata.get("image_size")

    if not image_size:

        raise ValueError(
            "metadata.json does not contain the 'image_size' field."
        )


    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:

        raise ValueError(
            "metadata.json 'image_size' must contain two values, "
            "for example [224, 224]."
        )


    img_size = (
        int(image_size[0]),
        int(image_size[1]),
    )


    # --------------------------------------------------------
    # Select preprocessing function
    # --------------------------------------------------------

    preprocess_fn = PREPROCESS_FUNCS[best_model_name]


    # --------------------------------------------------------
    # Download model from Hugging Face
    # --------------------------------------------------------

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_MODEL_FILENAME,
    )


    # --------------------------------------------------------
    # Load Keras model
    # --------------------------------------------------------

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )


    return (
        model,
        class_names,
        preprocess_fn,
        img_size,
        best_model_name,
    )


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        model,
        class_names,
        preprocess_fn,
        img_size,
        best_model_name,

    ) = load_everything()


except Exception as e:

    st.error(
        "⚠️ AgroVision AI could not load the model."
    )

    st.warning(
        "Please check that your GitHub files and Hugging Face "
        "model repository are correctly configured."
    )

    with st.expander("Technical details"):

        st.code(
            str(e),
            language="text",
        )

    st.stop()


# ============================================================
# MODEL INFORMATION
# ============================================================

m1, m2, m3 = st.columns(3)


with m1:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-icon">
                🧠
            </div>

            <div class="metric-title">
                AI Architecture
            </div>

            <div class="metric-value">
                {best_model_name}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with m2:

    st.markdown(
        """
        <div class="metric-card">

            <div class="metric-icon">
                🌱
            </div>

            <div class="metric-title">
                Crop Categories
            </div>

            <div class="metric-value">
                Cassava + Maize
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with m3:

    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-icon">
                🔬
            </div>

            <div class="metric-title">
                Classification Classes
            </div>

            <div class="metric-value">
                {len(class_names)}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")
st.write("")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(
    model,
    class_names,
    preprocess_fn,
    img_size,
    pil_image,
):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = (
        pil_image
        .convert("RGB")
        .resize(
            img_size,
            Image.Resampling.LANCZOS,
        )
    )


    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    x = np.expand_dims(
        image_array,
        axis=0,
    )


    # --------------------------------------------------------
    # Apply model-specific preprocessing
    # --------------------------------------------------------

    x = preprocess_fn(x)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions = model.predict(
        x,
        verbose=0,
    )


    if predictions is None or len(predictions) == 0:

        raise ValueError(
            "The AI model returned an empty prediction."
        )


    preds = np.asarray(
        predictions[0],
        dtype=np.float32,
    )


    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if len(preds) != len(class_names):

        raise ValueError(
            "The model returned "
            f"{len(preds)} predictions, but class_names.json "
            f"contains {len(class_names)} classes."
        )


    # --------------------------------------------------------
    # Find highest probability
    # --------------------------------------------------------

    top_idx = int(
        np.argmax(preds)
    )


    confidence = float(
        preds[top_idx]
    )


    return (
        class_names[top_idx],
        confidence,
        preds,
    )


# ============================================================
# UPLOAD SECTION
# ============================================================

st.markdown(
    """
    <div class="section-label">
        AUTOMATED LEAF SCANNING
    </div>

    <div class="section-title">
        Upload a crop leaf image
    </div>

    <div class="section-description">
        For the best results, upload a clear image where the leaf is
        visible and reasonably well lit.
    </div>
    """,
    unsafe_allow_html=True,
)


uploaded = st.file_uploader(
    "Drag and drop a cassava or maize leaf image here",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    help="Supported formats: JPG, JPEG and PNG.",
)


# ============================================================
# ANALYSIS
# ============================================================

if uploaded is not None:

    try:

        image = Image.open(uploaded)

    except Exception:

        st.error(
            "The uploaded file could not be opened as an image."
        )

        st.stop()


    image_col, result_col = st.columns(
        [1, 1.15],
        gap="large",
    )


    # ========================================================
    # INPUT IMAGE
    # ========================================================

    with image_col:

        st.markdown(
            """
            <div class="section-label">
                INPUT IMAGE
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.image(
            image,
            caption="Leaf image submitted for AI analysis",
            use_container_width=True,
        )


        st.markdown(
            f"""
            <div class="info-box">

                <strong>📷 Image received</strong><br>

                Resolution:
                {image.width} × {image.height}px
                <br>

                Format:
                {image.format or "Image"}

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # RUN PREDICTION
    # ========================================================

    try:

        (
            label,
            confidence,
            all_preds,

        ) = predict(
            model,
            class_names,
            preprocess_fn,
            img_size,
            image,
        )

    except Exception as e:

        st.error(
            "⚠️ The AI model could not analyze this image."
        )

        with st.expander("Technical details"):

            st.code(
                str(e),
                language="text",
            )

        st.stop()


    # ========================================================
    # PROCESS LABEL
    # ========================================================

    if "__" in label:

        crop, condition = label.split(
            "__",
            1,
        )

    else:

        crop = "Unknown"
        condition = label


    clean_crop = (
        crop
        .replace("_", " ")
        .title()
    )


    clean_condition = (
        condition
        .replace("_", " ")
        .title()
    )


    confidence_percent = confidence * 100


    # ========================================================
    # HEALTH STATUS
    # ========================================================

    is_healthy = (
        "healthy" in condition.lower()
        or "normal" in condition.lower()
    )


    if is_healthy:

        result_class = "healthy-result"
        result_icon = "✅"

        result_message = (
            "The AI model did not detect a disease pattern."
        )

    else:

        result_class = "disease-result"
        result_icon = "⚠️"

        result_message = (
            "The AI model detected a disease-associated pattern."
        )


    # ========================================================
    # RESULT CARD
    # ========================================================

    with result_col:

        st.markdown(
            """
            <div class="section-label">
                AI DIAGNOSIS
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.markdown(
            f"""
            <div class="result-card {result_class}">

                <div style="
                    font-size:42px;
                    margin-bottom:8px;
                ">
                    {result_icon}
                </div>

                <div class="result-label">
                    Detected Crop
                </div>

                <div class="result-value">
                    {clean_crop}
                </div>

                <div style="
                    height:1px;
                    background:#e7eee8;
                    margin:17px 0;
                "></div>

                <div class="result-label">
                    Automated Diagnosis
                </div>

                <div class="result-value">
                    {clean_condition}
                </div>

                <div class="info-box">
                    {result_message}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ====================================================
        # CONFIDENCE CARD
        # ====================================================

        st.write("")


        st.markdown(
            f"""
            <div class="metric-card">

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">

                    <div>

                        <div class="metric-title">
                            MODEL CONFIDENCE
                        </div>

                        <div class="metric-value">
                            {confidence_percent:.1f}%
                        </div>

                    </div>

                    <div style="
                        font-size:30px;
                    ">
                        🎯
                    </div>

                </div>


                <div style="
                    background:#e7eee8;
                    height:10px;
                    border-radius:20px;
                    margin-top:15px;
                    overflow:hidden;
                ">

                    <div style="
                        width:{min(max(confidence_percent, 0), 100):.1f}%;
                        height:100%;
                        border-radius:20px;
                        background:
                            linear-gradient(
                                90deg,
                                #81c784,
                                #2e7d32
                            );
                    "></div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ========================================================
    # ALL CLASS PROBABILITIES
    # ========================================================

    st.write("")
    st.write("")


    st.markdown(
        """
        <div class="section-label">
            MODEL INSIGHTS
        </div>

        <div class="section-title">
            Classification probability
        </div>

        <div class="section-description">
            Probability assigned to each disease class by the AI model.
        </div>
        """,
        unsafe_allow_html=True,
    )


    probability_col1, probability_col2 = st.columns(
        [1.2, 1],
        gap="large",
    )


    sorted_predictions = sorted(
        zip(
            class_names,
            all_preds,
        ),
        key=lambda p: -float(p[1]),
    )


    # ========================================================
    # PROBABILITY BARS
    # ========================================================

    with probability_col1:

        for name, prob in sorted_predictions:

            display_name = (
                name
                .replace("__", " → ")
                .replace("_", " ")
                .title()
            )


            percentage = float(prob) * 100


            st.markdown(
                f"""
                <div style="
                    margin:13px 0;
                ">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:5px;
                        font-size:13px;
                    ">

                        <span style="
                            color:#36543e;
                            font-weight:600;
                        ">
                            {display_name}
                        </span>

                        <strong style="
                            color:#267a40;
                        ">
                            {percentage:.2f}%
                        </strong>

                    </div>


                    <div style="
                        height:8px;
                        background:#e6eee7;
                        border-radius:20px;
                        overflow:hidden;
                    ">

                        <div style="
                            width:{min(max(percentage, 0), 100):.2f}%;
                            height:100%;
                            border-radius:20px;
                            background:
                                linear-gradient(
                                    90deg,
                                    #9ccc65,
                                    #2e7d32
                                );
                        "></div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


    # ========================================================
    # ANALYSIS SUMMARY
    # ========================================================

    with probability_col2:

        crop_icon = (
            "🌿"
            if crop.lower() == "cassava"
            else "🌽"
        )


        st.markdown(
            f"""
            <div class="crop-card">

                <div style="
                    font-size:42px;
                ">
                    {crop_icon}
                </div>


                <div class="crop-name">
                    Automated Farm Intelligence
                </div>


                <div class="crop-description">

                    AgroVision uses computer vision and a trained
                    deep-learning model to analyze leaf characteristics
                    and estimate the most likely crop health condition.

                </div>


                <div style="
                    height:1px;
                    background:#e4ece5;
                    margin:18px 0;
                "></div>


                <div style="
                    font-size:12px;
                    color:#718076;
                    line-height:1.7;
                ">

                    <strong style="
                        color:#356843;
                    ">
                        Current analysis
                    </strong>

                    <br>

                    Crop:
                    {clean_crop}

                    <br>

                    Diagnosis:
                    {clean_condition}

                    <br>

                    Confidence:
                    {confidence_percent:.1f}%

                    <br>

                    AI model:
                    {best_model_name}

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# USER GUIDANCE
# ============================================================

else:

    st.markdown(
        """
        <div class="info-box">

            <strong>
                🌱 Ready for automated crop analysis?
            </strong>

            <br>

            Upload a clear cassava or maize leaf image above.
            The AI system will automatically process the image,
            identify the most likely disease class, and display
            the model's confidence score.

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <strong>
            🌿 AgroVision AI
        </strong>

        &nbsp;•&nbsp;

        Intelligent Crop Disease Diagnosis

        &nbsp;•&nbsp;

        Smart Agriculture & Computer Vision

        <br>

        Built for automated cassava and maize crop monitoring.

    </div>
    """,
    unsafe_allow_html=True,
)
