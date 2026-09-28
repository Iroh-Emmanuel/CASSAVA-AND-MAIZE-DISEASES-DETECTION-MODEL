

"""AgroVision AI - Smart Agriculture Crop Disease Diagnosis."""
import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

APP_DIR = Path(__file__).resolve().parent
CLASS_NAMES_PATH = APP_DIR / "class_names.json"
METADATA_PATH = APP_DIR / "metadata.json"
LOCAL_MODEL_PATH = APP_DIR / "model.keras"

# Exact Hugging Face repository containing model.keras
HF_REPO_ID = "Iroh-Emmanuel/CASSAVA-AND-MAIZE-DISEASES-DETECTION-MODEL"
HF_MODEL_FILENAME = "model.keras"

PREPROCESS_FUNCS = {
    "ResNet50": tf.keras.applications.resnet50.preprocess_input,
    "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
    "VGG16": tf.keras.applications.vgg16.preprocess_input,
}

st.set_page_config(
    page_title="AgroVision AI | Crop Disease Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp{background:radial-gradient(circle at 10% 10%,rgba(46,125,50,.08),transparent 28%),radial-gradient(circle at 90% 20%,rgba(139,195,74,.08),transparent 25%),#f6faf5}
.main .block-container{max-width:1250px;padding-top:2rem;padding-bottom:3rem}
#MainMenu,footer{visibility:hidden} header{background:transparent!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#123d27 0%,#195c35 55%,#103b26 100%);border-right:1px solid rgba(255,255,255,.08)}
section[data-testid="stSidebar"] *{color:#f4fff5!important}
.sidebar-brand{text-align:center;padding:15px 5px 25px}.sidebar-logo{width:75px;height:75px;margin:auto;border-radius:50%;background:linear-gradient(135deg,#a5d66a,#43a047);display:flex;align-items:center;justify-content:center;font-size:38px;box-shadow:0 10px 30px rgba(0,0,0,.2)}
.sidebar-title{font-size:22px;font-weight:800;margin-top:14px}.sidebar-subtitle{font-size:13px;opacity:.75;line-height:1.5}.sidebar-card{background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.1);border-radius:14px;padding:15px;margin:12px 0}.sidebar-card-title{font-weight:700;font-size:14px;margin-bottom:8px}.sidebar-stat{display:flex;justify-content:space-between;margin:7px 0;font-size:13px}.status-dot{display:inline-block;width:9px;height:9px;border-radius:50%;background:#7CFC70;margin-right:7px;box-shadow:0 0 8px #7CFC70}
.hero{position:relative;overflow:hidden;border-radius:28px;padding:45px 48px;margin-bottom:25px;background:linear-gradient(120deg,rgba(10,62,35,.98),rgba(28,105,54,.96));box-shadow:0 20px 50px rgba(25,85,45,.18);color:#fff}.hero-title{font-size:46px;line-height:1.08;font-weight:850;margin:0;max-width:720px}.hero-title span{color:#b8e986}.hero-badge{display:inline-block;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.16);border-radius:30px;padding:7px 14px;font-size:12px;font-weight:700;margin-bottom:15px}.hero-text{font-size:17px;line-height:1.65;max-width:700px;margin-top:17px;color:rgba(255,255,255,.82)}.hero-pills{display:flex;flex-wrap:wrap;gap:9px;margin-top:22px}.hero-pill{background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.13);padding:8px 13px;border-radius:20px;font-size:12px}
.section-label{color:#338447;font-size:12px;text-transform:uppercase;font-weight:800;letter-spacing:1.5px;margin-bottom:5px}.section-title{color:#173d27;font-size:27px;font-weight:800;margin-bottom:5px}.section-description{color:#6b7b70;font-size:14px;margin-bottom:20px}.crop-card,.metric-card{border-radius:18px;padding:20px;background:#fff;border:1px solid #e1ebe2;box-shadow:0 8px 25px rgba(24,70,35,.06);height:100%}.crop-icon,.metric-icon{font-size:38px}.crop-name{font-weight:800;color:#1c4930;font-size:18px}.crop-description{color:#718076;font-size:13px;line-height:1.55;margin-top:6px}.metric-title{color:#718076;font-size:12px;margin-top:8px}.metric-value{color:#1c4930;font-size:21px;font-weight:800;margin-top:2px}
[data-testid="stFileUploader"]{background:#fff;border-radius:22px;padding:15px;border:2px dashed #b9d9be;box-shadow:0 10px 35px rgba(30,80,40,.07)}
.result-card{background:#fff;border-radius:22px;padding:26px;border:1px solid #dfeae1;box-shadow:0 12px 35px rgba(30,80,40,.09);margin-top:10px}.result-label{font-size:11px;color:#6b7d70;text-transform:uppercase;letter-spacing:1.3px;font-weight:800}.result-value{color:#173d27;font-size:30px;font-weight:850;margin-top:3px}.healthy-result{border-left:6px solid #43a047}.disease-result{border-left:6px solid #e67e22}.info-box{background:#edf7ee;border:1px solid #d2e9d5;border-radius:16px;padding:16px 18px;color:#365c40;font-size:13px;line-height:1.6;margin-top:18px}.footer{text-align:center;margin-top:50px;padding-top:20px;border-top:1px solid #dfe8df;color:#7b8a7e;font-size:12px}.footer strong{color:#367447}.stButton>button{border-radius:12px;border:none;background:#267a40;color:white;font-weight:700}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""<div class="sidebar-brand"><div class="sidebar-logo">🌿</div><div class="sidebar-title">AgroVision AI</div><div class="sidebar-subtitle">Intelligent crop monitoring<br>powered by deep learning</div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-card"><div class="sidebar-card-title">⚡ System Status</div><div class="sidebar-stat"><span>AI Engine</span><span><span class="status-dot"></span>Ready</span></div><div class="sidebar-stat"><span>Crop Coverage</span><strong>2 Crops</strong></div><div class="sidebar-stat"><span>Analysis</span><strong>Automated</strong></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-card"><div class="sidebar-card-title">🌾 Supported Crops</div><div class="sidebar-stat"><span>🌿 Cassava</span><span>✓</span></div><div class="sidebar-stat"><span>🌽 Maize</span><span>✓</span></div></div>""", unsafe_allow_html=True)
    st.markdown("""<div class="sidebar-card"><div class="sidebar-card-title">🤖 How It Works</div><div style="font-size:13px;line-height:1.65;opacity:.85">1. Upload a crop leaf image.<br>2. AI preprocesses the image.<br>3. Deep learning model analyzes the leaf.<br>4. Disease class is identified.<br>5. Confidence scores are displayed.</div></div>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="hero-badge">✦ AI-POWERED SMART AGRICULTURE PLATFORM</div><h1 class="hero-title">Protecting Crops with <span>Intelligent Vision</span></h1><div class="hero-text">Upload a cassava or maize leaf and let our automated deep-learning system analyze it for potential diseases. Designed to support faster crop monitoring, early detection, and smarter agricultural decisions.</div><div class="hero-pills"><div class="hero-pill">🌿 Cassava Detection</div><div class="hero-pill">🌽 Maize Detection</div><div class="hero-pill">🧠 Deep Learning</div><div class="hero-pill">⚡ Automated Analysis</div></div></div>""", unsafe_allow_html=True)

st.markdown("""<div class="section-label">SMART FARMING</div><div class="section-title">What can AgroVision analyze?</div><div class="section-description">The system is designed to identify diseases affecting important agricultural crops using leaf-image classification.</div>""", unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1: st.markdown("""<div class="crop-card"><div class="crop-icon">🌿</div><div class="crop-name">Cassava</div><div class="crop-description">Analyze cassava leaves for visible disease patterns and automatically classify the detected condition using the trained computer-vision model.</div></div>""", unsafe_allow_html=True)
with c2: st.markdown("""<div class="crop-card"><div class="crop-icon">🌽</div><div class="crop-name">Maize</div><div class="crop-description">Screen maize leaf images for disease symptoms and receive an automated classification together with model confidence.</div></div>""", unsafe_allow_html=True)
st.write("")


def get_hf_token():
    """Read Hugging Face token from Streamlit Secrets or environment."""
    token = None
    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        pass
    return token or os.getenv("HF_TOKEN")


@st.cache_resource
def load_everything():
    missing = [p.name for p in (CLASS_NAMES_PATH, METADATA_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s): {', '.join(missing)}. Upload/commit them beside app.py.")

    try:
        class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e

    # metadata.json is the authoritative source for class ordering.
    class_names = metadata.get("class_names", class_names)
    best_model = metadata.get("best_model", "VGG16")
    if best_model not in PREPROCESS_FUNCS:
        raise ValueError(f"Unsupported model '{best_model}'. Expected one of {list(PREPROCESS_FUNCS)}.")

    expected_classes = int(metadata.get("num_classes", len(class_names)))
    if len(class_names) != expected_classes:
        raise ValueError(f"Class-count mismatch: metadata expects {expected_classes}, found {len(class_names)}.")

    image_size = metadata.get("image_size", [224,224])
    if not isinstance(image_size,(list,tuple)) or len(image_size)!=2:
        raise ValueError("metadata.json image_size must be [height, width].")
    img_size=(int(image_size[0]),int(image_size[1]))

    required = metadata.get("preprocessing_required")
    expected_required = {
        "VGG16":"tf.keras.applications.vgg16.preprocess_input",
        "ResNet50":"tf.keras.applications.resnet50.preprocess_input",
        "MobileNetV2":"tf.keras.applications.mobilenet_v2.preprocess_input",
    }[best_model]
    if required and required != expected_required:
        raise ValueError(f"Preprocessing mismatch: metadata specifies '{required}' for {best_model}; expected '{expected_required}'.")

    # Local model takes precedence, which makes local testing possible.
    if LOCAL_MODEL_PATH.exists():
        model_path=str(LOCAL_MODEL_PATH)
        source="Local model.keras"
    else:
        token=get_hf_token()
        if not token:
            raise RuntimeError(
                "Hugging Face token is missing. If the repository is private, add HF_TOKEN to Streamlit Secrets. "
                f"Repository: {HF_REPO_ID}"
            )
        try:
            model_path=hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_MODEL_FILENAME,
                token=token,
            )
            source=f"Hugging Face: {HF_REPO_ID}"
        except HfHubHTTPError as e:
            raise RuntimeError(
                f"Hugging Face could not access '{HF_MODEL_FILENAME}' in '{HF_REPO_ID}'. "
                "Verify the repository ID, file name/path, repository visibility, and that HF_TOKEN has read access. "
                f"Original error: {e}"
            ) from e

    try:
        model=tf.keras.models.load_model(model_path)
    except Exception as e:
        raise RuntimeError(f"model.keras was found but TensorFlow/Keras could not load it: {e}") from e

    # Validate output count where available.
    try:
        output_units=int(model.output_shape[-1])
        if output_units != len(class_names):
            raise ValueError(f"Model output mismatch: model produces {output_units} outputs, but {len(class_names)} class names are configured.")
    except (AttributeError,TypeError):
        pass

    return model,class_names,PREPROCESS_FUNCS[best_model],img_size,best_model,source,metadata


def predict(model,class_names,preprocess_fn,img_size,pil_image):
    img=pil_image.convert("RGB").resize(img_size)
    x=np.expand_dims(np.asarray(img,dtype=np.float32),axis=0)
    x=preprocess_fn(x)
    preds=np.asarray(model.predict(x,verbose=0)[0],dtype=np.float32).reshape(-1)
    if len(preds)!=len(class_names):
        raise ValueError(f"Model returned {len(preds)} predictions but {len(class_names)} classes are configured.")
    # Softmax only if the model output is not already a probability vector.
    if np.any(preds < 0) or not np.isclose(float(np.sum(preds)),1.0,atol=1e-3):
        e=np.exp(preds-np.max(preds)); preds=e/e.sum()
    idx=int(np.argmax(preds))
    return class_names[idx],float(preds[idx]),preds


try:
    model,class_names,preprocess_fn,img_size,best_model_name,model_source,metadata=load_everything()
except Exception as e:
    st.error("❌ Could not load the AI model.")
    st.code(str(e),language="text")
    st.info("For a private Hugging Face repository, configure HF_TOKEN in Streamlit Secrets. Do not hard-code the token in app.py.")
    st.stop()

m1,m2,m3=st.columns(3)
with m1: st.markdown(f'<div class="metric-card"><div class="metric-icon">🧠</div><div class="metric-title">AI Architecture</div><div class="metric-value">{best_model_name}</div></div>',unsafe_allow_html=True)
with m2: st.markdown('<div class="metric-card"><div class="metric-icon">🌱</div><div class="metric-title">Crop Categories</div><div class="metric-value">Cassava + Maize</div></div>',unsafe_allow_html=True)
with m3: st.markdown(f'<div class="metric-card"><div class="metric-icon">🔬</div><div class="metric-title">Classification Classes</div><div class="metric-value">{len(class_names)}</div></div>',unsafe_allow_html=True)
st.write("")

st.markdown("""<div class="section-label">AUTOMATED LEAF SCANNING</div><div class="section-title">Upload a crop leaf image</div><div class="section-description">For the best results, upload a clear image where the leaf is visible and reasonably well lit.</div>""",unsafe_allow_html=True)
uploaded=st.file_uploader("Drag and drop a cassava or maize leaf image here",type=["jpg","jpeg","png"],help="Supported formats: JPG, JPEG and PNG.")

if uploaded:
    try: image=Image.open(uploaded)
    except Exception as e: st.error(f"Could not read the uploaded image: {e}"); st.stop()

    image_col,result_col=st.columns([1,1.15],gap="large")
    with image_col:
        st.markdown('<div class="section-label">INPUT IMAGE</div>',unsafe_allow_html=True)
        st.image(image,caption="Leaf image submitted for AI analysis",use_container_width=True)
        st.markdown(f'<div class="info-box"><strong>📷 Image received</strong><br>Resolution: {image.width} × {image.height}px<br>Format: {image.format or "Image"}</div>',unsafe_allow_html=True)

    try: label,confidence,all_preds=predict(model,class_names,preprocess_fn,img_size,image)
    except Exception as e: st.error(f"Prediction failed: {e}"); st.stop()

    crop,condition=label.split("__",1) if "__" in label else ("Unknown",label)
    clean_crop=crop.replace("_"," ").title(); clean_condition=condition.replace("_"," ").title()
    confidence_percent=confidence*100
    is_healthy="healthy" in condition.lower() or "normal" in condition.lower()
    result_class="healthy-result" if is_healthy else "disease-result"
    result_icon="✅" if is_healthy else "⚠️"
    result_message="The AI model did not detect a disease pattern." if is_healthy else "The AI model detected a disease-associated pattern."

    with result_col:
        st.markdown('<div class="section-label">AI DIAGNOSIS</div>',unsafe_allow_html=True)
        st.markdown(f'<div class="result-card {result_class}"><div style="font-size:42px;margin-bottom:8px">{result_icon}</div><div class="result-label">Detected Crop</div><div class="result-value">{clean_crop}</div><div style="height:1px;background:#e7eee8;margin:17px 0"></div><div class="result-label">Automated Diagnosis</div><div class="result-value">{clean_condition}</div><div class="info-box">{result_message}</div></div>',unsafe_allow_html=True)
        st.write("")
        st.markdown(f'<div class="metric-card"><div style="display:flex;justify-content:space-between;align-items:center"><div><div class="metric-title">MODEL CONFIDENCE</div><div class="metric-value">{confidence_percent:.1f}%</div></div><div style="font-size:30px">🎯</div></div><div style="background:#e7eee8;height:10px;border-radius:20px;margin-top:15px;overflow:hidden"><div style="width:{min(confidence_percent,100):.1f}%;height:100%;border-radius:20px;background:linear-gradient(90deg,#81c784,#2e7d32)"></div></div></div>',unsafe_allow_html=True)

    st.write(""); st.write("")
    st.markdown('<div class="section-label">MODEL INSIGHTS</div><div class="section-title">Classification probability</div><div class="section-description">Probability assigned to each disease class by the AI model.</div>',unsafe_allow_html=True)
    pc1,pc2=st.columns([1.2,1],gap="large")
    sorted_predictions=sorted(zip(class_names,all_preds),key=lambda p:-p[1])
    with pc1:
        for name,prob in sorted_predictions:
            display_name=name.replace("__"," → ").replace("_"," ").title(); percentage=float(prob)*100
            st.markdown(f'<div style="margin:13px 0"><div style="display:flex;justify-content:space-between;margin-bottom:5px;font-size:13px"><span style="color:#36543e;font-weight:600">{display_name}</span><strong style="color:#267a40">{percentage:.2f}%</strong></div><div style="height:8px;background:#e6eee7;border-radius:20px;overflow:hidden"><div style="width:{min(percentage,100):.2f}%;height:100%;border-radius:20px;background:linear-gradient(90deg,#9ccc65,#2e7d32)"></div></div></div>',unsafe_allow_html=True)
    with pc2:
        st.markdown(f'<div class="crop-card"><div style="font-size:42px">{"🌿" if crop.lower()=="cassava" else "🌽"}</div><div class="crop-name">Automated Farm Intelligence</div><div class="crop-description">AgroVision uses computer vision and a trained deep-learning model to analyze leaf characteristics and estimate the most likely crop health condition.</div><div style="height:1px;background:#e4ece5;margin:18px 0"></div><div style="font-size:12px;color:#718076;line-height:1.7"><strong style="color:#356843">Current analysis</strong><br>Crop: {clean_crop}<br>Diagnosis: {clean_condition}<br>Confidence: {confidence_percent:.1f}%<br>AI model: {best_model_name}<br>Model source: {model_source}</div></div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box"><strong>🌱 Ready for automated crop analysis?</strong><br>Upload a clear cassava or maize leaf image above. The AI system will automatically process the image, identify the most likely disease class, and display the model confidence score.</div>',unsafe_allow_html=True)

st.markdown('<div class="footer"><strong>🌿 AgroVision AI</strong> &nbsp;•&nbsp; Intelligent Crop Disease Diagnosis &nbsp;•&nbsp; Smart Agriculture & Computer Vision<br>Built for automated cassava and maize crop monitoring.</div>',unsafe_allow_html=True)


