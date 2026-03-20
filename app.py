"""
app.py — Streamlit Cloud deployment
------------------------------------
Standalone file — no local pipeline imports.
ELA computed directly here.
Calls GCP API for classification.
"""

import io
import requests
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import streamlit as st

# ── GCP API URL ───────────────────────────────────────────────────────────────
API_URL = "https://doc-intel-api-165472080585.asia-south1.run.app"

# ── ELA function (copied inline — no import needed) ───────────────────────────
def compute_ela(pil_image: Image.Image):
    """
    Error Level Analysis — detects tampered regions.
    Returns ela_rgb (H,W,3) uint8 for display.
    """
    ELA_QUALITY = 90
    ELA_AMPLIFY = 15
    SIZE        = (512, 512)

    original = pil_image.convert("RGB").resize(SIZE, Image.LANCZOS)
    buf = io.BytesIO()
    original.save(buf, format="JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_arr   = np.array(original,     dtype=np.float32)
    recomp_arr = np.array(recompressed, dtype=np.float32)
    diff       = np.abs(orig_arr - recomp_arr)
    ela_rgb    = np.clip(diff * ELA_AMPLIFY, 0, 255).astype(np.uint8)
    return ela_rgb


def preprocess(pil_image: Image.Image) -> Image.Image:
    """Clean image before ELA."""
    if pil_image.mode == "RGBA":
        bg = Image.new("RGB", pil_image.size, (255, 255, 255))
        bg.paste(pil_image, mask=pil_image.split()[3])
        image = bg
    else:
        image = pil_image.convert("RGB")
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    image = ImageOps.autocontrast(image, cutoff=1)
    return image


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Document Intelligence System",
    page_icon  = "🔍",
    layout     = "wide",
)

st.title("🔍 Document Intelligence System")
st.caption(
    "Upload a document image to check if it has been digitally tampered with. "
    "Powered by Error Level Analysis + Random Forest — deployed on GCP Cloud Run."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.write(
        "This system uses **Error Level Analysis (ELA)** "
        "combined with a **Random Forest classifier** trained on "
        "12,614 images from the CASIA v2 forensics dataset."
    )
    st.divider()

    st.subheader("API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            st.success("API server: Online")
            st.caption(f"Model loaded: {data.get('model_loaded')}")
        else:
            st.error("API server: Error")
    except Exception:
        st.error("API server: Unreachable")

    st.divider()
    st.subheader("Stats")
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        if r.status_code == 200:
            s = r.json()
            st.metric("Total processed", s.get("total", 0))
            st.metric("Tampered found",  s.get("total_tampered", 0))
            st.metric("Model accuracy",  "79.39%")
    except Exception:
        st.write("No stats yet.")

    st.divider()
    st.subheader("Tech Stack")
    st.write("ELA + Random Forest + FastAPI + Docker + GCP Cloud Run")

# ── Main upload area ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Choose a document image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Passports, degree certificates, employment letters, licenses etc."
)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    pil_image  = Image.open(io.BytesIO(file_bytes))

    # ── Show original + ELA side by side ──────────────────────────────────────
    st.subheader("Image Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            pil_image,
            caption="Uploaded document",
            use_container_width=True
        )

    with col2:
        with st.spinner("Computing ELA map..."):
            clean   = preprocess(pil_image)
            ela_rgb = compute_ela(clean)
        st.image(
            ela_rgb,
            caption="ELA map — bright regions = possible tampering",
            use_container_width=True,
        )

    # ── Run verification via GCP API ──────────────────────────────────────────
    st.subheader("Verification Result")
    with st.spinner("Sending to GCP API for classification..."):
        try:
            response = requests.post(
                f"{API_URL}/verify",
                files={"file": (uploaded_file.name, file_bytes, "image/jpeg")},
                timeout=60,
            )

            if response.status_code == 200:
                result = response.json()

                # ── Verdict banner ────────────────────────────────────────────
                if result["is_uncertain"]:
                    st.warning(f"⚠️  {result['verdict']}")
                elif result["label"] == "tampered":
                    st.error(f"🚨  {result['verdict']}")
                else:
                    st.success(f"✅  {result['verdict']}")

                # ── Metrics row ───────────────────────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Label",          result["label"].upper())
                m2.metric("Confidence",     f"{result['confidence']:.1%}")
                m3.metric("ELA mean",       f"{result['ela_mean']:.2f}")
                m4.metric("ELA region std", f"{result['ela_region_std']:.2f}")

                # ── Probability bars ──────────────────────────────────────────
                st.write("**Probability breakdown**")
                p1, p2 = st.columns(2)
                p1.progress(
                    result["authentic_prob"],
                    text=f"Authentic: {result['authentic_prob']:.1%}"
                )
                p2.progress(
                    result["tampered_prob"],
                    text=f"Tampered: {result['tampered_prob']:.1%}"
                )

                # ── OCR text ──────────────────────────────────────────────────
                if result["ocr_text"]:
                    with st.expander("📄 Extracted text (OCR)"):
                        st.write(result["ocr_text"])
                        st.caption(
                            f"OCR confidence: {result['ocr_confidence']:.1%}"
                        )

                # ── Technical details ─────────────────────────────────────────
                with st.expander("🔧 Technical details"):
                    st.json(result)

            else:
                st.error(
                    f"API error {response.status_code}: {response.text}"
                )

        except requests.exceptions.Timeout:
            st.error(
                "Request timed out. GCP Cloud Run may be cold-starting "
                "(first request takes ~10s). Please try again."
            )
        except requests.exceptions.ConnectionError:
            st.error("Cannot reach GCP API. Check your internet connection.")

# ── Verification history ──────────────────────────────────────────────────────
st.divider()
if st.button("📋 Show verification history"):
    try:
        r = requests.get(f"{API_URL}/history?limit=10", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data["total"] == 0:
                st.info("No verifications yet — upload a document above.")
            else:
                import pandas as pd
                df = pd.DataFrame(data["results"])
                cols = [
                    "id", "filename", "label",
                    "confidence", "ela_region_std", "created_at"
                ]
                st.dataframe(
                    df[[c for c in cols if c in df.columns]],
                    use_container_width=True
                )
    except Exception as e:
        st.error(f"Could not fetch history: {e}")