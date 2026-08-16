
import os
# Disable oneDNN for PaddlePaddle 3.0 on Windows CPU (prevents PIR attribute error)
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import streamlit as st
import numpy as np
from PIL import Image
import io


st.set_page_config(
    page_title="Hate Speech Detection",
    page_icon="🚫",
    layout="wide"
)


st.title("🚫 Hate Speech Detection System")
st.markdown("---")


st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose analysis type:",
    ["📝 Text Analysis", "🖼️ Image Analysis"],
    index=0
)

st.sidebar.markdown("---")

st.sidebar.header("Language Selection")
language_option = st.sidebar.selectbox(
    "Choose language:",
    ["English", "Roman Urdu"]
)

language = "english" if language_option == "English" else "roman_urdu"

# ── Model Backend Selection (BiLSTM or HuggingFace zero-shot)
st.sidebar.header("Model Backend")
backend_option = st.sidebar.selectbox(
    "Choose backend:",
    ["BiLSTM (local)", "HuggingFace (zero-shot)", "HuggingFace (text-classifier)"]
)


@st.cache_resource
def load_hf_zero_shot():
    try:
        from huggingface_inference import load_hf_predictor
        return load_hf_predictor()
    except Exception as e:
        st.session_state['hf_zero_load_error'] = str(e)
        return None


@st.cache_resource
def load_hf_text():
    try:
        from huggingface_text_classifier import load_hf_text_predictor
        return load_hf_text_predictor()
    except Exception as e:
        st.session_state['hf_text_load_error'] = str(e)
        return None


hf_zero = None
hf_text = None
if backend_option == "HuggingFace (zero-shot)":
    hf_zero = load_hf_zero_shot()
elif backend_option == "HuggingFace (text-classifier)":
    hf_text = load_hf_text()


@st.cache_resource
def load_bilstm_model(language='english'):
    """Load BiLSTM model and tokenizer"""
    try:
        from lstm_inference import load_lstm_predictor

        if language == 'roman_urdu':
            base_name = "bilstm_model_roman_urdu"
            tokenizer_file = "tokenizer_roman_urdu.pkl"
            config_file = "lstm_config_roman_urdu.pkl"
        else:
            base_name = "bilstm_model"
            tokenizer_file = "tokenizer.pkl"
            config_file = "lstm_config.pkl"

        # Support common save patterns: root or models/, and .h5 or .keras.
        model_candidates = [
            f"{base_name}.h5",
            f"{base_name}.keras",
            os.path.join("models", f"{base_name}.h5"),
            os.path.join("models", f"{base_name}.keras"),
        ]
        model_path = next((p for p in model_candidates if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(
                f"Could not find BiLSTM model file. Tried: {', '.join(model_candidates)}"
            )

        if not os.path.exists(tokenizer_file):
            raise FileNotFoundError(f"Missing tokenizer file: {tokenizer_file}")

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Missing config file: {config_file}")

        predictor = load_lstm_predictor(model_path, language=language)
        return predictor
    except Exception as e:
        st.session_state["lstm_load_error"] = str(e)
        return None


model = load_bilstm_model(language=language)

if model is not None and analysis_mode == "🖼️ Image Analysis":
    # ─── Image Analysis Mode ───
    st.header("🖼️ Image Hate Speech Detection")
    st.markdown("Upload a meme or image containing text. The system will extract the text and analyze it for hate speech.")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, WEBP, TIFF"
    )

    # Track uploaded file — clear old results when image changes or is removed
    current_file = uploaded_image.name if uploaded_image is not None else None
    if current_file != st.session_state.get('last_uploaded_file'):
        st.session_state['last_uploaded_file'] = current_file
        st.session_state.pop('extracted_text', None)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)

        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown("---")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn2:
            extract_btn = st.button("Extract Text from Image", type="primary", use_container_width=True)

        # Extract text and store in session state
        if extract_btn:
            with st.spinner("Extracting text from image using PaddleOCR v4..."):
                try:
                    from ocr_pipeline import MemeOCRPipeline, OCR_LAST_DEBUG

                    @st.cache_resource
                    def get_paddle_pipeline():
                        return MemeOCRPipeline(use_gpu=False)

                    pipeline = get_paddle_pipeline()
                    ocr_result = pipeline.extract_from_pil(image)

                    extracted = ocr_result.get("all_text", "")

                    # Populate debug from OCR_LAST_DEBUG (filled by extract_from_pil backward compat path)
                    # Also update OCR_LAST_DEBUG directly with structured result
                    OCR_LAST_DEBUG.update({
                        "top_text": ocr_result.get("top_text", ""),
                        "bottom_text": ocr_result.get("bottom_text", ""),
                        "num_regions": ocr_result.get("num_regions", 0),
                        "raw_regions": ocr_result.get("raw_regions", []),
                        "best_variant": "paddle_multi_path",
                        "best_count": len(extracted),
                    })

                    st.session_state['extracted_text'] = extracted
                    st.session_state['raw_ocr_text'] = extracted
                    st.session_state['ocr_structured'] = ocr_result

                    # Build debug info for the debug panel
                    debug_info = {
                        'method': 'PaddleOCR_v4_multi_path',
                        'image_size': image.size,
                        'top_text': ocr_result.get("top_text", ""),
                        'bottom_text': ocr_result.get("bottom_text", ""),
                        'num_regions': ocr_result.get("num_regions", 0),
                        'raw_regions': ocr_result.get("raw_regions", []),
                        'best_variant': 'paddle_multi_path',
                        'chars_extracted': len(extracted),
                    }
                    st.session_state['ocr_debug'] = debug_info

                except Exception as e:
                    import traceback
                    st.error(f"❌ OCR Error: {str(e)}")
                    st.error(f"Details: {traceback.format_exc()}")
                    st.info("💡 Tips to fix:\n- Ensure image has clear, visible text\n- Try adjusting image contrast\n- Upload a higher resolution image")
                    st.session_state['extracted_text'] = ""

        # If user selected HuggingFace backend and model is available, allow HF classification
        if backend_option.startswith("HuggingFace") and hf_model is not None and st.session_state.get('extracted_text') and str(st.session_state.get('extracted_text')).strip():
            with st.expander("🔁 Analyze with HuggingFace Zero-Shot"):
                hf_label, hf_conf, hf_probs = hf_model.predict(st.session_state['extracted_text'], return_probabilities=True)
                st.write(f"**HF Prediction:** {hf_label} ({hf_conf:.1f}%)")
                st.write("**HF Probabilities:**")
                st.write({"Hate Speech": f"{hf_probs[0]*100:.2f}%", "Offensive Language": f"{hf_probs[1]*100:.2f}%", "Neither": f"{hf_probs[2]*100:.2f}%"})

        # Show extracted text and allow editing
        if st.session_state.get('extracted_text') and str(st.session_state.get('extracted_text')).strip():
            # Show raw OCR output so the user can see what was corrected
            if 'raw_ocr_text' in st.session_state and st.session_state['raw_ocr_text'] != st.session_state['extracted_text']:
                with st.expander("🔍 Raw OCR Output (before correction)"):
                    st.code(st.session_state['raw_ocr_text'])
            
            # Show structured OCR results (top/bottom text)
            if 'ocr_structured' in st.session_state:
                ocr_res = st.session_state['ocr_structured']
                top_t = ocr_res.get('top_text', '')
                bot_t = ocr_res.get('bottom_text', '')
                if top_t or bot_t:
                    with st.expander("📍 Caption Separation (Top / Bottom)"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Top Caption:**")
                            st.info(top_t if top_t else "(none)")
                        with c2:
                            st.markdown("**Bottom Caption:**")
                            st.info(bot_t if bot_t else "(none)")

            # Show debug info
            if 'ocr_debug' in st.session_state:
                with st.expander("🐛 OCR Debug Info"):
                    debug_info = st.session_state['ocr_debug']
                    image_size = debug_info.get('image_size', (0, 0))
                    method = debug_info.get('method', 'unknown')
                    chars_extracted = debug_info.get('chars_extracted', 0)
                    num_regions = debug_info.get('num_regions', 0)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Image Size", f"{image_size[0]}×{image_size[1]}")
                    with col2:
                        st.metric("OCR Backend", method)
                    with col3:
                        st.metric("Chars Extracted", chars_extracted)
                    with col4:
                        st.metric("Text Regions", num_regions)

                    raw_regions = debug_info.get('raw_regions', [])
                    if raw_regions:
                        st.write("**Detected Regions:**")
                        for i, reg in enumerate(raw_regions, 1):
                            st.write(f"  {i}. \"{reg.get('text', '')}\" — conf: {reg.get('confidence', 0):.2f}")

            st.subheader("Extracted Text")
            st.caption("Text extracted via PaddleOCR v4 multi-path pipeline. Review and edit if needed, then click Analyze.")
            edited_text = st.text_area(
                "Extracted text (editable):",
                value=st.session_state['extracted_text'],
                height=120,
                key="editable_extracted_text"
            )

            col_a1, col_a2, col_a3 = st.columns([1, 1, 1])
            with col_a2:
                analyze_btn = st.button("Analyze for Hate Speech", type="primary", use_container_width=True)

            if analyze_btn and edited_text.strip():
                with st.spinner("Analyzing text for hate speech..."):
                    try:
                        predicted_label, confidence, probabilities = model.predict(edited_text, return_probabilities=True)
                        probabilities = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities

                        # Display results
                        st.markdown("---")
                        st.header("Prediction Results")

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            if predicted_label == "Hate Speech":
                                st.error(f"**Prediction: {predicted_label}**")
                                st.error(f"Confidence: {confidence:.2f}%")
                            elif predicted_label == "Offensive Language":
                                st.warning(f"**Prediction: {predicted_label}**")
                                st.warning(f"Confidence: {confidence:.2f}%")
                            else:
                                st.success(f"**Prediction: {predicted_label}**")
                                st.success(f"Confidence: {confidence:.2f}%")

                        st.subheader("Detailed Probabilities")
                        prob_cols = st.columns(3)
                        labels = ["Hate Speech", "Offensive Language", "Neither"]
                        colors = ["Red", "Yellow", "Green"]

                        for i, (label, color, prob) in enumerate(zip(labels, colors, probabilities)):
                            with prob_cols[i]:
                                st.metric(
                                    label=f"{color} - {label}",
                                    value=f"{prob*100:.2f}%"
                                )

                        st.subheader("Probability Distribution")
                        for i, (label, prob) in enumerate(zip(labels, probabilities)):
                            st.write(f"**{label}**")
                            st.progress(prob)

                        with st.expander("View Preprocessed Text"):
                            preprocessed = model.preprocess_text(edited_text)
                            st.code(preprocessed)

                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.info("Please make sure the models are trained and saved correctly.")

        elif 'extracted_text' in st.session_state and not str(st.session_state.get('extracted_text') or '').strip():
            st.warning("No text could be extracted from the image. Please try a clearer image with visible text.")

    st.markdown("---")
    st.header("ℹ️ How It Works")
    st.info("""
    **Image Hate Speech Detection Pipeline (PaddleOCR v4):**
    1. 📤 **Upload** — Upload a meme or image containing text
    2. 🔍 **Preprocess** — Multi-scale enhancement (CLAHE, bilateral filter, 3 image versions)
    3. 🔎 **OCR** — PaddleOCR v4 extracts text from all versions with NMS deduplication
    4. 📍 **Cluster** — DBSCAN separates top & bottom captions spatially
    5. 🧠 **Classify** — Extracted text is analyzed by the BiLSTM model
    6. 📊 **Results** — View the classification and confidence scores
    """)

elif model is not None and analysis_mode == "📝 Text Analysis":
    
    st.header("Enter Text to Analyze")
    
    
    if language == "roman_urdu":
        placeholder = "Roman Urdu text daalain (e.g., 'kya mein bhooka hon? kutia ab tum ney ye poocha hai')..."
    else:
        placeholder = "Enter text to check for hate speech, offensive language, or neither..."
    
    user_input = st.text_area(
        "Type or paste your text here:",
        height=150,
        placeholder=placeholder
    )
    
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        predict_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    
    if predict_button and user_input.strip():
        with st.spinner("Analyzing text..."):
            try:
                predicted_label, confidence, probabilities = model.predict(user_input, return_probabilities=True)
                probabilities = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                
                
                st.markdown("---")
                st.header("📊 Prediction Results")
                
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    
                    if predicted_label == "Hate Speech":
                        st.error(f"**Prediction: {predicted_label}**")
                        st.error(f"Confidence: {confidence:.2f}%")
                    elif predicted_label == "Offensive Language":
                        st.warning(f"**Prediction: {predicted_label}**")
                        st.warning(f"Confidence: {confidence:.2f}%")
                    else:
                        st.success(f"**Prediction: {predicted_label}**")
                        st.success(f"Confidence: {confidence:.2f}%")
                
                
                st.subheader("📈 Detailed Probabilities")
                prob_cols = st.columns(3)
                
                labels = ["Hate Speech", "Offensive Language", "Neither"]
                colors = ["🔴", "🟡", "🟢"]
                
                for i, (label, color, prob) in enumerate(zip(labels, colors, probabilities)):
                    with prob_cols[i]:
                        st.metric(
                            label=f"{color} {label}",
                            value=f"{prob*100:.2f}%"
                        )
                
                
                st.subheader("📊 Probability Distribution")
                for i, (label, prob) in enumerate(zip(labels, probabilities)):
                    st.write(f"**{label}**")
                    st.progress(prob)
                
                
                with st.expander("🔍 View Preprocessed Text"):
                    preprocessed = model.preprocess_text(user_input)
                    st.code(preprocessed)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please make sure the models are trained and saved correctly.")
    
    elif predict_button and not user_input.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    
    
    st.markdown("---")
    st.header("💡 Example Texts")
    
    if language == "roman_urdu":
        example_texts = {
            "Hate Speech (Hostile)": [
                "kya mein bhooka hon? kutia ab tum ney ye poocha hai, mein ho sakta hon",
                "mein siyah logon se nafrat karta hon",
                "Syrians ghar jao aur maro"
            ],
            "Neither (Neutral)": [
                "mujhe pizza khanay ki shadeed khuwahish hai lekin mere paas paisy nahi hain",
                "Pakistan ne 1992 ka cricket world cup jeeta",
                "islam aman ka mazhab hai"
            ]
        }
    else:
        example_texts = {
            "Hate Speech": [
                "I hate all people from that country",
                "They should all go back where they came from"
            ],
            "Offensive Language": [
                "That's so stupid and annoying",
                "What a dumb thing to say"
            ],
            "Neither": [
                "The weather is nice today",
                "I love learning about machine learning"
            ]
        }
    
    for category, texts in example_texts.items():
        with st.expander(f"📝 {category} Examples"):
            for text in texts:
                if st.button(f"Use: '{text[:50]}...'", key=f"{category}_{text[:10]}"):
                    st.session_state.example_text = text
                    st.rerun()
    
    
    if 'example_text' in st.session_state:
        user_input = st.text_area(
            "Type or paste your text here:",
            value=st.session_state.example_text,
            height=150
        )
        del st.session_state.example_text
    
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info("""
    This system uses machine learning to classify text into three categories:
    - **Hate Speech**: Content that promotes hatred or violence against groups
    - **Offensive Language**: Content that is offensive but not necessarily hate speech
    - **Neither**: Content that is neither hate speech nor offensive
    
    The model uses a Bidirectional LSTM (BiLSTM) deep learning architecture
    to analyze text patterns and make predictions.
    """)
    
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Information")
    st.sidebar.info(f"Language: **{language_option}**")
    st.sidebar.info("Current Model: **BiLSTM (Advanced)**")
    st.sidebar.caption("🧠 Deep Learning Model")
    st.sidebar.caption("Uses sequence-based tokenization and neural networks")
    st.sidebar.caption("Bidirectional LSTM reads text both ways for better context")
    if language == "roman_urdu":
        st.sidebar.caption("🇵🇰 Trained on Roman Urdu dataset")

else:
    st.error("⚠️ BiLSTM Model not found!")
    if "lstm_load_error" in st.session_state:
        st.error(f"Load error: {st.session_state['lstm_load_error']}")
    
    if language == "roman_urdu":
        st.info("""
        To use the Roman Urdu BiLSTM model, you need to:
        1. Make sure 'Hate Speech Roman Urdu (HS-RU-20).csv' is in the current directory
        2. Run 'train_lstm_model_roman_urdu.py' to train and save the model
        3. Refresh this page
        
        The training script will create:
        - bilstm_model_roman_urdu.h5 (Keras model)
        - tokenizer_roman_urdu.pkl (Text tokenizer)
        - lstm_config_roman_urdu.pkl (Model configuration)
        """)
    else:
        st.info("""
        To use the English BiLSTM model, you need to:
        1. Make sure 'labeled_data.csv' is in the current directory
        2. Run 'train_lstm_model.py' to train and save the model
        3. Refresh this page
        
        The training script will create:
        - bilstm_model.h5 (Keras model)
        - tokenizer.pkl (Text tokenizer)
        - lstm_config.pkl (Model configuration)
        """)

