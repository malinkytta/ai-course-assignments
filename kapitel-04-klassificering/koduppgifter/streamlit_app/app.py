import streamlit as st
import numpy as np
import os
from PIL import Image
import joblib
from streamlit_drawable_canvas import st_canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "mnist_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "mnist_scaler.pkl")

st.set_page_config(page_title="MNIST Gissa siffran", layout="centered")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 20% 0%, #1e2a4a 0%, #0e1117 45%),
                    radial-gradient(circle at 80% 100%, #2a1e3d 0%, #0e1117 55%);
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(6px);
        border-radius: 16px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.25);
    }

    h1 {
        background: linear-gradient(90deg, #7dd3fc, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gissa siffran")
st.write("Rita en handskriven siffra i rutan nedan!")


try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.toast("Modell laddad! 🎉")
except FileNotFoundError:
    st.warning(
        "Modell eller scaler hittades inte. Du behöver träna och spara dem först."
    )
    st.info("""Spara från din notebook med:
        
```python
joblib.dump(extra_trees_clf, 'streamlit_app/models/mnist_model.pkl')
joblib.dump(scaler_2, 'streamlit_app/models/mnist_scaler.pkl')
```""")
    st.stop()

draw_col, result_col = st.columns(2)

with draw_col:
    with st.container(border=True, height=435):
        st.subheader("✏️ Rita en siffra")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            return_image_data=True,
            key="canvas",
        )
        st.caption("Sudda allt med papperskorgen i verktygsraden ovanför rutan ↑")

image_to_process = None

if canvas_result.image_data is not None:
    if canvas_result.image_data[:, :, :3].sum() > 0:
        image_to_process = canvas_result.image_data

with result_col:
    with st.container(border=True, height=435):
        if image_to_process is not None:
            # Konvertera ritningen till gråskala
            img = Image.fromarray(image_to_process.astype(np.uint8)).convert("L")
            img_np = np.array(img)

            # Hitta var siffran är ritad
            mask = img_np > 20
            coords = np.argwhere(mask)

            # Beskär bilden så bara siffran syns
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            digit_crop = img.crop((x0, y0, x1, y1))

            # Gör siffran mindre (max 20x20)
            w, h = digit_crop.size
            scale = 20 / max(w, h)
            new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
            digit_resized = digit_crop.resize((new_w, new_h))

            # Centrera siffran i en 28x28 bild
            img_resized = Image.new("L", (28, 28), color=0)
            paste_x = (28 - new_w) // 2
            paste_y = (28 - new_h) // 2
            img_resized.paste(digit_resized, (paste_x, paste_y))

            # Gör bild till tal
            img_array = np.array(img_resized).astype(np.float64)

            img_flat = img_array.reshape(1, -1)

            # Skala bilden som vi gjorde när vi tränade
            img_scaled = scaler.transform(img_flat)

            # Predicera siffran
            predicted_digit = model.predict(img_scaled)[0]
            probabilities = model.predict_proba(img_scaled)[0]
            confidence = np.max(probabilities) * 100

            st.subheader("Resultat")
            st.metric(
                "Förutsagd siffra", predicted_digit, delta=f"{confidence:.1f}% säkert"
            )

            st.image(img_array.astype(np.uint8), width=150)
        else:
            st.subheader("Resultat")
            st.info("Rita en siffra till vänster så visas gissningen här.")

if image_to_process is not None:
    st.subheader("Sannolikhet för varje siffra")
    chart_data = {}
    for i in range(10):
        chart_data[str(i)] = float(probabilities[i] * 100)
    st.bar_chart(chart_data)
