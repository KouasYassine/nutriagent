
from pathlib import Path
from io import BytesIO
import hashlib
import json
import sys

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


from config import RESULTS_DIR, TARGETS
from inference import Ensemble, Predictor

CONFORMAL = RESULTS_DIR / "uncertainty" / "conformal.json"
DEFAULT_MODEL = "A2_Mass2"

st.set_page_config(
    page_title="NutriAgent",
    page_icon="🥗",
    layout="wide"
)


@st.cache_resource
def load_conformal():
    """Interval parameters from src/uncertainty.py (None if not computed yet)."""
    if not CONFORMAL.exists():
        return None
    return json.loads(CONFORMAL.read_text())


@st.cache_resource
def load_predictor():
    """Use the model the intervals were calibrated for, so the ranges stay valid.
    Names: "v3_full", "v3_full+tta" (flip averaging) or an ensemble saved by evaluate.py."""
    conf = load_conformal()
    name = conf["model"] if conf else DEFAULT_MODEL
    tta = name.endswith("+tta")
    base = name.removesuffix("+tta")
    members_file = RESULTS_DIR / "evaluation" / f"{base}.members.txt"
    if members_file.exists():
        predictor = Ensemble(members_file.read_text().split())
        tta = True
    else:
        predictor = Predictor(base)
    return predictor, tta, name


def intervals(values):
    """90 % conformal range for each nutrient."""
    conf = load_conformal()
    if conf is None:
        return None
    from uncertainty import interval

    mode = conf["recommended_mode"]
    out = {}
    for t in TARGETS:
        lo, hi = interval([values[t]], conf["quantiles"][mode][t], mode, conf["offsets"][t])
        out[t] = (float(lo[0]), float(hi[0]))
    return out


def predict(image):
    predictor, tta, _ = load_predictor()
    values = predictor.predict_image(image, tta=tta)
    return {t: float(v) for t, v in zip(TARGETS, values)}


def scale_portion(predictions, actual_mass):
    estimated_mass = predictions["mass"]

    if estimated_mass <= 0:
        raise ValueError("AI-estimated mass must be positive.")

    factor = actual_mass / estimated_mass

    updated = {
        target: predictions[target] * factor
        for target in TARGETS
    }
    updated["mass"] = actual_mass
    return updated


def reference_estimate(mass, per_100g):
    result = {
        target: per_100g[target] * mass / 100
        for target in TARGETS
        if target != "mass"
    }
    result["mass"] = mass
    return result


def show_nutrition(values, ranges=None):
    def help_text(t, unit, fmt):
        if ranges is None:
            return None
        lo, hi = ranges[t]
        return f"90% range: {lo:{fmt}} – {hi:{fmt}} {unit}"

    c1, c2 = st.columns(2)
    c1.metric("🔥 Calories", f"{values['calories']:.0f} kcal")
    c2.metric("⚖️ Mass", f"{values['mass']:.0f} g")
    if ranges is not None:
        c1.caption(help_text("calories", "kcal", ".0f"))
        c2.caption(help_text("mass", "g", ".0f"))

    cols = st.columns(3)
    for col, target in zip(cols, ["fat", "carbs", "protein"]):
        col.metric(target.capitalize(), f"{values[target]:.1f} g")
        if ranges is not None:
            col.caption(help_text(target, "g", ".1f"))


st.title("🥗 NutriAgent")
st.write("Calories and macronutrients from a single food photo, with calibrated uncertainty ranges.")

uploaded = st.file_uploader(
    "Upload a food photograph",
    type=["png", "jpg", "jpeg"]
)

if uploaded is not None:
    raw = uploaded.getvalue()
    image_id = hashlib.sha256(raw).hexdigest()
    image = Image.open(BytesIO(raw)).convert("RGB")

    st.image(image, caption="Your meal", width=500)

    if st.button("Analyze my meal", type="primary"):
        with st.spinner("Analyzing..."):
            st.session_state["prediction"] = predict(image)
            st.session_state["prediction_image"] = image_id

    if (
        "prediction" in st.session_state
        and st.session_state.get("prediction_image") == image_id
    ):
        original = st.session_state["prediction"]

        st.subheader("AI-only estimate")
        ranges = intervals(original)
        show_nutrition(original, ranges)
        if ranges is not None:
            st.caption(
                "Ranges are conformal prediction intervals calibrated separately for "
                "small, medium and large portions: on held-out Nutrition5k dishes, "
                "the true value fell inside them about 90% of the time."
            )

        st.divider()
        st.subheader("Optional refinement")

        refine = st.checkbox(
            "I want to improve the estimate manually",
            key="refine"
        )

        if refine:
            method = st.radio(
                "How would you like to refine it?",
                [
                    "Correct portion mass",
                    "Use nutrition values per 100 g"
                ]
            )

            default_mass = max(1.0, round(original["mass"], 1))

            actual_mass = st.number_input(
                "Actual or estimated portion mass (g)",
                min_value=1.0,
                value=float(default_mass),
                step=5.0
            )

            if method == "Correct portion mass":
                refined = scale_portion(original, actual_mass)
                source = "AI prediction scaled by user-provided mass"

                st.info(
                    "This adjusts portion size only. It assumes "
                    "the AI's estimated nutrient composition is correct."
                )

            else:
                ingredient = st.selectbox(
                    "Food reference",
                    [
                        "Avocado — illustrative reference",
                        "Custom values per 100 g"
                    ]
                )

                if ingredient.startswith("Avocado"):
                    # Approximate generic avocado reference.
                    # Not a verified Nutrition5k or CIQUAL entry.
                    per_100g = {
                        "calories": 160.0,
                        "fat": 14.7,
                        "carbs": 8.5,
                        "protein": 2.0
                    }
                else:
                    per_100g = {}

                    st.write(
                        "Enter nutrition values from a food label "
                        "or reference database."
                    )

                    for target in [
                        "calories", "fat", "carbs", "protein"
                    ]:
                        unit = "kcal" if target == "calories" else "g"
                        per_100g[target] = st.number_input(
                            f"{target.capitalize()} ({unit}/100 g)",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            key=f"reference_{target}"
                        )

                refined = reference_estimate(
                    actual_mass, per_100g
                )

                source = (
                    "Generic avocado reference"
                    if ingredient.startswith("Avocado")
                    else "User-provided nutrition reference"
                )

                st.info(
                    "This calculates nutrition for the specified "
                    "food composition and mass. For mixed meals, "
                    "use a reference for the whole meal or "
                    "calculate each ingredient separately."
                )

            st.subheader("Refined estimate")
            show_nutrition(refined)

            comparison = pd.DataFrame({
                "Nutrient": [
                    "Calories (kcal)", "Mass (g)",
                    "Fat (g)", "Carbs (g)", "Protein (g)"
                ],
                "AI estimate": [
                    original[t] for t in TARGETS
                ],
                "Refined estimate": [
                    refined[t] for t in TARGETS
                ]
            }).set_index("Nutrient")

            st.dataframe(
                comparison.round(1),
                use_container_width=True
            )

            st.caption(f"Refinement method: {source}")

        st.warning(
            "Nutrition estimates are approximate. Images alone "
            "cannot reliably reveal ingredients, oils, preparation "
            "methods or exact portion sizes."
        )

st.divider()
st.caption(f"NutriAgent · model: {load_predictor()[2]} · trained on Nutrition5k · conformal intervals · PyTorch")
