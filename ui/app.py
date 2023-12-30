import sys
import random
from pathlib import Path
import streamlit as st
sys.path.append(".")
import config.config as config
from topic_classification import main, utils

tips = utils.load_dict(config.TIPS_FILE)

# Title
st.title("Topic Classification")

tab1, tab2, tab3 = st.tabs(["🔢 Data", "📊 Performance", "🚀 Inference"])

# Sections
with tab1:
    st.header("🔢 Data")
    df = utils.load_data(config.PREDICTIONS_DATA_FILE)
    st.text(f"Tweets (count: {len(df)})")
    st.write(df)

with tab2:
    st.header("📊 Performance")
    performance_fp = Path(config.CONFIG_DIR, "performance.json")
    performance = utils.load_dict(filepath=performance_fp)
    st.text("Overall:")
    st.write(performance["overall"])
    topic = st.selectbox("Choose a topic: ", list(performance["class"].keys()))
    st.write(performance["class"][topic])
    slice_ = st.selectbox("Choose a slice: ", list(performance["slices"].keys()))
    st.write(performance["slices"][slice_])

with tab3:
    st.header("🚀 Inference")
    run_id = st.text_input("Enter run ID:", open(Path(config.CONFIG_DIR, "run_id.txt")).read())
    text = st.text_area("My Contact's Tweet:", "🏀🎾🏈 Sports bring us together, transcending borders and differences, uniting us under the banner of athleticism and passion! Whether it's the thrill of a last-minute goal, a breathtaking slam dunk, or a hard-fought match, sports ignite our spirits and remind us of the power of teamwork and dedication. Let's cheer for our favorite athletes and celebrate the magic of sports! 🎉 #Sports #Passion #Teamwork")
    pressed = st.button("Suggest Conversation Topic")
    if pressed:
        prediction = main.predict_topic(text=text, run_id=run_id)[0]
        # tips = tips[prediction["predicted_topic"]]
        # random_tip = random.choice(tips)
        st.info(prediction["predicted_topic"])


