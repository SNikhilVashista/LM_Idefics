import streamlit as st
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from gtts import gTTS
import torch
from PIL import Image
import os

# Set the device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the Idefics2-8b Model and Processor
@st.cache_resource
def load_model():
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    # Load processor
    processor = AutoProcessor.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        do_image_splitting=False,  # Not splitting large images
        size={"longest_edge": 448, "shortest_edge": 378},  # Lower resolution for memory efficiency
    )
    # Load model
    model = AutoModelForVision2Seq.from_pretrained(
        "HuggingFaceM4/idefics2-8b",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    return processor, model

processor, model = load_model()

# Streamlit Application
st.title("Idefics2-8b LM")
st.write("Upload an image to generate a description and listen to it as audio.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Define the prompt
    prompt = "<image>Describe this image."

    # Prepare inputs for the model
    st.write("Generating description...")
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt")    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Generate the text description
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Clean up the generated text
   # generated_text = generated_text.replace("<image>", "").replace("</image>", "")
    # Post-process to remove redundant phrases
    cleaned_text = generated_text.replace("Describe this image.", "").strip()

    # Display the generated text
    st.write("### Generated Description:")
    st.write(cleaned_text)

    # Convert the text to audio
    st.write("Converting description to audio...")
    tts = gTTS(cleaned_text)
    audio_path = "output.mp3"
    tts.save(audio_path)

    # Play the audio
    audio_file = open(audio_path, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

    # Clean up the audio file
    audio_file.close()
    os.remove(audio_path)
