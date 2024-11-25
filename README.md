# Idefics2 Image-to-audio App

This is a **Streamlit application** that generates text descriptions for uploaded images using the **Idefics2-8b** model and converts the generated text into audio.

## Features
- **Upload an Image**: The app accepts images in `jpg`, `png`, or `jpeg` formats.
- **Generate a Description**: Processes the image using the Idefics2-8b model and generates a human-readable description.
- **Listen to Audio**: Converts the generated description into audio using the `gTTS` library and plays it directly in the app.

---

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- A GPU with sufficient memory (or CPU for slower performance)

### Clone the Repository
```bash
git clone https://github.com/SNikhilVashista/LM_Idefics.git
cd LM_Idefics
