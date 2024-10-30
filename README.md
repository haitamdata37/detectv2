# 📱💻 Student Device Detector

This project uses a pre-trained DETR (DEtection TRansformers) model to detect specific devices (phones and laptops) in uploaded images. The application is built using Streamlit, providing a simple user interface to help identify if restricted devices are present.

## 🎯 Features
- **Object Detection**: Detects objects in uploaded images, including phones and laptops.
- **Device Warning**: Displays a warning if a phone or laptop is detected.
- **Object List**: Lists all detected objects with confidence scores, even if no restricted devices are found.
- **Interactive UI**: Easy-to-use Streamlit interface for image upload and results.

- 🚀 Usage
Upload an Image: In the Streamlit app, upload an image to check for phones and laptops.
View Results: If a phone or laptop is detected, a warning will appear. Otherwise, a success message is shown.
Review Detected Objects: The app lists all detected objects along with their confidence scores, so you can see everything identified in the image.
📚 Model Information
This app uses the facebook/detr-resnet-50 model from Hugging Face, with the no_timm revision to avoid additional dependencies.

🔧 File Structure
app.py: The main application file that runs the Streamlit app.
requirements.txt: Lists all required dependencies for the project.
README.md: Project description, setup instructions, and usage guide.
