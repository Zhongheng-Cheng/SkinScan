# SkinScan

Welcome to the Dermatology Generative AI Application README. This document serves as a guide to understand and utilize our AI-powered tool designed for dermatological purposes.

## Purpose

The Dermatology Generative AI App leverages artificial intelligence to assist dermatologists, medical professionals, and researchers in analyzing and generating insights from dermatological images. It utilizes deep learning algorithms trained on a diverse dataset of skin conditions to provide valuable diagnostic support and research capabilities.

## Features

- **Image Analysis Dermatological Condition Detection**: Automatically identifies and categorizes skin conditions based on uploaded images. - **Feature Extraction**: Extracts features and patterns from dermatological images for detailed analysis.
- **Diagnostic Support Decision Support System**: Provides diagnostic recommendations and differential diagnoses based on input images.
- **Second Opinion Tool**: Offers a second opinion by comparing input images with a database of similar cases.
- **Integration API Integration**: Allows seamless integration with existing healthcare systems or research platforms via API.
- **Web Interface**: User-friendly web interface for easy access and interaction.
- **Doctor Search and Appointments**: Calling APIs for retrieving information of Doctors and their contact details

## Installation

```bash
# Cloning the repository
git clone https://github.com/Zhongheng-Cheng/SkinScan
cd SkinScan

# [Optional] Creating virtual environment
python -m venv venv
source venv/bin/activate

# Download dependencies
pip install -r requirements.txt

# Setup Gemini API key
touch .env
# Enter your Google Gemini API Key in ".env" like this:
# GOOGLE_API_KEY="..."

# Run the project
python app.py
```

## Demo Video

[SkinScan demo video - Google Drive](https://drive.google.com/file/d/1F1nKQVgbT7sY_a0bdngg-V7cqawF9vdx/view?usp=sharing)
