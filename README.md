# SkinScan AI

SkinScan AI is a revolutionary telemedicine platform that leverages Artificial Intelligence (AI) to provide instant skin condition diagnoses and personalized treatment recommendations. Users can upload images of their skin concerns, and our AI-powered system will analyze the images, and provide diagnosis, treatment options, and connect them with a board-certified dermatologist for further consultation.

## Features

- **Multimodal Analysis**: Automatically identifies and categorizes skin conditions based on uploaded images or videos.
- **Feature Extraction**: Extracts features and patterns from dermatological images for detailed analysis.
- **Diagnostic Support Decision Support System**: Provides diagnostic recommendations and differential diagnoses based on input images.
- **Second Opinion Tool**: Offers a second opinion by comparing input images with a database of similar cases.
- **Integration API Integration**: Allows seamless integration with existing healthcare systems or research platforms via API.
- **Web Interface**: User-friendly web interface for easy access and interaction.
- **Doctor Search and Appointments**: Calls APIs to retrieve information about Doctors and their contact details.

## How It Works
- **Text Messaging**: Users can directly chat with an AI doctor through text messaging.
- **Upload**: Users can upload images or videos of their skin condition via the platform.
- **AI Analysis**: The Gemini API processes the uploaded media to diagnose the condition.
- **Diagnosis & Recommendations**: Users receive a detailed diagnosis and personalized treatment recommendations.
- **Consultation Option**: If further consultation is needed, users can easily connect with a healthcare professional.

## Getting Started

To start using the platform, follow these steps:

### Environment Setup

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
```

### Database Setup

The application now relies on the Postgresql database. Run the following commands to set up the database in Docker.

```bash
docker pull postgres
docker run --name my_postgres -e POSTGRES_USER=healthiai -e POSTGRES_PASSWORD=healthiai -e POSTGRES_DB=skin_scan_db -p 5432:5432 -d postgres
# (optional) Use `docker ps` to check image availability
docker exec -it my_postgres psql -U healthiai -d skin_scan_db
```

In the Postgresql command line:

```
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL
);
```

Then, use the `exit` command to exit the Postgresql command line.

### Run the project

```bash
python app.py
```

## Demonstration Video

Here is a link to the demonstration video with voice explanation:

[SkinScan AI demo video - Google Drive](https://drive.google.com/file/d/1qmvLnJVpUX_jPJYlNcXRKaGByzmLq93L/view?usp=sharing)
