# Start with Python 3.10 slim image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy all files from current directory to container
COPY . .

# Expose ports needed for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501
