# Use the official Python image from the Docker Hub
FROM python:3.10

# Add metadata with labels
LABEL maintainer="end to end mlops project"
LABEL version="1.0"
LABEL description="This is a Docker image for the rain prediction ML model."

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . .

# Install any dependencies listed in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose a different port to avoid conflict with Jenkins
EXPOSE 5000

# Define the command to run the application
CMD ["python", "src/rainprediction.py"]