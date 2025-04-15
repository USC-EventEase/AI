# Use an official Python slim image
FROM python:3.13-slim

# Ensure Python output is not buffered (so log messages show immediately)
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# (Optional) Install any system dependencies if needed (e.g. gcc)
RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the SentenceTransformer model so that it is cached in the image.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy all of your application files to the container
COPY . .

# Expose the port your Flask app will run on (6043 in your code)
EXPOSE 3002

# Set the default command to run your app.
CMD ["python", "app.py"]
