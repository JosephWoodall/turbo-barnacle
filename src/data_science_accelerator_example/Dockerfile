# Choose your desired base image. Here, we're using the official Python 3.8 image from Docker Hub.
FROM python:3.8

# Set the working directory in the container to /app.
WORKDIR /app

# Copy the requirements.txt file into the container and install the required packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the local data_science_accelerator directory into the container at /app.
COPY . .

# Set environment variables if necessary (e.g. for Flask app).
ENV FLASK_APP=app.py

# Run the command to start your application (e.g. a Flask app).
CMD ["python", "app.py"]
