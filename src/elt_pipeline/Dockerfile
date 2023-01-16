FROM python:3.9-slim

# Set the working directory
WORKDIR /usr/src/app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "scripts/run_pipeline.sh"]