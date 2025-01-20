FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the module
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "train.py"]

# Args
CMD ["--model_version", "v1", "--data_percentage", "0.15"]