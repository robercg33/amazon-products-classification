FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the module
COPY . .

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]

# Args
CMD ["--output_route", "clean_data.parquet", "--batch_size", "512", "--max_length", "256"]