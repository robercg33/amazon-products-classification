FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the module
COPY . .

# Expose the port
EXPOSE 5000

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Set the environment variable for flask
ENV FLASK_APP=src.run

# Set the entrypoint
CMD ["flask", "run", "--host=0.0.0.0"]