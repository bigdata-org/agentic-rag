# Use the official Python 3.10-slim image
FROM  python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

# Set the working directory in the container
WORKDIR /app

# Copy and install dependencies separately to leverage Docker cache
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose the port
EXPOSE ${PORT}

# Use shell form so that ${PORT} is expanded when the container starts
CMD sh -c "python -m uvicorn app:app --host 0.0.0.0 --port ${PORT}"