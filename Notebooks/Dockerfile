# -----------------------------
# Step 1: Base Image
# -----------------------------
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# -----------------------------
# Step 2: Copy and Install Dependencies
# -----------------------------
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Step 3: Copy Application Files
# -----------------------------
COPY . .

# -----------------------------
# Step 4: Expose Streamlit Port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Step 5: Run Streamlit App
# -----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
