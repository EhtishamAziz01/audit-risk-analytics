FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY dashboard/ dashboard/
COPY .streamlit/ .streamlit/
COPY data/processed/ data/processed/

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the dashboard
CMD ["streamlit", "run", "dashboard/streamlit_app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501"]
