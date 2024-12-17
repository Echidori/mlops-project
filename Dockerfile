FROM python:3.12

WORKDIR /

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment variables, data, and source files
COPY .env ./
COPY data/ ./data/
COPY server/ ./server/
COPY src/ ./src/

# Expose port 2704 (which you will map to 8000)
EXPOSE 2704

# CMD will no longer include the port or the background running
CMD ["fastapi", "run", "server/server.py"]