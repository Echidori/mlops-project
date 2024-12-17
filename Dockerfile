FROM python:3.12

WORKDIR /

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY .env ./
COPY data/ ./
COPY server ./
COPY src/ ./

EXPOSE 2704

CMD ["fastapi", "run", "server/server.py", "--port", "2704"]