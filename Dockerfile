from python:3.12

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY server ./
COPY data ./
EXPOSE 2704

CMD ["fastapi", "run", "server/server.py", "--port", "2704"]