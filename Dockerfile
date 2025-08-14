FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs

# Копируем только requirements.txt сначала (для кэширования)
COPY requirements.txt .

# Установка зависимостей
RUN python -m pip install --no-cache-dir -r requirements.txt

# Копируем оставшийся исходный код
COPY . .

# Точки монтирования
VOLUME /app/input
VOLUME /app/output

CMD ["python", "./app/app.py"]