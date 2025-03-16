# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем зависимости для системы
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем pip и обновляем его
RUN pip install --no-cache-dir --upgrade pip

# Устанавливаем зависимости из requirements.txt
RUN pip install --no-cache-dir fastapi
RUN pip install --no-cache-dir uvicorn
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir sentence_transformers 
RUN pip install --no-cache-dir torch

# Копируем код приложения
COPY app.py /app/app.py

# Устанавливаем рабочую директорию
WORKDIR /app

# Указываем команду для запуска приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]