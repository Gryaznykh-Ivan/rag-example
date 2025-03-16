# Используем официальный образ Python
FROM python:3.9-slim

# Устанавливаем pip и обновляем его
RUN pip install --upgrade pip

# Устанавливаем зависимости из requirements.txt
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pydantic
RUN pip install transformers
RUN pip install torch
RUN pip install 'accelerate>=0.26.0'

# Копируем код приложения
COPY app.py /app/app.py

# Устанавливаем рабочую директорию
WORKDIR /app

# Указываем команду для запуска приложения
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
