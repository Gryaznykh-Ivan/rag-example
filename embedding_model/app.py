from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Загрузка модели
model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

# Создание FastAPI приложения
app = FastAPI()

# Определение модели данных для запроса
class TextInput(BaseModel):
    chunks: list[str]

@app.post("/embedding")
async def predict(input_data: TextInput):
    try:
        embeddings = model.encode(input_data.chunks).tolist()
        return { "embeddings": embeddings }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))