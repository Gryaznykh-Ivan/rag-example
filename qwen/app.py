from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Загрузка модели и токенизатора
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Создание FastAPI приложения
app = FastAPI()


# Определение модели данных для запроса
class MessagesInput(BaseModel):
    prompt: str
    system: str


@app.post("/generate")
async def predict(input_data: MessagesInput):
    try:
        # Задание для модели
        messages = [
            {"role": "system", "content": input_data.system},
            {"role": "user", "content": input_data.prompt},
        ]

        # Применение шаблона чата
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Токенизация с созданием маски внимания
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
        ).to(model.device)

        # Генерация текста
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
        )

        # Удаление входных токенов из сгенерированных
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        # Декодирование сгенерированных токенов
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
