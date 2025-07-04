from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from model_config import MODEL_MAP
import uvicorn
from fastapi.responses import JSONResponse
from PIL import Image
import shutil
import uuid
from pathlib import Path
import time
from ultralytics import YOLO
import logging


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent  # Путь к папке app
TMP_DIR = BASE_DIR / "tmp"  # Папка app/tmp
TMP_DIR.mkdir(exist_ok=True)

# Определяем MODEL_MAP с ключами, соответствующими вашим моделям
MODEL_MAP = {
    'Auto_damage_united_with_third_part_yolo_model': 'Auto_damage_united_with_third_part_yolo_model.pt',
    'Auto_damage_yolo_THIRD_PART_model': 'Auto_damage_yolo_THIRD_PART_model.pt',
    'Auto_damage_yolo_united_1_2_model': 'Auto_damage_yolo_united_1_2_model.pt',
    'Auto_damage_yolo_SECOND_PART_model': 'Auto_damage_yolo_SECOND_PART_model.pt',
    'Auto_damage_yolo_FIRST_PART_model': 'Auto_damage_yolo_FIRST_PART_model.pt'
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/app/tmp", StaticFiles(directory="app/tmp"), name="tmp")

templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": ["Все модели"]}  # Только "Все модели"
    )

@app.post("/process")
async def process_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form(...),
    imgsz: int = Form(...),
    conf: float = Form(...),
    iou: float = Form(...)
):
    logger.info(f"Получен запрос: model_name={model_name}, imgsz={imgsz}, conf={conf}, iou={iou}, files={len(files)}")

    if not files:
        logger.error("Нет файлов для обработки")
        return JSONResponse(status_code=400, content={"error": "Нет файлов для обработки"})

    for file in files:
        if not file.content_type.startswith('image/'):
            logger.error(f"Файл {file.filename} не является изображением")
            return JSONResponse(status_code=400, content={"error": f"Файл {file.filename} не является изображением"})

    session_id = str(uuid.uuid4())
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    logger.info(f"Создана директория сессии: {session_dir}")

    results_all_models = {}

    model_names = list(MODEL_MAP.keys())  # Все модели из MODEL_MAP
    logger.info(f"Модели для обработки: {model_names}")

    saved_file_paths = []
    original_filenames = []
    for file in files:
        tmp_file_path = session_dir / f"{uuid.uuid4().hex}_{file.filename}"
        contents = await file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(contents)
        saved_file_paths.append(tmp_file_path)
        original_filenames.append(file.filename)
        logger.info(f"Сохранен файл: {tmp_file_path}")

    for mn in model_names:
        pt_file = BASE_DIR / "models_from_hub" / MODEL_MAP[mn]
        logger.info(f"Проверка файла модели: {pt_file}")
        if not pt_file.exists():
            logger.error(f"Файл модели {pt_file} не найден")
            return JSONResponse(status_code=500, content={"error": f"Файл модели {pt_file} не найден"})

        logger.info(f"Загрузка модели: {mn}")
        try:
            model = YOLO(str(pt_file))
            results = model([str(p) for p in saved_file_paths], conf=conf, iou=iou, imgsz=imgsz)
            logger.info(f"Модель {mn} обработала {len(results)} изображений")
        except Exception as e:
            logger.error(f"Ошибка при обработке моделью {mn}: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Ошибка при обработке моделью {mn}: {str(e)}"})

        result_files = []
        for i, r in enumerate(results):
            try:
                im_bgr = r.plot()
                im_rgb = Image.fromarray(im_bgr[..., ::-1])
                result_filename = f"{mn}_{original_filenames[i]}"
                output_path = session_dir / result_filename
                im_rgb.save(output_path)
                logger.info(f"Сохранено обработанное изображение: {output_path}")
                result_files.append({
                    "original_filename": original_filenames[i],
                    "result_filename": f"{session_id}/{result_filename}"
                })
            except Exception as e:
                logger.error(f"Ошибка при сохранении результата для {original_filenames[i]}: {str(e)}")
                continue
        results_all_models[mn] = result_files

    logger.info(f"Результат обработки: {results_all_models}")
    return {"results": results_all_models}

@app.post("/clear_tmp")
async def clear_tmp():
    try:
        if TMP_DIR.exists():
            for session_dir in TMP_DIR.iterdir():
                if session_dir.is_dir():
                    shutil.rmtree(session_dir)
                    logger.info(f"Удалена директория: {session_dir}")
            TMP_DIR.mkdir(exist_ok=True)
            logger.info("Папка app/tmp успешно очищена")
        return {"status": "success", "message": "Папка app/tmp очищена"}
    except Exception as e:
        logger.error(f"Ошибка при очистке папки app/tmp: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Ошибка при очистке папки app/tmp: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)