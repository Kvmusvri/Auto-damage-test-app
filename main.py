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

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "app" / "tmp"  # Изменили путь на app/tmp
TMP_DIR.mkdir(exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/app/tmp", StaticFiles(directory="app/tmp"), name="tmp")  # Изменили монтирование на /app/tmp

templates = Jinja2Templates(directory="app/templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": list(MODEL_MAP.keys()) + ["Все модели"]}
    )

@app.post("/process")
async def process_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form(...),
    imgsz: int = Form(...),
    conf: float = Form(...),
    iou: float = Form(...)
):
    if not files:
        return JSONResponse(status_code=400, content={"error": "Нет файлов для обработки"})

    for file in files:
        if not file.content_type.startswith('image/'):
            return JSONResponse(status_code=400, content={"error": f"Файл {file.filename} не является изображением"})

    session_id = str(uuid.uuid4())
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    results_all_models = {}

    if model_name == "Все модели":
        model_names = list(MODEL_MAP.keys())
    else:
        if model_name not in MODEL_MAP:
            return JSONResponse(status_code=400, content={"error": "Модель не найдена"})
        model_names = [model_name]

    saved_file_paths = []
    original_filenames = []
    for file in files:
        tmp_file_path = session_dir / f"{uuid.uuid4().hex}_{file.filename}"
        contents = await file.read()
        with open(tmp_file_path, "wb") as f:
            f.write(contents)
        saved_file_paths.append(tmp_file_path)
        original_filenames.append(file.filename)

    for mn in model_names:
        pt_file = BASE_DIR / "models_from_hub" / f"{mn}.pt"
        if not pt_file.exists():
            return JSONResponse(status_code=500, content={"error": f"Файл модели {pt_file} не найден"})
        model = YOLO(str(pt_file))
        try:
            results = model([str(p) for p in saved_file_paths], conf=conf, iou=iou, imgsz=imgsz)
        except Exception as e:
            logger.error(f"Ошибка при обработке моделью {mn}: {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Ошибка при обработке моделью {mn}: {str(e)}"})

        result_files = []
        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            result_filename = f"{mn}_{original_filenames[i]}"
            output_path = session_dir / result_filename
            im_rgb.save(output_path)
            logger.info(f"Сохранено: {output_path}")
            result_files.append({"original_filename": original_filenames[i], "result_filename": f"{session_id}/{result_filename}"})
        results_all_models[mn] = result_files

    return {"results": results_all_models}

@app.post("/clear_tmp")
async def clear_tmp():
    try:
        if TMP_DIR.exists():
            shutil.rmtree(TMP_DIR)
            TMP_DIR.mkdir()
            logger.info("Папка app/tmp успешно очищена")
        return {"status": "success", "message": "Папка app/tmp очищена"}
    except Exception as e:
        logger.error(f"Ошибка при очистке папки app/tmp: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Ошибка при очистке папки app/tmp: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)