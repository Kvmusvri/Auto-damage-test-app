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
from ultralytics.utils.plotting import colors
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging
import torch
import gc
from datetime import datetime
import sys
import cv2
import numpy as np
import traceback
from typing import List, Dict, Any
import random

# Настройка логирования
def setup_logging():
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Создаем логгер
    logger = logging.getLogger('app')
    logger.setLevel(logging.INFO)
    
    # Очищаем существующие обработчики
    logger.handlers.clear()
    
    # Формат логов
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # Обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Обработчик для файла
    file_handler = logging.FileHandler(
        log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log", 
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

BASE_DIR = Path(__file__).resolve().parent
TMP_DIR = BASE_DIR / "tmp"
TMP_DIR.mkdir(exist_ok=True)

# Глобальный словарь для хранения цветов классов (как в YOLO)
CLASS_COLORS = {}

def get_class_color(class_id: int) -> tuple:
    """Получает цвет для класса, ПРИНУДИТЕЛЬНО исключая черный цвет"""
    if class_id not in CLASS_COLORS:
        # Используем функцию colors() от YOLO
        class_color = colors(class_id, bgr=True)
        logger.info(f"🎨 YOLO вернул цвет для класса {class_id}: BGR={class_color}")
        
        # БАН-ЛИСТ для черного цвета - принудительно заменяем на красный
        b, g, r = class_color[0], class_color[1], class_color[2]  # BGR
        
        # ПРИНУДИТЕЛЬНО заменяем любой черный цвет на красный
        if (b == 0 and g == 0 and r == 0) or (b < 30 and g < 30 and r < 30):
            # Заменяем на красный цвет
            class_color = (0, 0, 255)  # BGR формат: (0, 0, 255) = красный
            logger.info(f"🎨 БАН-ЛИСТ: черный цвет для класса {class_id} заменен на КРАСНЫЙ: {class_color}")
        else:
            logger.info(f"🎨 НОВЫЙ цвет для класса {class_id}: {class_color}")
        
        CLASS_COLORS[class_id] = class_color
    else:
        logger.info(f"🎨 ИСПОЛЬЗУЮ кэшированный цвет для класса {class_id}: {CLASS_COLORS[class_id]}")
    
    return CLASS_COLORS[class_id]

# Глобальный кэш для моделей
MODEL_CACHE = {}

# Обновленный MODEL_MAP с поддержкой новых типов обработки
MODEL_MAP = {
    'auto_damage_united_with_third_part_yolo_model': 'auto_damage_united_with_third_part_yolo_model.pt',
    'auto_damage_yolo_first_part_model': 'auto_damage_yolo_first_part_model.pt',
    'auto_damage_yolo_second_part_model': 'auto_damage_yolo_second_part_model.pt',
    'auto_damage_yolo_third_part_model': 'auto_damage_yolo_third_part_model.pt',
    'auto_damage_yolo_united_1_2_model': 'auto_damage_yolo_united_1_2_model.pt',
    'auto_parts_full_dataset_1': 'auto_parts_full_dataset_1.pt'
}

# Маппинг типов обработки на модели
PROCESSING_TYPES = {
    'all_models': list(MODEL_MAP.keys()),
    'damage_only': [k for k in MODEL_MAP.keys() if 'damage' in k],
    'parts_only': [k for k in MODEL_MAP.keys() if 'parts' in k],
    'damage_parts': ['auto_damage_united_with_third_part_yolo_model', 'auto_parts_full_dataset_1']
}

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/tmp", StaticFiles(directory="app/tmp"), name="tmp")

templates = Jinja2Templates(directory="app/templates")

def load_model(model_name: str):
    """Загружает модель из кэша или загружает новую"""
    try:
        if model_name in MODEL_CACHE:
            logger.info(f"📦 Модель {model_name} загружена из кэша")
            return MODEL_CACHE[model_name]
        
        pt_file = BASE_DIR / "models_from_hub" / MODEL_MAP[model_name]
        if not pt_file.exists():
            logger.error(f"❌ Модель {model_name} не найдена по пути: {pt_file}")
            logger.error(f"📂 Содержимое директории models_from_hub:")
            models_dir = BASE_DIR / "models_from_hub"
            if models_dir.exists():
                for file in models_dir.iterdir():
                    logger.error(f"   - {file.name}")
            else:
                logger.error(f"   Директория {models_dir} не существует")
            raise FileNotFoundError(f"Файл модели {pt_file} не найден")
        
        logger.info(f"🔄 Загрузка модели {model_name}...")
        start_time = time.time()
        
        # Принудительно используем CPU для сервера
        device = 'cpu'
        logger.info(f"💻 Используется устройство: {device}")
        
        model = YOLO(str(pt_file))
        model.to(device)
        
        # Оптимизация модели для CPU
        model.fuse()
        
        # Дополнительные оптимизации для CPU
        model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"⚡ Модель {model_name} загружена за {load_time:.2f}с")
        
        MODEL_CACHE[model_name] = model
        return model
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели {model_name}: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return None

def optimize_image_size(image_path: Path, max_size: int = 512):
    """Оптимизирует размер изображения для быстрой обработки на CPU"""
    try:
        with Image.open(image_path) as img:
            # Получаем размеры
            width, height = img.size
            
            # Если изображение больше max_size, уменьшаем его
            if max(width, height) > max_size:
                # Вычисляем новые размеры с сохранением пропорций
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                
                # Изменяем размер
                img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Сохраняем с оптимизацией
                img_resized.save(image_path, 'JPEG', quality=80, optimize=True)
                
                logger.info(f"🔧 Изображение {image_path.name} оптимизировано: {width}x{height} -> {new_width}x{new_height}")
            else:
                logger.info(f"🔧 Изображение {image_path.name} уже оптимального размера: {width}x{height}")
                
    except Exception as e:
        logger.error(f"❌ Ошибка оптимизации изображения {image_path}: {str(e)}")

def extract_detection_data(result, model_name):
    """Извлекает данные обнаружения из результата YOLO с масками сегментации"""
    try:
        detections = []
        
        logger.info(f"🔍 Извлечение данных для модели {model_name}")
        logger.info(f"📊 Результат имеет boxes: {hasattr(result, 'boxes')}")
        
        # Проверяем, что result - это объект Results (не список)
        if hasattr(result, 'boxes') and result.boxes is not None:
            boxes = result.boxes
            logger.info(f"📦 Количество боксов: {len(boxes)}")
            
            if len(boxes) > 0:
                # Безопасное получение данных с проверкой типа
                try:
                    # Проверяем, что это тензоры и безопасно конвертируем
                    if hasattr(boxes.xyxy, 'cpu'):
                        xyxy = boxes.xyxy.cpu().numpy()
                    else:
                        xyxy = boxes.xyxy.numpy()
                    
                    if hasattr(boxes.cls, 'cpu'):
                        classes = boxes.cls.cpu().numpy()
                    else:
                        classes = boxes.cls.numpy()
                    
                    if hasattr(boxes.conf, 'cpu'):
                        confidences = boxes.conf.cpu().numpy()
                    else:
                        confidences = boxes.conf.numpy()
                    
                    class_names = result.names
                    
                    # Получаем маски сегментации
                    masks_xy = None
                    masks_xyn = None
                    if hasattr(result, 'masks') and result.masks is not None:
                        # Получаем маски в формате полигонов (списки)
                        masks_xy = result.masks.xy  # полигоны в пикселях
                        masks_xyn = result.masks.xyn  # нормализованные полигоны
                        logger.info(f"🎭 Найдены маски: {len(masks_xy) if masks_xy else 0} масок")
                        if masks_xy and len(masks_xy) > 0:
                            logger.info(f"🎭 Пример маски [0]: тип={type(masks_xy[0])}, длина={len(masks_xy[0]) if hasattr(masks_xy[0], '__len__') else 'N/A'}")
                        if masks_xy and len(masks_xy) > 0:
                            sample_mask = masks_xy[0]
                            logger.info(f"🎭 Пример маски [0]: форма={sample_mask.shape if hasattr(sample_mask, 'shape') else 'N/A'}, первые 6 значений={sample_mask[:6] if hasattr(sample_mask, '__getitem__') else 'N/A'}")
                    
                    for i in range(len(boxes)):
                        class_id = int(classes[i])
                        # Получаем цвет для конкретного класса (как в YOLO)
                        class_color = get_class_color(class_id)
                        
                        detection = {
                            'bbox': [float(x) for x in xyxy[i].tolist()],
                            'class_id': class_id,
                            'class_name': str(class_names[class_id]),
                            'confidence': float(confidences[i]),
                            'model_color': list(class_color)  # Сохраняем цвет класса
                        }
                        
                        logger.info(f"🎨 Класс {class_id} '{class_names[class_id]}' получил цвет {list(class_color)}")
                        
                        # Добавляем маски сегментации если есть
                        if masks_xy is not None and i < len(masks_xy):
                            # Конвертируем маски в списки float
                            # masks_xy[i] - это numpy array с формой (N, 2) где каждая строка [x, y]
                            mask_array = masks_xy[i]
                            if hasattr(mask_array, 'tolist'):
                                mask_list = mask_array.tolist()
                            else:
                                mask_list = list(mask_array)
                            # Преобразуем [[x1,y1], [x2,y2], ...] в [x1,y1,x2,y2,...]
                            detection['mask_polygon'] = [float(coord) for point in mask_list for coord in point]
                            
                            mask_norm_array = masks_xyn[i]
                            if hasattr(mask_norm_array, 'tolist'):
                                mask_norm_list = mask_norm_array.tolist()
                            else:
                                mask_norm_list = list(mask_norm_array)
                            # Преобразуем [[x1,y1], [x2,y2], ...] в [x1,y1,x2,y2,...]
                            detection['mask_polygon_normalized'] = [float(coord) for point in mask_norm_list for coord in point]
                        
                        detections.append(detection)
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка при обработке боксов для модели {model_name}: {str(e)}")
                    logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
                    return []
        
        logger.info(f"✅ Извлечено {len(detections)} детекций для модели {model_name}")
        if detections:
            for i, det in enumerate(detections[:3]):  # Показываем первые 3
                logger.info(f"  {i+1}. {det['class_name']} - {det['confidence']:.2f}")
        
        return detections
        
    except Exception as e:
        logger.error(f"❌ Общая ошибка в extract_detection_data для модели {model_name}: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return []

def combine_detection_results(results_list):
    """Объединяет результаты нескольких моделей на одном изображении с масками сегментации"""
    try:
        if not results_list or len(results_list) == 0:
            return None, []
        
        # Берем первое изображение как основу
        first_result = results_list[0]
        if not first_result:
            return None, []
        
        # Получаем изображение из первого результата
        combined_img = first_result.orig_img.copy()
        
        # Собираем все обнаружения
        all_detections = []
        
        # Цвета для разных моделей
        model_colors = {
            'auto_damage_united_with_third_part_yolo_model': (0, 255, 0),  # Зеленый для повреждений
            'auto_parts_full_dataset_1': (255, 0, 0)  # Красный для деталей
        }
        
        for result_idx, result in enumerate(results_list):
            if result is None:
                continue
                
            # Проверяем, что result - это объект Results (не список)
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                if len(boxes) > 0:
                    try:
                        # Безопасное получение данных с проверкой типа
                        if hasattr(boxes.xyxy, 'cpu'):
                            xyxy = boxes.xyxy.cpu().numpy()
                        else:
                            xyxy = boxes.xyxy.numpy()
                        
                        if hasattr(boxes.cls, 'cpu'):
                            classes = boxes.cls.cpu().numpy()
                        else:
                            classes = boxes.cls.numpy()
                        
                        if hasattr(boxes.conf, 'cpu'):
                            confidences = boxes.conf.cpu().numpy()
                        else:
                            confidences = boxes.conf.numpy()
                        
                        class_names = result.names
                        
                        # Получаем маски сегментации
                        masks_xy = None
                        if hasattr(result, 'masks') and result.masks is not None:
                            masks_xy = result.masks.xy  # полигоны в пикселях
                        
                        for i in range(len(boxes)):
                            class_id = int(classes[i])
                            # Получаем цвет для конкретного класса (как в YOLO)
                            class_color = get_class_color(class_id)
                            
                            detection = {
                                'bbox': [float(x) for x in xyxy[i].tolist()],
                                'class_id': class_id,
                                'class_name': str(class_names[class_id]),
                                'confidence': float(confidences[i]),
                                'model_color': list(class_color)  # Сохраняем цвет класса
                            }
                            
                            logger.info(f"🎨 Mix модель: Класс {class_id} '{class_names[class_id]}' получил цвет {list(class_color)}")
                            
                            # Добавляем маски сегментации если есть
                            if masks_xy is not None and i < len(masks_xy):
                                # Конвертируем маски в списки float
                                # masks_xy[i] - это numpy array с формой (N, 2) где каждая строка [x, y]
                                mask_array = masks_xy[i]
                                if hasattr(mask_array, 'tolist'):
                                    mask_list = mask_array.tolist()
                                else:
                                    mask_list = list(mask_array)
                                # Преобразуем [[x1,y1], [x2,y2], ...] в [x1,y1,x2,y2,...]
                                detection['mask_polygon'] = [float(coord) for point in mask_list for coord in point]
                            
                            all_detections.append(detection)
                            
                    except Exception as e:
                        logger.error(f"❌ Ошибка при обработке боксов в combine_detection_results для result_idx {result_idx}: {str(e)}")
                        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
                        continue
        
        # Используем PIL для рисования текста с поддержкой кириллицы
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        # Конвертируем OpenCV изображение в PIL
        pil_img = Image.fromarray(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        # Пытаемся загрузить шрифт с поддержкой кириллицы
        try:
            # Попробуем найти системный шрифт
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                # Альтернативный шрифт
                font = ImageFont.truetype("DejaVuSans.ttf", 16)
            except:
                # Используем стандартный шрифт
                font = ImageFont.load_default()
        
        # Рисуем все обнаружения
        for detection in all_detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Используем цвет модели
            color = detection.get('model_color', (0, 255, 0))
            
            # Если есть маска сегментации, рисуем её
            if 'mask_polygon' in detection:
                mask_polygon = detection['mask_polygon']
                if len(mask_polygon) > 0:
                    # Конвертируем полигон в формат для PIL
                    polygon_points = []
                    for j in range(0, len(mask_polygon), 2):
                        if j + 1 < len(mask_polygon):
                            polygon_points.append((mask_polygon[j], mask_polygon[j + 1]))
                    
                    if len(polygon_points) > 2:
                        # Рисуем заливку маски с прозрачностью
                        # Создаем временное изображение для маски
                        mask_img = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
                        mask_draw = ImageDraw.Draw(mask_img)
                        
                        # Рисуем полигон с прозрачностью
                        mask_draw.polygon(polygon_points, fill=(color[0], color[1], color[2], 76))  # 30% прозрачность
                        
                        # Накладываем маску на основное изображение
                        pil_img = Image.alpha_composite(pil_img.convert('RGBA'), mask_img).convert('RGB')
                        draw = ImageDraw.Draw(pil_img)
            
            # Рисуем границу
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # Добавляем текст с классом и уверенностью
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            
            # Получаем размер текста
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # Позиционируем текст так, чтобы он всегда был виден
            text_x = x1
            text_y = y1 - text_height - 5
            
            # Если текст выходит за границы изображения, перемещаем его
            img_width, img_height = pil_img.size
            if text_y < 0:
                text_y = y2 + 5  # Помещаем текст под объект
            if text_x + text_width > img_width:
                text_x = img_width - text_width - 5
            if text_x < 0:
                text_x = 5
            
            # Рисуем фон для текста
            draw.rectangle([text_x, text_y, text_x + text_width + 5, text_y + text_height + 5], fill=color)
            
            # Рисуем текст
            draw.text((text_x + 2, text_y + 2), label, fill=(0, 0, 0), font=font)
        
        # Конвертируем обратно в OpenCV формат
        combined_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        return combined_img, all_detections
        
    except Exception as e:
        logger.error(f"❌ Общая ошибка в combine_detection_results: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return None, []

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process")
async def process_images(
    files: list[UploadFile] = File(...),
    model_name: str = Form(...),
    imgsz: int = Form(...),
    conf: float = Form(...),
    iou: float = Form(...)
):
    request_start = time.time()
    logger.info(f"🚀 НАЧАЛО ОБРАБОТКИ: {len(files)} файлов, imgsz={imgsz}, conf={conf}, iou={iou}")
    logger.info(f"📋 Детали запроса:")
    logger.info(f"  - model_name: {model_name}")
    logger.info(f"  - imgsz: {imgsz} (тип: {type(imgsz)})")
    logger.info(f"  - conf: {conf} (тип: {type(conf)})")
    logger.info(f"  - iou: {iou} (тип: {type(iou)})")

    for i, file in enumerate(files):
        logger.info(f"  - файл {i+1}: {file.filename} ({file.content_type}, {file.size} байт)")

    # Определяем какие модели использовать
    if model_name in PROCESSING_TYPES:
        model_names = PROCESSING_TYPES[model_name]
        logger.info(f"🎯 Выбран тип обработки '{model_name}': {len(model_names)} моделей")
    else:
        # Если передано конкретное имя модели
        model_names = [model_name] if model_name in MODEL_MAP else []
        logger.info(f"🎯 Используется конкретная модель: {model_name}")

    if not model_names:
        logger.error(f"❌ Неизвестный тип обработки или модель: {model_name}")
        return JSONResponse(
            status_code=400,
            content={"error": f"Неизвестный тип обработки: {model_name}"}
        )

    # Создаем временную сессию
    session_id = str(uuid.uuid4())
    session_dir = TMP_DIR / session_id
    session_dir.mkdir(exist_ok=True)
    
    # Очищаем кэш цветов для нового запроса
    global CLASS_COLORS
    CLASS_COLORS.clear()
    logger.info(f"🎨 Кэш цветов очищен для новой сессии")
    
    logger.info(f"📁 Создана сессия: {session_id}")

    try:
        # Сохраняем и оптимизируем изображения
        saved_file_paths = []
        logger.info(f"💾 Начинаю сохранение {len(files)} файлов...")
        
        for i, file in enumerate(files):
            logger.info(f"📁 [{i+1}/{len(files)}] Сохранение файла: {file.filename}")
            
            # Читаем содержимое файла
            contents = await file.read()
            logger.info(f"📄 Размер файла {file.filename}: {len(contents)} байт")
            
            # Сохраняем файл
            tmp_file_path = session_dir / file.filename
            with open(tmp_file_path, "wb") as f:
                f.write(contents)
            
            logger.info(f"💾 Файл сохранен: {tmp_file_path}")
            
            # Оптимизируем изображение
            logger.info(f"🔧 Оптимизация изображения: {file.filename}")
            optimize_image_size(tmp_file_path, max_size=512)
            
            saved_file_paths.append(tmp_file_path)

        # Обрабатываем изображения моделями
        results_all_models = {}
        detections_all = {}
        original_images = {}  # Пути к оригинальным изображениям
        
        # Специальная обработка для "Повреждения + детали"
        if model_name == 'damage_parts':
            logger.info("🔄 Специальная обработка: объединение результатов повреждений и деталей")
            
            # Обрабатываем каждое изображение
            for i, file_path in enumerate(saved_file_paths):
                filename = file_path.name
                
                try:
                    # Загружаем модели
                    damage_model = 'auto_damage_united_with_third_part_yolo_model'
                    parts_model = 'auto_parts_full_dataset_1'
                    
                    damage_model_obj = load_model(damage_model)
                    parts_model_obj = load_model(parts_model)
                    
                    # Получаем результаты для одного файла
                    damage_results = damage_model_obj(str(file_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                    parts_results = parts_model_obj(str(file_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                    
                    if damage_results and parts_results:
                        # Объединяем результаты - damage_results и parts_results это списки, берем первый элемент
                        combined_img, combined_detections = combine_detection_results([damage_results[0], parts_results[0]])
                        
                        if combined_img is not None:
                            # Создаем один объединенный результат (без сохранения PNG)
                            combined_result = {
                                "original_filename": filename,
                                "result_path": f"/tmp/{session_id}/{filename}"  # Путь к оригиналу
                            }
                            
                            # Сохраняем только объединенный результат
                            if 'combined' not in results_all_models:
                                results_all_models['combined'] = []
                            results_all_models['combined'].append(combined_result)
                            
                            # Сохраняем объединенные данные обнаружения
                            if filename not in detections_all:
                                detections_all[filename] = {}
                            detections_all[filename]['combined'] = combined_detections
                            
                            # Сохраняем путь к оригинальному изображению
                            if filename not in original_images:
                                original_images[filename] = f"/tmp/{session_id}/{filename}"
                            
                            logger.info(f"✅ Объединен результат для {filename}")
                
                except Exception as e:
                    logger.error(f"❌ Ошибка объединения для {filename}: {str(e)}")
                    logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
                    continue
        else:
            # Обычная обработка для всех остальных типов
            for i, mn in enumerate(model_names):
                model_start = time.time()
                logger.info(f"🤖 [{i+1}/{len(model_names)}] Обработка модели: {mn}")
                
                try:
                    model = load_model(mn)
                    inference_start = time.time()
                    logger.info(f"🔍 [{i}/{len(model_names)}] {mn}: начало инференса...")
                    
                    results = model([str(p) for p in saved_file_paths], conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                    inference_time = time.time() - inference_start
                    logger.info(f"⚡ Модель {mn}: инференс за {inference_time:.2f}с ({inference_time/len(files):.2f}с на изображение)")
                    
                    # Сохраняем результаты
                    save_start = time.time()
                    model_results = []
                    
                    # results - это список объектов Results, по одному для каждого изображения
                    for j, result in enumerate(results):
                        # Сохраняем путь к оригинальному изображению
                        original_filename = saved_file_paths[j].name
                        if original_filename not in original_images:
                            original_images[original_filename] = f"/tmp/{session_id}/{original_filename}"
                        
                        # Извлекаем данные обнаружения
                        detections = extract_detection_data(result, mn)
                        logger.info(f"🔍 Извлечено {len(detections)} детекций для {original_filename} модель {mn}")

                        # Сохраняем результат (без PNG файла)
                        model_result = {
                            "original_filename": original_filename,
                            "result_path": f"/tmp/{session_id}/{original_filename}"  # Путь к оригиналу
                        }
                        model_results.append(model_result)
                        
                        # Сохраняем данные обнаружения
                        if detections:
                            if original_filename not in detections_all:
                                detections_all[original_filename] = {}
                            detections_all[original_filename][mn] = detections
                            logger.info(f"💾 Сохранено {len(detections)} детекций для {original_filename} модель {mn}")
                        else:
                            logger.warning(f"⚠️ Нет детекций для {original_filename} модель {mn}")
                    
                    results_all_models[mn] = model_results
                    
                    save_time = time.time() - save_start
                    model_time = time.time() - model_start
                    logger.info(f"✅ Модель {mn}: инференс {inference_time:.2f}с, сохранение {save_time:.2f}с, всего {model_time:.2f}с")
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка модели {mn}: {str(e)}")
                    logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
                    results_all_models[mn] = []
                    continue

        total_time = time.time() - request_start
        avg_time_per_model = total_time / len(model_names) if model_names else 0
        avg_time_per_image = total_time / len(files) if files else 0

        logger.info(f"🎉 ЗАВЕРШЕНО: {total_time:.2f}с")
        logger.info(f"📊 Статистика: {len(results_all_models)} моделей, {len(files)} изображений")
        logger.info(f"⏱️ Среднее время на модель: {avg_time_per_model:.2f}с")
        logger.info(f"⏱️ Среднее время на изображение: {avg_time_per_image:.2f}с")
        
        # Логируем итоговые данные детекций
        logger.info(f"📋 ИТОГОВЫЕ ДАННЫЕ ДЕТЕКЦИЙ:")
        logger.info(f"  - Всего файлов с детекциями: {len(detections_all)}")
        for filename, models_data in detections_all.items():
            logger.info(f"  - Файл {filename}:")
            for model_name, detections in models_data.items():
                logger.info(f"    - Модель {model_name}: {len(detections)} детекций")
                for i, detection in enumerate(detections[:3]):  # Показываем первые 3
                    logger.info(f"      {i+1}. {detection.get('class_name', 'Unknown')} - {detection.get('confidence', 0):.2f}")

        # Формируем ответ
        response_data = {
            "results": results_all_models,
            "detections": detections_all,
            "original_images": original_images,
            "processing_time": total_time,
            "models_processed": len(model_names),
            "files_processed": len(files)
        }

        return response_data

    except Exception as e:
        logger.error(f"❌ Ошибка обработки: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка обработки: {str(e)}"}
        )

@app.post("/clear_tmp")
async def clear_tmp():
    """Очищает ВСЕ временные файлы"""
    try:
        logger.info("🧹 Начало очистки ВСЕХ временных файлов...")
        
        # Удаляем ВСЕ сессии и файлы в tmp
        cleared_sessions = 0
        cleared_files = 0
        
        for item in TMP_DIR.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                    cleared_sessions += 1
                    logger.info(f"🗑️ Удалена сессия: {item.name}")
                elif item.is_file():
                    item.unlink()
                    cleared_files += 1
                    logger.info(f"🗑️ Удален файл: {item.name}")
            except Exception as e:
                logger.error(f"❌ Ошибка удаления {item.name}: {str(e)}")
                continue
        
        logger.info(f"✅ Очистка завершена. Удалено сессий: {cleared_sessions}, файлов: {cleared_files}")
        return {"message": f"Очищено {cleared_sessions} сессий и {cleared_files} файлов"}
        
    except Exception as e:
        logger.error(f"❌ Ошибка очистки: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка очистки: {str(e)}"}
        )

@app.get("/status")
async def get_status():
    """Возвращает статус приложения"""
    try:
        logger.info("📊 Запрос статуса приложения")
        
        # Проверяем доступность CUDA
        cuda_available = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0
        
        # Подсчитываем временные файлы
        tmp_files_count = sum(1 for _ in TMP_DIR.rglob('*') if _.is_file())
        tmp_sessions_count = sum(1 for _ in TMP_DIR.iterdir() if _.is_dir())
        
        # Информация о кэше моделей
        cached_models = list(MODEL_CACHE.keys())
        
        status_info = {
            "status": "running",
            "cuda_available": cuda_available,
            "cuda_device_count": cuda_device_count,
            "device_used": "cpu",
            "tmp_files_count": tmp_files_count,
            "tmp_sessions_count": tmp_sessions_count,
            "cached_models": cached_models,
            "available_models": list(MODEL_MAP.keys()),
            "processing_types": list(PROCESSING_TYPES.keys())
        }
        
        logger.info(f"📊 Статус: {status_info}")
        return status_info
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статуса: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Ошибка получения статуса: {str(e)}"}
        )

if __name__ == "__main__":
    logger.info("🚀 Запуск приложения...")
    logger.info(f"🐍 Версия PyTorch: {torch.__version__}")
    logger.info(f"🔧 CUDA доступна: {torch.cuda.is_available()}")
    logger.info(f"💻 Используется устройство: CPU")
    logger.info(f"📏 Максимальный размер изображения: 512px")
    logger.info(f"📁 Временная директория: {TMP_DIR}")
    logger.info(f"🎯 Доступные типы обработки: {list(PROCESSING_TYPES.keys())}")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)