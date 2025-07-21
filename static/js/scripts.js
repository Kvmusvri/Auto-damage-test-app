"use strict";

// Константы
const MODEL_DISPLAY_NAMES = {
  'Auto_damage_united_with_third_part_yolo_model': 'Полный датасет 1+2+3 части',
  'Auto_damage_yolo_THIRD_PART_model': 'Датасет только 3 часть',
  'Auto_damage_yolo_united_1_2_model': 'Полный датасет 1+2 части',
  'Auto_damage_yolo_SECOND_PART_model': 'Только 2 часть',
  'Auto_damage_yolo_FIRST_PART_model': 'Только 1 часть'
};

// DOM элементы
const dropArea = document.getElementById('dropArea');
const fileInput = document.getElementById('fileInput');
const gallery = document.getElementById('gallery');
const uploadForm = document.getElementById('uploadForm');
const confRange = document.getElementById('conf');
const confValue = document.getElementById('confValue');
const iouRange = document.getElementById('iou');
const iouValue = document.getElementById('iouValue');
const processingMessage = document.getElementById('processingMessage');
const notification = document.getElementById('notification');
const resultsContainer = document.getElementById('resultsContainer');
const startBtn = document.getElementById('startBtn');
const modal = document.getElementById('modal');
const modalImage = document.getElementById('modalImage');
const closeModal = document.getElementById('closeModal');
const closePanel = document.getElementById('closePanel');
const detectionPanel = document.getElementById('detectionPanel');
const detectionList = document.getElementById('detectionList');

// Глобальные переменные
let files = [];
let allFiles = [];
let currentParams = {};
let detectionData = {}; // Хранение данных обнаружения для каждого изображения
let originalImages = {}; // Хранение оригинальных изображений для перерисовки
let filteredImages = {}; // Хранение отфильтрованных изображений
let currentModalImage = null; // Текущее изображение в модальном окне
let resultImages = {}; // Хранение изображений с результатами

// Инициализация приложения
document.addEventListener('DOMContentLoaded', function() {
  initializeEventListeners();
  initializeRangeInputs();
});

// Инициализация обработчиков событий
function initializeEventListeners() {
  console.log('🔧 Инициализация обработчиков событий...');
  
  // Инициализация drag & drop
  initializeDragAndDrop();
  
  // Обработчик отправки формы
  const form = document.getElementById('uploadForm');
  form.addEventListener('submit', handleFormSubmit);
  
  // Обработчики модального окна
  const modal = document.getElementById('modal');
  const closeModal = document.getElementById('closeModal');
  const detectionPanel = document.getElementById('detectionPanel');
  const closePanel = document.getElementById('closePanel');
  
  closeModal.addEventListener('click', () => {
    modal.classList.remove('active');
    currentModalImage = null; // Очищаем информацию о текущем изображении
    
    // Сбрасываем состояние детекций
    hiddenModalDetections.clear();
    currentModalDetections = [];
    modalDetectionColors.clear();
  });
  
  modal.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.remove('active');
      currentModalImage = null; // Очищаем информацию о текущем изображении
      
      // Сбрасываем состояние детекций
      hiddenModalDetections.clear();
      currentModalDetections = [];
      modalDetectionColors.clear();
    }
  });
  
  closePanel.addEventListener('click', () => {
    detectionPanel.classList.remove('active');
  });
  
  console.log('✅ Обработчики событий инициализированы');
}

// Инициализация слайдеров
function initializeRangeInputs() {
  const confSlider = document.getElementById('conf');
  const iouSlider = document.getElementById('iou');
  const confValue = document.getElementById('confValue');
  const iouValue = document.getElementById('iouValue');

  confSlider.addEventListener('input', (e) => {
    confValue.textContent = e.target.value;
  });

  iouSlider.addEventListener('input', (e) => {
    iouValue.textContent = e.target.value;
  });
}

function handleDragOver(e) {
  e.preventDefault();
}

function handleDrop(e) {
    e.preventDefault();
  const droppedFiles = Array.from(e.dataTransfer.files);
  handleFiles(droppedFiles);
}

function handleFileSelect(e) {
  const selectedFiles = Array.from(e.target.files);
  handleFiles(selectedFiles);
}

// Обработка загруженных файлов
function handleFiles(newFiles) {
  console.log('📁 Обработка файлов:', newFiles.length);
  
  const imageFiles = newFiles.filter(file => file.type.startsWith('image/'));
  
  imageFiles.forEach(file => {
    if (!allFiles.some(existingFile => existingFile.name === file.name)) {
      allFiles.push(file);
      files.push(file);
      console.log('✅ Файл добавлен:', file.name);
    } else {
      console.log('⚠️ Файл уже существует:', file.name);
    }
  });
  
  updateGallery();
}

function updateGallery() {
  const gallery = document.getElementById('gallery');
  gallery.innerHTML = '';
  
  files.forEach((file, index) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'gallery-wrapper';
    
    const img = document.createElement('img');
    img.src = URL.createObjectURL(file);
    img.alt = file.name;
    
    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-btn';
    removeBtn.textContent = '×';
    removeBtn.onclick = () => removeFile(index);
    
    wrapper.appendChild(img);
    wrapper.appendChild(removeBtn);
    gallery.appendChild(wrapper);
  });
}

function removeFile(index) {
  files.splice(index, 1);
  updateGallery();
  
  // Переинициализируем drag & drop после удаления файлов
  initializeDragAndDrop();
}

// Функция для инициализации drag & drop
function initializeDragAndDrop() {
  const dropArea = document.getElementById('dropArea');
  const fileInput = document.getElementById('fileInput');
  
  // Удаляем старые обработчики
  dropArea.removeEventListener('dragover', handleDragOver);
  dropArea.removeEventListener('drop', handleDrop);
  dropArea.removeEventListener('click', () => fileInput.click());
  fileInput.removeEventListener('change', handleFileSelect);
  
  // Добавляем новые обработчики
  dropArea.addEventListener('dragover', handleDragOver);
  dropArea.addEventListener('drop', handleDrop);
  dropArea.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', handleFileSelect);
}

// Обработка отправки формы
async function handleFormSubmit(e) {
    e.preventDefault();
  
  console.log('🚀 НАЧАЛО ОБРАБОТКИ ФОРМЫ');
  console.log('📁 Файлы для обработки:', files.length);
  console.log('📋 Все файлы:', allFiles.length);

  if (files.length === 0) {
    showNotification('Пожалуйста, выберите хотя бы одно изображение');
    return;
  }

  const startBtn = document.getElementById('startBtn');
  startBtn.disabled = true;
  startBtn.textContent = 'Обработка...';

  const processingMessage = document.getElementById('processingMessage');
  processingMessage.style.display = 'block';

  const formData = new FormData();
  
  // Добавляем файлы
  files.forEach(file => {
    formData.append('files', file);
  });

  const modelName = document.getElementById('modelSelect').value;
  const imgsz = document.getElementById('imgsz').value;
  const conf = document.getElementById('conf').value;
  const iou = document.getElementById('iou').value;

  formData.append('model_name', modelName);
  formData.append('imgsz', imgsz);
  formData.append('conf', conf);
  formData.append('iou', iou);

  console.log('⚙️ Параметры запроса:');
  console.log('  - model_name:', modelName);
  console.log('  - imgsz:', imgsz);
  console.log('  - conf:', conf);
  console.log('  - iou:', iou);

  currentParams = {
    imgsz: imgsz,
    conf: conf,
    iou: iou
  };

  const requestStart = Date.now();
  console.log('📡 Отправка запроса на /process...');

  try {
    const response = await fetch('/process', {
      method: 'POST',
      body: formData
    });

    const requestTime = Date.now() - requestStart;
    console.log(`📡 Ответ получен за ${requestTime}мс`);
    console.log('📊 Статус ответа:', response.status, response.statusText);

    const data = await response.json();
    console.log('📦 Данные ответа:', data);

    if (!response.ok) {
      throw new Error(data.error || 'Ошибка обработки');
    }

    console.log('✅ Обработка завершена успешно');
    console.log('📊 Результаты по моделям:', Object.keys(data.results));

    // Сохраняем данные обнаружения
    detectionData = data.detections || {};
    console.log('🔍 Данные детекций:', detectionData);
    
    // Сохраняем оригинальные изображения для перерисовки
    originalImages = data.original_images || {};

    updateResults(data.results);
    processingMessage.style.display = 'none';
    showNotification('Обработка завершена успешно!', 'success');

  } catch (error) {
    console.error('❌ Ошибка при обработке:', error);
    processingMessage.style.display = 'none';
    showNotification('Ошибка при обработке: ' + error.message);
  } finally {
    startBtn.disabled = false;
    startBtn.textContent = 'Начать обработку';
    console.log('🏁 Обработка формы завершена');
  }
}

// Показать уведомление
function showNotification(message, type = 'error') {
  const notification = document.getElementById('notification');
  notification.textContent = message;
  notification.style.display = 'block';
  notification.style.background = type === 'success' ? '#28a745' : '#dc3545';
  
  setTimeout(() => {
    notification.style.display = 'none';
  }, 5000);
}

// Обновление результатов
function updateResults(results) {
  console.log('🔄 Обновление результатов...');
  console.log('📊 Новые результаты:', results);
  
  const container = document.getElementById('resultsContainer');
  container.innerHTML = '';

  if (!results || Object.keys(results).length === 0) {
    container.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 2rem;">Нет результатов для отображения</p>';
    return;
  }

  const table = document.createElement('table');
  
  // Создаем заголовок таблицы
  const thead = document.createElement('thead');
  const headerRow = document.createElement('tr');
  
  // Заголовки для каждой модели
  Object.keys(results).forEach(modelName => {
    const th = document.createElement('th');
    th.textContent = getModelDisplayName(modelName);
    headerRow.appendChild(th);
  });
  
  thead.appendChild(headerRow);
  table.appendChild(thead);
  
  // Создаем тело таблицы
  const tbody = document.createElement('tbody');
  
  // Получаем все уникальные имена файлов
  const allFileNames = new Set();
  Object.values(results).forEach(modelResults => {
    modelResults.forEach(result => {
      allFileNames.add(result.original_filename);
    });
  });
  
  // Создаем строки для каждого файла
  Array.from(allFileNames).sort().forEach(filename => {
    const row = document.createElement('tr');
    
    // Ячейки для каждой модели
    Object.keys(results).forEach(modelName => {
      const td = document.createElement('td');
      const modelResult = results[modelName].find(r => r.original_filename === filename);
      
      if (modelResult && modelResult.result_path) {
        // Создаем контейнер для canvas
        const container = document.createElement('div');
        container.className = 'result-image-container';
        
        // Создаем скрытое оригинальное изображение
        const originalImg = document.createElement('img');
        originalImg.src = originalImages[filename] || `/tmp/${filename}`;
        originalImg.alt = 'Оригинал';
        originalImg.className = 'result-original-image';
        originalImg.style.display = 'none';
        
        // Создаем canvas для отображения результатов
        const canvas = document.createElement('canvas');
        canvas.className = 'result-canvas';
        canvas.dataset.filename = filename;
        canvas.dataset.model = modelName;
        canvas.title = `${getModelDisplayName(modelName)} — ${filename}`;
        
        // Добавляем обработчик клика для открытия модального окна
        canvas.onclick = (e) => {
          e.stopPropagation();
          openModalWithPanel(originalImages[filename] || `/tmp/${filename}`, filename, modelName);
        };
        
        const paramsDiv = document.createElement('div');
        paramsDiv.className = 'image-params';
        paramsDiv.textContent = `imgsz: ${currentParams.imgsz}, conf: ${currentParams.conf}, iou: ${currentParams.iou}`;
        
        container.appendChild(originalImg);
        container.appendChild(canvas);
        container.appendChild(paramsDiv);
        
        td.appendChild(container);
        
        // Инициализируем canvas после добавления в DOM
        setTimeout(() => {
          initializeResultCanvas(canvas, originalImg, filename, modelName);
        }, 100);
      } else {
        td.className = 'empty-cell';
        td.textContent = 'Нет результата';
      }
      
      row.appendChild(td);
    });
    
    tbody.appendChild(row);
  });
  
  table.appendChild(tbody);
  container.appendChild(table);
}

function getModelDisplayName(modelName) {
  const displayNames = {
    'auto_damage_united_with_third_part_yolo_model': 'Повреждения (полная модель)',
    'auto_damage_yolo_first_part_model': 'Повреждения (часть 1)',
    'auto_damage_yolo_second_part_model': 'Повреждения (часть 2)',
    'auto_damage_yolo_third_part_model': 'Повреждения (часть 3)',
    'auto_damage_yolo_united_1_2_model': 'Повреждения (1+2 части)',
    'auto_parts_full_dataset_1': 'Детали (полная модель)'
  };
  
  return displayNames[modelName] || modelName;
}

function openModal(imageSrc, filename, modelName) {
  const modal = document.getElementById('modal');
  const modalImage = document.getElementById('modalImage');
  const modalDetectionList = document.getElementById('modalDetectionList');
  
  modalImage.src = imageSrc;
  
  // Показываем детекции в модальном окне
  showModalDetections(filename, modelName);
  
  modal.classList.add('active');
  
  // Сохраняем информацию о текущем изображении в модальном окне
  currentModalImage = {
    filename: filename,
    modelName: modelName,
    src: imageSrc
  };
}

function openModalWithPanel(imageSrc, filename, modelName) {
  openModal(imageSrc, filename, modelName);
}

function rgbToHex(r, g, b) {
  return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Глобальные переменные для управления детекциями в модальном окне
let currentModalDetections = [];
let hiddenModalDetections = new Set();
let modalDetectionColors = new Map(); // Кэш цветов для стабильности



function showModalDetections(filename, modelName) {
  console.log(`Показ детекций в модальном окне для: ${filename}, ${modelName}`);

  const modalDetectionList = document.getElementById('modalDetectionList');

  // Получаем детекции из данных
  const detections = detectionData?.[filename]?.[modelName] || [];
  console.log('🎯 Детекции из detectionData:', detections);

  // Сохраняем детекции глобально и сбрасываем скрытые
  currentModalDetections = detections;
  hiddenModalDetections.clear();
  
  // Создаем стабильные цвета для детекций (сохраняем оригинальные из backend)
  modalDetectionColors.clear();
  detections.forEach((detection, index) => {
    let color = [0, 255, 0]; // По умолчанию зеленый
    if (detection.model_color && Array.isArray(detection.model_color)) {
      // Сохраняем оригинальный цвет из backend
      color = [...detection.model_color]; // Копируем массив
    }
    modalDetectionColors.set(index, color);
    console.log(`🎨 Сохранен цвет для детекции ${index}: ${detection.class_name} - [${color.join(', ')}]`);
  });

  // Очищаем список
  modalDetectionList.innerHTML = '';

  if (detections.length === 0) {
    modalDetectionList.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 1rem;">Нет обнаруженных объектов</p>';
    return;
  }

  // Создаем элементы для каждого обнаружения
  detections.forEach((detection, index) => {
    const detectionButton = document.createElement('button');
    detectionButton.className = 'modal-detection-button';
    detectionButton.dataset.index = index;
    detectionButton.type = 'button';

    // Определяем цвет из кэша
    let color = '#00FF00'; // По умолчанию зеленый
    const cachedColor = modalDetectionColors.get(index);
    if (cachedColor) {
      color = rgbToHex(cachedColor[0], cachedColor[1], cachedColor[2]);
    }

    detectionButton.innerHTML = `
      <div class="modal-detection-color" style="background-color: ${color};"></div>
      <div class="modal-detection-info">
        <div class="modal-detection-label">${detection.class_name}</div>
        <div class="modal-detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
      </div>
    `;

    // Добавляем обработчик клика
    detectionButton.addEventListener('click', () => {
      console.log('🖱️ Клик по кнопке детекции:', index, detection.class_name);
      toggleModalDetection(index, detectionButton);
    });

    modalDetectionList.appendChild(detectionButton);
  });

  // Инициализируем canvas после создания кнопок
  setTimeout(() => {
    initializeModalCanvas();
  }, 100);
}

function toggleModalDetection(index, detectionButton) {
  console.log('🔄 Переключение кнопки детекции:', index);
  
  if (detectionButton.classList.contains('hidden')) {
    // Показываем детекцию
    detectionButton.classList.remove('hidden');
    hiddenModalDetections.delete(index);
    console.log('👁️ Показываем детекцию:', index);
    
    // Добавляем детекцию на canvas
    toggleModalDetectionOnCanvas(index, true);
  } else {
    // Скрываем детекцию
    detectionButton.classList.add('hidden');
    hiddenModalDetections.add(index);
    console.log('🙈 Скрываем детекцию:', index);
    
    // Удаляем детекцию с canvas
    toggleModalDetectionOnCanvas(index, false);
  }
}

function initializeModalCanvas() {
  const canvas = document.getElementById('modalCanvas');
  const imgElement = document.getElementById('modalImage');
  
  if (!imgElement || !canvas) {
    console.error('❌ Не найден canvas или изображение');
    return;
  }

  console.log('🎨 Инициализация canvas...');

  // Ждем загрузки изображения
  if (imgElement.complete && imgElement.naturalWidth > 0) {
    setupCanvas();
  } else {
    imgElement.onload = setupCanvas;
  }

  function setupCanvas() {
    console.log('🎨 Настройка canvas...');
    const ctx = canvas.getContext('2d');
    const imgRect = imgElement.getBoundingClientRect();
    
    console.log('📐 Размеры изображения:', imgRect.width, 'x', imgRect.height);
    
    canvas.width = imgRect.width;
    canvas.height = imgRect.height;
    
    // Копируем изображение на canvas
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
    
    console.log('✅ Canvas настроен, рисуем детекции...');
    
    // Рисуем все видимые детекции
    drawDetectionsOnCanvas();
  }
}

function redrawModalCanvas() {
  const canvas = document.getElementById('modalCanvas');
  const imgElement = document.getElementById('modalImage');
  
  if (!canvas || !imgElement) {
    console.error('❌ Не найден canvas или изображение для перерисовки');
    return;
  }

  console.log('🎨 Перерисовка canvas...');

  const ctx = canvas.getContext('2d');
  
  // Очищаем canvas
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  // Копируем изображение
  ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
  
  // Рисуем только видимые детекции
  drawDetectionsOnCanvas();
}

// Функция для добавления/удаления конкретной детекции без перерисовки всего
function toggleModalDetectionOnCanvas(index, show) {
  const canvas = document.getElementById('modalCanvas');
  const imgElement = document.getElementById('modalImage');
  
  if (!canvas || !imgElement || !currentModalDetections[index]) {
    return;
  }

  const ctx = canvas.getContext('2d');
  const detection = currentModalDetections[index];
  
  if (show) {
    // Добавляем детекцию обратно
    console.log(`🎨 Добавляем детекцию ${index}: ${detection.class_name}`);
    drawSingleDetection(ctx, detection, index);
  } else {
    // Удаляем детекцию - перерисовываем всю область
    console.log(`🎨 Удаляем детекцию ${index}: ${detection.class_name}`);
    
    // Очищаем весь canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Перерисовываем изображение
    ctx.drawImage(imgElement, 0, 0, canvas.width, canvas.height);
    
    // Перерисовываем все видимые детекции в правильном порядке
    currentModalDetections.forEach((det, idx) => {
      if (!hiddenModalDetections.has(idx)) {
        drawSingleDetection(ctx, det, idx);
      }
    });
  }
}

function drawSingleDetection(ctx, detection, index) {
  const bbox = detection.bbox;
  if (!bbox || bbox.length !== 4) {
    return;
  }

  // Получаем стабильный цвет детекции из кэша
  let color = modalDetectionColors.get(index) || [0, 255, 0];
  const rgbColor = `rgb(${color[2]}, ${color[1]}, ${color[0]})`;

  const [x1, y1, x2, y2] = bbox;
  const width = x2 - x1;
  const height = y2 - y1;

  // Рисуем маску ПЕРЕД рамкой (если есть)
  if (detection.mask_polygon && detection.mask_polygon.length > 0) {
    ctx.fillStyle = rgbColor + 'B3'; // 70% прозрачности (B3 = 179 в hex, примерно 70%)
    ctx.beginPath();
    
    for (let i = 0; i < detection.mask_polygon.length; i += 2) {
      const x = detection.mask_polygon[i];
      const y = detection.mask_polygon[i + 1];
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    }
    
    ctx.closePath();
    ctx.fill();
  }

  // Рисуем bounding box (рамку) - тот же цвет что и маска
  ctx.strokeStyle = rgbColor;
  ctx.lineWidth = 3; // Делаем рамку толще для лучшей видимости
  ctx.strokeRect(x1, y1, width, height);

  // Рисуем текст с фоном для лучшей читаемости
  const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
  ctx.font = 'bold 14px Arial'; // Делаем текст жирным
  const textMetrics = ctx.measureText(label);
  const textWidth = textMetrics.width;
  const textHeight = 16;
  
  // Рисуем фон для текста
  ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
  ctx.fillRect(x1, y1 - textHeight - 8, textWidth + 6, textHeight + 4);
  
  // Рисуем текст
  ctx.fillStyle = rgbColor;
  ctx.fillText(label, x1 + 3, y1 - 6);
  
  console.log(`🎨 Нарисована детекция ${index}: ${detection.class_name} цвет [${color.join(', ')}]`);
}



function drawDetectionsOnCanvas() {
  const canvas = document.getElementById('modalCanvas');
  
  if (!canvas || !currentModalDetections.length) {
    console.log('❌ Нет canvas или детекций для рисования');
    return;
  }

  const ctx = canvas.getContext('2d');
  console.log('🎨 Рисование детекций на canvas...');

  let drawnCount = 0;
  currentModalDetections.forEach((detection, index) => {
    // Пропускаем скрытые детекции
    if (hiddenModalDetections.has(index)) {
      console.log(`🙈 Пропускаем скрытую детекцию ${index}: ${detection.class_name}`);
      return;
    }

    const bbox = detection.bbox;
    if (!bbox || bbox.length !== 4) {
      console.log(`❌ Некорректный bbox для детекции ${index}`);
      return;
    }

    // Получаем стабильный цвет детекции из кэша
    let color = modalDetectionColors.get(index) || [0, 255, 0];
    
    // Конвертируем BGR в RGB для canvas
    const rgbColor = `rgb(${color[2]}, ${color[1]}, ${color[0]})`;

    // Рисуем маску ПЕРЕД рамкой (чтобы рамка была поверх)
    if (detection.mask_polygon && detection.mask_polygon.length > 0) {
      ctx.fillStyle = rgbColor + 'B3'; // 70% прозрачности
      ctx.beginPath();
      
      for (let i = 0; i < detection.mask_polygon.length; i += 2) {
        const x = detection.mask_polygon[i];
        const y = detection.mask_polygon[i + 1];
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.closePath();
      ctx.fill();
    }

    // Рисуем bounding box
    const [x1, y1, x2, y2] = bbox;
    const width = x2 - x1;
    const height = y2 - y1;

    ctx.strokeStyle = rgbColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1, y1, width, height);

    // Рисуем текст с фоном для лучшей читаемости
    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
    
    // Измеряем размер текста
    ctx.font = '14px Arial';
    const textMetrics = ctx.measureText(label);
    const textWidth = textMetrics.width;
    const textHeight = 14;
    
    // Рисуем фон для текста
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(x1, y1 - textHeight - 5, textWidth + 4, textHeight + 2);
    
    // Рисуем текст
    ctx.fillStyle = rgbColor;
    ctx.fillText(label, x1 + 2, y1 - 5);

    drawnCount++;
    console.log(`✅ Нарисована детекция ${index}: ${detection.class_name}`);
  });

  console.log(`🎨 Всего нарисовано детекций: ${drawnCount}`);
}

function initializeResultCanvas(canvas, originalImg, filename, modelName) {
  console.log(`🎨 Инициализация canvas для ${filename}, ${modelName}`);
  
  if (!canvas || !originalImg) {
    console.error('❌ Не найден canvas или оригинальное изображение');
    return;
  }

  // Ждем загрузки изображения
  if (originalImg.complete && originalImg.naturalWidth > 0) {
    setupResultCanvas();
  } else {
    originalImg.onload = setupResultCanvas;
  }

  function setupResultCanvas() {
    console.log(`🎨 Настройка canvas для ${filename}...`);
    const ctx = canvas.getContext('2d');
    
    // Получаем размеры изображения
    const imgWidth = originalImg.naturalWidth;
    const imgHeight = originalImg.naturalHeight;
    
    // Вычисляем размеры для отображения (максимум 300x300)
    const maxSize = 300;
    let displayWidth = imgWidth;
    let displayHeight = imgHeight;
    
    if (imgWidth > maxSize || imgHeight > maxSize) {
      const ratio = Math.min(maxSize / imgWidth, maxSize / imgHeight);
      displayWidth = imgWidth * ratio;
      displayHeight = imgHeight * ratio;
    }
    
    // Устанавливаем размеры canvas
    canvas.width = displayWidth;
    canvas.height = displayHeight;
    
    console.log(`📐 Размеры canvas: ${displayWidth} x ${displayHeight}`);
    
    // Копируем изображение на canvas
    ctx.drawImage(originalImg, 0, 0, displayWidth, displayHeight);
    
    // Рисуем детекции
    drawResultDetections(canvas, filename, modelName, imgWidth, imgHeight, displayWidth, displayHeight);
  }
}

function drawResultDetections(canvas, filename, modelName, originalWidth, originalHeight, displayWidth, displayHeight) {
  const ctx = canvas.getContext('2d');
  const detections = detectionData?.[filename]?.[modelName] || [];
  
  if (!detections.length) {
    console.log(`❌ Нет детекций для ${filename}, ${modelName}`);
    return;
  }
  
  console.log(`🎨 Рисование ${detections.length} детекций для ${filename}, ${modelName}`);
  
  // Вычисляем масштаб
  const scaleX = displayWidth / originalWidth;
  const scaleY = displayHeight / originalHeight;
  
  detections.forEach((detection, index) => {
    const bbox = detection.bbox;
    if (!bbox || bbox.length !== 4) {
      console.log(`❌ Некорректный bbox для детекции ${index}`);
      return;
    }

    // Получаем цвет детекции
    let color = [0, 255, 0]; // По умолчанию зеленый
    if (detection.model_color && Array.isArray(detection.model_color)) {
      color = detection.model_color;
    }

    // Конвертируем BGR в RGB для canvas
    const rgbColor = `rgb(${color[2]}, ${color[1]}, ${color[0]})`;

    // Масштабируем координаты
    const [x1, y1, x2, y2] = bbox.map((coord, i) => 
      i % 2 === 0 ? coord * scaleX : coord * scaleY
    );
    
    const width = x2 - x1;
    const height = y2 - y1;

    // Рисуем bounding box
    ctx.strokeStyle = rgbColor;
    ctx.lineWidth = Math.max(1, 2 * Math.min(scaleX, scaleY));
    ctx.strokeRect(x1, y1, width, height);

    // Рисуем маску если есть
    if (detection.mask_polygon && detection.mask_polygon.length > 0) {
      ctx.fillStyle = rgbColor + 'B3'; // 70% прозрачности
      ctx.beginPath();
      
      for (let i = 0; i < detection.mask_polygon.length; i += 2) {
        const x = detection.mask_polygon[i] * scaleX;
        const y = detection.mask_polygon[i + 1] * scaleY;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      
      ctx.closePath();
      ctx.fill();
    }

    // Рисуем текст (уменьшенный размер для маленьких изображений)
    const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
    ctx.fillStyle = rgbColor;
    ctx.font = `${Math.max(10, 14 * Math.min(scaleX, scaleY))}px Arial`;
    ctx.fillText(label, x1, y1 - 5);

    console.log(`✅ Нарисована детекция ${index}: ${detection.class_name}`);
  });
  
  console.log(`🎨 Всего нарисовано детекций: ${detections.length}`);
}

function openDetectionPanel(filename, modelName) {
  console.log(`Открытие панели обнаружения для: ${filename}, ${modelName}`);
  
  const detectionList = document.getElementById('detectionList');
  const detections = detectionData[filename]?.[modelName] || [];
  
  if (detections.length === 0) {
    detectionList.innerHTML = '<p style="text-align: center; color: #6c757d; padding: 1rem;">Нет обнаруженных объектов</p>';
    return;
  }
  
  // Очищаем список
  detectionList.innerHTML = '';
  
  // Инициализируем фильтр для этого изображения
  if (!filteredImages[filename]) {
    filteredImages[filename] = {};
  }
  if (!filteredImages[filename][modelName]) {
    // По умолчанию все обнаружения видимы
    filteredImages[filename][modelName] = new Set(detections.map((_, i) => i));
  }
  
  // Определяем цвет модели
  let modelColor = '#00FF00'; // По умолчанию зеленый
  if (modelName.includes('damage')) {
    modelColor = '#00FF00'; // Зеленый для повреждений
  } else if (modelName.includes('parts')) {
    modelColor = '#FF0000'; // Красный для деталей
  } else if (modelName === 'combined') {
    // Для комбинированной модели используем цвета из данных
    modelColor = '#00FF00'; // По умолчанию зеленый
  }
  
  // Создаем элементы для каждого обнаружения
  detections.forEach((detection, index) => {
    const detectionItem = document.createElement('div');
    detectionItem.className = 'detection-item';
    
    const isVisible = filteredImages[filename][modelName].has(index);
    
    // Используем цвет из данных обнаружения или цвет модели
    let color = modelColor;
    if (detection.model_color) {
      if (Array.isArray(detection.model_color)) {
        // RGB кортеж [r, g, b]
        color = rgbToHex(detection.model_color[0], detection.model_color[1], detection.model_color[2]);
      } else if (typeof detection.model_color === 'string') {
        color = detection.model_color;
      }
    }
    
    detectionItem.innerHTML = `
      <div class="detection-color" style="background-color: ${color};"></div>
      <div class="detection-info">
        <div class="detection-label">${detection.class_name}</div>
        <div class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</div>
      </div>
      <div class="detection-toggle">
        <input type="checkbox" ${isVisible ? 'checked' : ''} 
               onchange="toggleDetection('${filename}', '${modelName}', ${index}, this.checked)">
      </div>
    `;
    
    detectionList.appendChild(detectionItem);
  });
  
  // Показываем панель
  const detectionPanel = document.getElementById('detectionPanel');
  detectionPanel.classList.add('active');
}

function getDetectionColor(index) {
  const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
  ];
  return colors[index % colors.length];
}

function toggleDetection(filename, modelName, detectionIndex, visible) {
  console.log(`Переключение обнаружения: ${filename}, ${modelName}, индекс ${detectionIndex}, видимость: ${visible}`);
  
  // Получаем данные обнаружения
  const detections = detectionData[filename]?.[modelName] || [];
  
  if (detections.length === 0) {
    console.log('Нет данных обнаружения для фильтрации');
    return;
  }
  
  // Создаем или обновляем фильтр для этого изображения
  if (!filteredImages[filename]) {
    filteredImages[filename] = {};
  }
  if (!filteredImages[filename][modelName]) {
    // Инициализируем фильтр со всеми видимыми обнаружениями
    filteredImages[filename][modelName] = new Set(detections.map((_, i) => i));
  }
  
  // Обновляем фильтр
  if (visible) {
    filteredImages[filename][modelName].add(detectionIndex);
  } else {
    filteredImages[filename][modelName].delete(detectionIndex);
  }
  
  // Перерисовываем изображение
  redrawFilteredImage(filename, modelName);
}

function redrawFilteredImage(filename, modelName) {
  // Получаем изображение с результатами
  const resultImagePath = resultImages[filename]?.[modelName];
  if (!resultImagePath) {
    console.log('Нет изображения с результатами для перерисовки');
    return;
  }
  
  // Получаем данные обнаружения
  const detections = detectionData[filename]?.[modelName] || [];
  const visibleDetections = filteredImages[filename]?.[modelName] || new Set();
  
  // Если все обнаружения видимы, показываем оригинальное изображение с результатами
  if (visibleDetections.size === detections.length) {
    updateModalAndTableImages(filename, modelName, resultImagePath);
    return;
  }
  
  // Если нет видимых обнаружений, показываем оригинальное изображение
  if (visibleDetections.size === 0) {
    const originalImagePath = originalImages[filename];
    if (originalImagePath) {
      updateModalAndTableImages(filename, modelName, originalImagePath);
    }
    return;
  }
  
  // Создаем canvas для перерисовки
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  
  // Загружаем оригинальное изображение (без результатов)
  const originalImg = new Image();
  originalImg.onload = function() {
    canvas.width = originalImg.width;
    canvas.height = originalImg.height;
    
    // Рисуем оригинальное изображение
    ctx.drawImage(originalImg, 0, 0);
    
    // Рисуем только видимые обнаружения
    detections.forEach((detection, index) => {
      if (visibleDetections.has(index)) {
        const bbox = detection.bbox;
        const x1 = bbox[0];
        const y1 = bbox[1];
        const x2 = bbox[2];
        const y2 = bbox[3];
        
        // Используем оригинальный цвет модели
        let color = '#00FF00'; // По умолчанию зеленый
        if (detection.model_color) {
          if (Array.isArray(detection.model_color)) {
            // RGB кортеж [r, g, b]
            color = rgbToHex(detection.model_color[0], detection.model_color[1], detection.model_color[2]);
          } else if (typeof detection.model_color === 'string') {
            color = detection.model_color;
          }
        }
        
        // Если есть маска сегментации, рисуем её
        if (detection.mask_polygon) {
          const maskPolygon = detection.mask_polygon;
          if (maskPolygon.length > 0) {
            // Рисуем полигон маски
            ctx.beginPath();
            for (let j = 0; j < maskPolygon.length; j += 2) {
              if (j === 0) {
                ctx.moveTo(maskPolygon[j], maskPolygon[j + 1]);
              } else {
                ctx.lineTo(maskPolygon[j], maskPolygon[j + 1]);
              }
            }
            ctx.closePath();
            
            // Рисуем заливку маски с прозрачностью
            ctx.fillStyle = color + '4D'; // 30% прозрачность
            ctx.fill();
          }
        }
        
        // Рисуем границу
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Добавляем текст с фоном
        const label = `${detection.class_name} ${(detection.confidence * 100).toFixed(1)}%`;
        
        // Измеряем размер текста
        ctx.font = '12px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 12;
        
        // Позиционируем текст так, чтобы он всегда был виден
        let textX = x1;
        let textY = y1 - 5;
        
        // Если текст выходит за границы изображения, перемещаем его
        if (textY < textHeight) {
          textY = y2 + textHeight + 5; // Помещаем текст под объект
        }
        if (textX + textWidth > originalImg.width) {
          textX = originalImg.width - textWidth - 5;
        }
        if (textX < 0) {
          textX = 5;
        }
        
        // Рисуем фон для текста
        ctx.fillStyle = color;
        ctx.fillRect(textX, textY - textHeight, textWidth + 4, textHeight + 2);
        
        // Рисуем текст
        ctx.fillStyle = '#000000';
        ctx.fillText(label, textX + 2, textY - 2);
      }
    });
    
    // Обновляем изображения
    updateModalAndTableImages(filename, modelName, canvas.toDataURL());
  };
  
  originalImg.crossOrigin = 'anonymous';
  originalImg.src = originalImages[filename];
}

function updateModalAndTableImages(filename, modelName, imageSrc) {
  // Обновляем изображение в модальном окне
  const modalImage = document.getElementById('modalImage');
  if (modalImage && currentModalImage && 
      currentModalImage.filename === filename && 
      currentModalImage.modelName === modelName) {
    modalImage.src = imageSrc;
    // Обновляем src в currentModalImage
    currentModalImage.src = imageSrc;
  }
  
  // Обновляем изображение в таблице результатов
  updateResultImage(filename, modelName, imageSrc);
}

function updateResultImage(filename, modelName, newImageSrc) {
  // Находим изображение в таблице результатов и обновляем его
  const resultImages = document.querySelectorAll(`img[alt*="${filename}"][alt*="${getModelDisplayName(modelName)}"]`);
  resultImages.forEach(img => {
    img.src = newImageSrc;
  });
}
