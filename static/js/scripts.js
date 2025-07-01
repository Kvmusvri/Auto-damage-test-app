document.getElementById('fileSelect').addEventListener('click', () => {
    document.getElementById('fileElem').click();
});

const dropArea = document.getElementById('drop-area');
const fileElem = document.getElementById('fileElem');
const gallery = document.getElementById('gallery');

dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.classList.add('highlight');
});
dropArea.addEventListener('dragleave', () => dropArea.classList.remove('highlight'));
dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.classList.remove('highlight');
    handleFiles(e.dataTransfer.files);
});
fileElem.addEventListener('change', e => {
    handleFiles(e.target.files);
});

function handleFiles(files) {
    [...files].forEach(previewFile);
}

function previewFile(file) {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onloadend = () => {
        const img = document.createElement('img');
        img.src = reader.result;
        gallery.appendChild(img);
    };
}

// Синхронизация ползунков и числовых полей
['conf', 'iou'].forEach(param => {
    const slider = document.getElementById(param);
    const input = document.getElementById(param + '-val');

    slider.addEventListener('input', () => input.value = slider.value);
    input.addEventListener('input', () => slider.value = input.value);
});
