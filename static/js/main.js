// main.js

document.addEventListener('DOMContentLoaded',
    function () {

        // ── Image preview on upload ──
        const fileInput = document.getElementById(
            'ecg-file-input'
        );
        if (fileInput) {
            fileInput.addEventListener(
                'change', function (e) {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function (ev) {
                            const preview = document
                                .getElementById('preview-img');
                            const container = document
                                .getElementById(
                                    'preview-container');
                            if (preview && container) {
                                preview.src = ev.target.result;
                                container.style.display = 'block';
                            }
                        };
                        reader.readAsDataURL(file);

                        // Update upload text
                        const uploadText = document
                            .getElementById('upload-text');
                        if (uploadText) {
                            uploadText.textContent =
                                `Selected: ${file.name}`;
                        }
                    }
                });
        }

        // ── Drag and drop ──
        const uploadArea = document
            .querySelector('.upload-area');
        if (uploadArea) {
            uploadArea.addEventListener(
                'dragover', (e) => {
                    e.preventDefault();
                    uploadArea.style.background = '#d0f0ff';
                });
            uploadArea.addEventListener(
                'dragleave', () => {
                    uploadArea.style.background = '#f8fdff';
                });
            uploadArea.addEventListener(
                'drop', (e) => {
                    e.preventDefault();
                    uploadArea.style.background = '#f8fdff';
                    const files = e.dataTransfer.files;
                    if (files.length && fileInput) {
                        fileInput.files = files;
                        fileInput.dispatchEvent(
                            new Event('change')
                        );
                    }
                });
        }

        // ── Animate probability bars ──
        const bars = document.querySelectorAll(
            '.prob-fill, .progress-fill'
        );
        bars.forEach(bar => {
            const w = bar.getAttribute('data-width');
            if (w) {
                setTimeout(() => {
                    bar.style.width = w + '%';
                }, 400);
            }
        });

        // ── Loading on form submit ──
        const form = document.querySelector(
            '.upload-form'
        );
        if (form) {
            form.addEventListener('submit', () => {
                const btn = form.querySelector(
                    '.btn-analyze'
                );
                if (btn) {
                    btn.textContent = '🔄 Analyzing...';
                    btn.disabled = true;
                }
            });
        }

        // ── Counter animation ──
        const counters = document.querySelectorAll(
            '.counter'
        );
        counters.forEach(counter => {
            const target = parseFloat(
                counter.getAttribute('data-target')
            );
            const isFloat = target % 1 !== 0;
            const duration = 2000;
            const steps = 60;
            const increment = target / steps;
            let current = 0;
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    counter.textContent = isFloat
                        ? target.toFixed(1)
                        : Math.floor(target);
                    clearInterval(timer);
                } else {
                    counter.textContent = isFloat
                        ? current.toFixed(1)
                        : Math.floor(current);
                }
            }, duration / steps);
        });
    });