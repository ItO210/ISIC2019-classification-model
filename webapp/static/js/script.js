document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const dropZone = document.getElementById("dropZone");
    const dropPlaceholder = document.getElementById("dropPlaceholder");
    const previewWrap = document.getElementById("previewWrap");
    const preview = document.getElementById("preview");
    const uploadBtn = document.getElementById("uploadBtn");
    const changeBtn = document.getElementById("changeBtn");
    const submitBtn = document.getElementById("submitBtn");
    const urlToggle = document.getElementById("urlToggle");
    const urlPanelClose = document.getElementById("urlPanelClose");
    const urlInputWrap = document.getElementById("urlInputWrap");
    const urlInput = document.getElementById("urlInput");
    const urlConfirm = document.getElementById("urlConfirm");
    const imageUrlHidden = document.getElementById("imageUrlHidden");
    const form = document.getElementById("mainForm");
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightboxImg");
    const lightboxClose = document.getElementById("lightboxClose");
    const lightboxBackdrop = document.getElementById("lightboxBackdrop");
    const analyzeOverlay = document.getElementById("analyzeOverlay");

    function showPreview(src) {
        if (!preview || !previewWrap || !dropPlaceholder || !dropZone) return;
        preview.src = src;
        previewWrap.style.display = "flex";
        dropPlaceholder.style.display = "none";
        dropZone.classList.add("has-image");
        dropZone.classList.remove("is-preview-new");
        void dropZone.offsetWidth;
        dropZone.classList.add("is-preview-new");
    }

    function resetDropZone() {
        if (!previewWrap || !dropPlaceholder || !dropZone || !fileInput || !imageUrlHidden) return;
        previewWrap.style.display = "none";
        dropPlaceholder.style.display = "flex";
        dropZone.classList.remove("has-image");
        dropZone.classList.remove("is-preview-new");
        fileInput.value = "";
        imageUrlHidden.value = "";
        if (urlInputWrap) urlInputWrap.classList.remove("visible");
        if (urlInput) urlInput.value = "";
    }

    function openLightbox(src) {
        if (!lightbox || !lightboxImg || !src) return;
        lightboxImg.src = src;
        lightbox.hidden = false;
        lightbox.setAttribute("aria-hidden", "false");
        document.body.style.overflow = "hidden";
    }

    function closeLightbox() {
        if (!lightbox || !lightboxImg) return;
        lightbox.hidden = true;
        lightbox.setAttribute("aria-hidden", "true");
        lightboxImg.removeAttribute("src");
        document.body.style.overflow = "";
    }

    function loadFile(file) {
        if (!file || !file.type.startsWith("image/")) return;
        if (imageUrlHidden) imageUrlHidden.value = "";
        const reader = new FileReader();
        reader.onload = (e) => showPreview(e.target.result);
        reader.readAsDataURL(file);
    }

    if (uploadBtn && fileInput) {
        uploadBtn.addEventListener("click", () => fileInput.click());
    }

    if (changeBtn && fileInput) {
        changeBtn.addEventListener("click", () => {
            resetDropZone();
            fileInput.click();
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files[0]) loadFile(fileInput.files[0]);
        });
    }

    if (dropZone) {
        dropZone.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropZone.classList.add("drag-over");
        });

        ["dragleave", "dragend"].forEach((evt) =>
            dropZone.addEventListener(evt, () => dropZone.classList.remove("drag-over"))
        );

        dropZone.addEventListener("drop", (e) => {
            e.preventDefault();
            dropZone.classList.remove("drag-over");
            const file = e.dataTransfer.files[0];
            if (file && fileInput) {
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                loadFile(file);
            }
        });
    }

    document.addEventListener("paste", (e) => {
        const items = e.clipboardData?.items;
        if (!items || !fileInput) return;
        for (const item of items) {
            if (item.type.startsWith("image/")) {
                const file = item.getAsFile();
                if (!file) continue;
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                loadFile(file);
                break;
            }
        }
    });

    if (urlToggle && urlInputWrap) {
        urlToggle.addEventListener("click", (e) => {
            e.stopPropagation();
            e.preventDefault();
            urlInputWrap.classList.toggle("visible");
            if (urlInputWrap.classList.contains("visible") && urlInput) {
                urlInput.focus();
            }
        });
    }

    if (urlPanelClose && urlInputWrap) {
        urlPanelClose.addEventListener("click", (e) => {
            e.stopPropagation();
            urlInputWrap.classList.remove("visible");
            if (urlInput) urlInput.value = "";
        });
    }

    function loadFromUrl() {
        if (!urlInput || !imageUrlHidden || !fileInput) return;
        const url = urlInput.value.trim();
        if (!url) return;

        const testImg = new Image();
        testImg.onload = () => {
            imageUrlHidden.value = url;
            fileInput.value = "";
            showPreview(url);
        };
        testImg.onerror = () => {
            imageUrlHidden.value = url;
            fileInput.value = "";
            showPreview(url);
        };
        testImg.src = url;
    }

    if (urlConfirm) {
        urlConfirm.addEventListener("click", loadFromUrl);
    }
    if (urlInput) {
        urlInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                loadFromUrl();
            }
        });
    }

    if (preview) {
        preview.addEventListener("click", () => {
            if (preview.src) openLightbox(preview.src);
        });
    }
    if (lightboxClose) lightboxClose.addEventListener("click", closeLightbox);
    if (lightboxBackdrop) lightboxBackdrop.addEventListener("click", closeLightbox);
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && lightbox && !lightbox.hidden) {
            closeLightbox();
        }
    });

    if (form && submitBtn) {
        form.addEventListener("submit", () => {
            const label = submitBtn.querySelector(".submit-btn__label");
            if (label) label.textContent = "Analizando...";
            submitBtn.disabled = true;
            if (analyzeOverlay) {
                analyzeOverlay.hidden = false;
                analyzeOverlay.setAttribute("aria-hidden", "false");
                document.body.style.overflow = "hidden";
            }
        });
    }
});
