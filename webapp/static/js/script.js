document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const preview = document.getElementById("preview");
    const form = document.querySelector("form");
    const button = document.getElementById("submitBtn");

    fileInput.addEventListener("change", function () {
        const file = fileInput.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = function (e) {
                preview.src = e.target.result;
                preview.style.display = "block";
            };

            reader.readAsDataURL(file);
        }
    });

    form.addEventListener("submit", function () {
        button.textContent = "Predicting...";
        button.disabled = true;
    });
});
