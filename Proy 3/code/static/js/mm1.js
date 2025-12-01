// -------- API server/cliente --------
async function callApi(endpoint, label) {
    const statusEl = document.getElementById("status");
    if (!statusEl) return;
    statusEl.textContent = `Ejecutando ${label}...`;
    statusEl.className = "status";

    try {
        const resp = await fetch(endpoint, { method: "POST" });
        const data = await resp.json();
        statusEl.textContent = data.message || `Respuesta de ${label}`;
        statusEl.className = "status " + (data.success ? "ok" : "error");
    } catch (err) {
        statusEl.textContent = `Error al llamar ${label}: ` + err;
        statusEl.className = "status error";
    }
}

// -------- Guardar experimento --------
async function guardarExperimento() {
    const el = document.getElementById("current-metrics");
    if (!el) return;

    const d = el.dataset;

    const payload = {
        lambda_server: parseFloat(d.lambdaServer),
        lambda_client: parseFloat(d.lambdaClient),
        mu_hat: parseFloat(d.muHat),
        rho_hat: parseFloat(d.rhoHat),
        Wq_emp: parseFloat(d.wqEmp),
        W_emp: parseFloat(d.wEmp),
        Lq_emp: parseFloat(d.lqEmp),
        L_emp: parseFloat(d.lEmp),
        Wq_teo: parseFloat(d.wqTeo),
        W_teo: parseFloat(d.wTeo),
        Lq_teo: parseFloat(d.lqTeo),
        L_teo: parseFloat(d.lTeo),
        Wq_sim: parseFloat(d.wqSim),
        W_sim: parseFloat(d.wSim),
        Lq_sim: parseFloat(d.lqSim),
        L_sim: parseFloat(d.lSim),
        lambda_teo: parseFloat(d.lambdaTeo),
        mu_teo: parseFloat(d.muTeo),
        n_sim: parseInt(d.nSim, 10),
    };

    if (d.lambdaW) {
        payload.lambda_wireshark = parseFloat(d.lambdaW);
    }

    const statusEl = document.getElementById("status");
    if (statusEl) {
        statusEl.textContent = "Guardando experimento...";
        statusEl.className = "status";
    }

    try {
        const resp = await fetch("/api/save_experiment", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await resp.json();
        if (statusEl) {
            statusEl.textContent = data.message || "Experimento guardado.";
            statusEl.className = "status " + (data.success ? "ok" : "error");
        }
    } catch (err) {
        if (statusEl) {
            statusEl.textContent = "Error al guardar experimento: " + err;
            statusEl.className = "status error";
        }
    }
}

// -------- Carrusel de imágenes --------
function initCarousel() {
    if (!window.MM1_HAS_PLOTS || !window.MM1_PLOTS || !window.MM1_PLOTS.length) {
        return;
    }

    const base = window.MM1_PLOTS_BASE || "/static/plots/";
    const items = window.MM1_PLOTS.map(([fname, title]) => ({
        src: base + fname,
        title: title
    }));

    let currentIndex = 0;

    const imgEl = document.getElementById("carousel-image");
    const captionEl = document.getElementById("carousel-caption");
    const dotsContainer = document.getElementById("carousel-dots");
    const prevBtn = document.getElementById("carousel-prev");
    const nextBtn = document.getElementById("carousel-next");

    if (!imgEl || !captionEl || !dotsContainer || !prevBtn || !nextBtn) {
        return;
    }

    function renderDots() {
        dotsContainer.innerHTML = "";
        items.forEach((_, idx) => {
            const dot = document.createElement("button");
            dot.type = "button";
            dot.className = "carousel-dot" + (idx === currentIndex ? " active" : "");
            dot.addEventListener("click", () => {
                currentIndex = idx;
                updateCarousel();
            });
            dotsContainer.appendChild(dot);
        });
    }

    function updateCarousel() {
        const item = items[currentIndex];
        imgEl.src = item.src;
        captionEl.textContent = item.title;
        renderDots();
    }

    prevBtn.addEventListener("click", () => {
        currentIndex = (currentIndex - 1 + items.length) % items.length;
        updateCarousel();
    });

    nextBtn.addEventListener("click", () => {
        currentIndex = (currentIndex + 1) % items.length;
        updateCarousel();
    });

    // inicial
    updateCarousel();
}

// -------- Arranque general --------
document.addEventListener("DOMContentLoaded", () => {
    const btnServer = document.getElementById("btn-server");
    const btnClient = document.getElementById("btn-client");
    const btnSave = document.getElementById("btn-save-exp");

    if (btnServer) {
        btnServer.addEventListener("click", () => callApi("/api/start_server", "servidor"));
    }
    if (btnClient) {
        btnClient.addEventListener("click", () => callApi("/api/start_client", "cliente"));
    }
    if (btnSave) {
        btnSave.addEventListener("click", (e) => {
            e.preventDefault();
            guardarExperimento();
        });
    }

    initCarousel();
});
