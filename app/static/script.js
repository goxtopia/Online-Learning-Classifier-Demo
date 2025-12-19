let currentDetections = [];
let currentFilename = "";
let selectedDetection = null;
let currentMode = "mlp";
let rtspInterval = null;

const imgElement = document.getElementById("uploadedImage");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const feedbackPanel = document.getElementById("feedback-panel");
const backdrop = document.getElementById("backdrop");
const selScoreSpan = document.getElementById("sel-score");
const loadingIndicator = document.getElementById("loading");
const alertArea = document.getElementById("alert-area");

function setMode(mode) {
    currentMode = mode;
    console.log("Mode switched to:", currentMode);
}

function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    if (fileInput.files.length === 0) {
        showAlert("Please select a file.");
        return;
    }

    loadingIndicator.style.display = "inline";
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("mode", currentMode);

    // Clear previous state
    currentDetections = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Hide image until loaded
    imgElement.style.opacity = 0.5;

    // Clear raw detections
    document.getElementById("raw-yolo-section").style.display = "none";
    document.getElementById("raw-detections-list").innerHTML = "";

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(response => {
        if (!response.ok) throw new Error("Upload failed");
        return response.json();
    })
    .then(data => {
        currentFilename = data.filename;
        currentDetections = data.detections;
        imgElement.src = data.url;
        imgElement.style.opacity = 1;
        
        imgElement.onload = () => {
            resizeCanvas();
            drawBoxes();
            renderRawDetections(data.detections);
            loadingIndicator.style.display = "none";
            
            if (currentDetections.length === 0) {
                showAlert("No persons detected in this image.");
            }
        };
    })
    .catch(error => {
        console.error("Error:", error);
        showAlert("Error uploading image.");
        loadingIndicator.style.display = "none";
    });
}

function resizeCanvas() {
    canvas.width = imgElement.width;
    canvas.height = imgElement.height;
}

function getScaleFactors() {
    const naturalW = imgElement.naturalWidth;
    const naturalH = imgElement.naturalHeight;
    const displayW = imgElement.width;
    const displayH = imgElement.height;
    return { x: displayW / naturalW, y: displayH / naturalH };
}

function drawBoxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = getScaleFactors();

    currentDetections.forEach(det => {
        // Filter out items that are strictly "filtered" by backend logic for the main view
        if (det.status === "filtered_clip") return;

        const [x1, y1, x2, y2] = det.bbox;
        const sx1 = x1 * scale.x;
        const sy1 = y1 * scale.y;
        const sx2 = x2 * scale.x;
        const sy2 = y2 * scale.y;
        const w = sx2 - sx1;
        const h = sy2 - sy1;

        ctx.lineWidth = 3;
        
        let color = "#28a745"; // Default green
        let text = "";

        if (currentMode === "mlp") {
             const isHuman = det.human_prob > 0.5;
             color = isHuman ? "#28a745" : "#dc3545";
             text = `P: ${(det.human_prob * 100).toFixed(1)}%`;
        } else {
            // CLIP Mode (Accepted items)
            color = "#28a745";
            text = "Human (CLIP)";
        }

        ctx.strokeStyle = color;
        ctx.strokeRect(sx1, sy1, w, h);

        // Draw label background
        ctx.fillStyle = color;
        const textWidth = ctx.measureText(text).width;
        ctx.fillRect(sx1, sy1 - 20, textWidth + 10, 20);

        // Draw label text
        ctx.fillStyle = "white";
        ctx.font = "bold 14px Arial";
        ctx.fillText(text, sx1 + 5, sy1 - 5);
    });
}

function renderRawDetections(detections) {
    const section = document.getElementById("raw-yolo-section");
    const container = document.getElementById("raw-detections-list");
    section.style.display = "block";
    container.innerHTML = "";

    if (detections.length === 0) {
        container.innerHTML = "<p>No detections found.</p>";
        return;
    }

    // We need to crop from the original image to show them here.
    // Since we don't have separate crop URLs for every detection from backend (only for RTSP/History),
    // we can draw them onto small canvases.
    const scale = getScaleFactors();

    detections.forEach((det, index) => {
        const div = document.createElement("div");
        div.className = "gallery-item";

        // Canvas for crop
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = 100;
        cropCanvas.height = 100;
        const ctxCrop = cropCanvas.getContext("2d");

        // Draw crop
        // det.bbox is in original coordinates
        const [x1, y1, x2, y2] = det.bbox;
        const w = x2 - x1;
        const h = y2 - y1;

        // Draw the specific region from the source image to the small canvas
        // imgElement might be scaled in DOM, but we use natural dimensions for source
        ctxCrop.drawImage(imgElement, x1, y1, w, h, 0, 0, 100, 100);

        div.appendChild(cropCanvas);

        // Info
        const info = document.createElement("div");
        info.style.fontSize = "12px";
        info.innerHTML = `
            <b>Status:</b> ${det.status}<br>
            <b>YOLO Conf:</b> ${(det.yolo_conf * 100).toFixed(1)}%<br>
            <b>Human Prob:</b> ${(det.human_prob * 100).toFixed(1)}%
        `;
        div.appendChild(info);

        // Highlight logic
        let isRejected = false;
        let badgeText = "Accepted";

        if (det.status === "filtered_clip") {
            isRejected = true;
            badgeText = "CLIP Filtered";
        } else if (det.status === "processed_mlp") {
             if (det.human_prob < 0.5) {
                 isRejected = true;
                 badgeText = "MLP Rejected";
             } else {
                 badgeText = "MLP Accepted";
             }
        } else {
             // accepted or other
             badgeText = "Accepted";
        }

        if (isRejected) {
            div.style.border = "3px solid #dc3545"; // Red for filtered
            div.style.backgroundColor = "#fff5f5";
            div.style.opacity = "0.7";
        } else {
             div.style.border = "3px solid #28a745"; // Green for accepted
             div.style.backgroundColor = "#f0fff4";
        }

        // Add badge
        const badge = document.createElement("span");
        badge.innerText = badgeText;
        badge.style.position = "absolute";
        badge.style.top = "5px";
        badge.style.left = "5px";
        badge.style.padding = "2px 5px";
        badge.style.borderRadius = "4px";
        badge.style.fontSize = "10px";
        badge.style.color = "white";
        badge.style.fontWeight = "bold";
        badge.style.backgroundColor = isRejected ? "#dc3545" : "#28a745";

        div.style.position = "relative";
        div.appendChild(badge);

        container.appendChild(div);
    });
}

canvas.addEventListener("click", (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const scale = getScaleFactors();
    
    selectedDetection = null;
    for (let i = currentDetections.length - 1; i >= 0; i--) {
        const det = currentDetections[i];
        if (det.status === "filtered_clip") continue; // Can't select filtered ones on main view

        const [bx1, by1, bx2, by2] = det.bbox;
        const sx1 = bx1 * scale.x;
        const sy1 = by1 * scale.y;
        const sx2 = bx2 * scale.x;
        const sy2 = by2 * scale.y;

        if (x >= sx1 && x <= sx2 && y >= sy1 && y <= sy2) {
            selectedDetection = det;
            break;
        }
    }

    if (selectedDetection) {
        openFeedback(selectedDetection);
    } else {
        closeFeedback();
    }
});

function openFeedback(det) {
    feedbackPanel.style.display = "block";
    backdrop.style.display = "block";
    if (currentMode === "mlp") {
        selScoreSpan.innerText = `${(det.human_prob * 100).toFixed(1)}% Human`;
    } else {
        selScoreSpan.innerText = "Presumed Human (CLIP)";
    }
    
    // Highlight selected
    drawBoxes();
    const scale = getScaleFactors();
    const [x1, y1, x2, y2] = det.bbox;
    ctx.lineWidth = 5;
    ctx.strokeStyle = "#ffc107"; // Yellow highlight
    ctx.strokeRect(x1 * scale.x, y1 * scale.y, (x2-x1)*scale.x, (y2-y1)*scale.y);
}

function closeFeedback() {
    feedbackPanel.style.display = "none";
    backdrop.style.display = "none";
    selectedDetection = null;
    drawBoxes();
}

function sendFeedback(isHuman) {
    if (!selectedDetection || !currentFilename) return;

    fetch("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            filename: currentFilename,
            bbox: selectedDetection.bbox,
            is_human: isHuman,
            mode: currentMode
        })
    })
    .then(res => res.json())
    .then(data => {
        showAlert("Feedback recorded!");
        closeFeedback();
        if (currentMode === "clip") {
            loadHistory(); // Refresh history if we added a negative
        }
    })
    .catch(err => {
        console.error(err);
        showAlert("Error sending feedback");
    });
}

// --- RTSP Functions ---

function startRtsp() {
    const url = document.getElementById("rtspUrl").value;
    if (!url) {
        showAlert("Please enter an RTSP URL");
        return;
    }

    fetch("/rtsp/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url })
    })
    .then(res => res.json())
    .then(data => {
        showAlert("RTSP Monitor Started");
        document.getElementById("rtsp-status").innerText = "Status: Running";
        document.getElementById("rtsp-events-container").style.display = "block";
        if (rtspInterval) clearInterval(rtspInterval);
        rtspInterval = setInterval(pollRtspEvents, 2000); // Poll every 2s
    })
    .catch(err => showAlert("Error starting RTSP"));
}

function stopRtsp() {
    fetch("/rtsp/stop", { method: "POST" })
    .then(res => res.json())
    .then(data => {
        showAlert("RTSP Monitor Stopped");
        document.getElementById("rtsp-status").innerText = "Status: Stopped";
        if (rtspInterval) clearInterval(rtspInterval);
    })
    .catch(err => showAlert("Error stopping RTSP"));
}

function pollRtspEvents() {
    fetch("/rtsp/events")
    .then(res => res.json())
    .then(events => {
        const container = document.getElementById("event-list");
        container.innerHTML = "";
        events.forEach(evt => {
            const div = document.createElement("div");
            div.className = "gallery-item";

            const img = document.createElement("img");
            img.src = evt.image_url;
            div.appendChild(img);

            const btnGroup = document.createElement("div");

            const btnYes = document.createElement("button");
            btnYes.innerText = "Human";
            btnYes.className = "success small";
            btnYes.onclick = () => labelRtspEvent(evt.id, true);

            const btnNo = document.createElement("button");
            btnNo.innerText = "Not Human";
            btnNo.className = "danger small";
            btnNo.onclick = () => labelRtspEvent(evt.id, false);

            btnGroup.appendChild(btnYes);
            btnGroup.appendChild(btnNo);
            div.appendChild(btnGroup);

            container.appendChild(div);
        });
    });
}

function labelRtspEvent(eventId, isHuman) {
    fetch("/rtsp/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ event_id: eventId, is_human: isHuman })
    })
    .then(res => res.json())
    .then(data => {
        pollRtspEvents(); // Refresh list
    });
}

// --- History Functions ---

function loadHistory() {
    fetch("/history")
    .then(res => res.json())
    .then(items => {
        const container = document.getElementById("history-list");
        container.innerHTML = "";
        if (items.length === 0) {
            container.innerHTML = "<p>No negative samples in CLIP history.</p>";
            return;
        }
        items.forEach(item => {
            const div = document.createElement("div");
            div.className = "gallery-item";

            // Assuming item has image_url or we reconstruct it.
            // Ideally backend returns a crop URL or we use the original with css crop.
            // Simpler: Backend saves crop for history display.

            const img = document.createElement("img");
            img.src = item.image_url || "/static/blank.jpg"; // Fallback
            div.appendChild(img);

            const btnDelete = document.createElement("button");
            btnDelete.innerText = "Delete";
            btnDelete.className = "danger small";
            btnDelete.onclick = () => deleteHistoryItem(item.id);

            div.appendChild(btnDelete);
            container.appendChild(div);
        });
    });
}

function deleteHistoryItem(itemId) {
    fetch(`/history/${itemId}`, { method: "DELETE" })
    .then(res => res.json())
    .then(data => {
        loadHistory();
    });
}

function showAlert(msg) {
    const div = document.createElement("div");
    div.className = "alert";
    div.innerText = msg;
    alertArea.appendChild(div);
    setTimeout(() => {
        div.style.opacity = "0";
        setTimeout(() => div.remove(), 500);
    }, 3000);
}
