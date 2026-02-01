const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const imageUpload = document.getElementById('image-upload');

let context = ""; // Store last diagnosis context

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim() === "";
});

// Handle Send
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Handle Image Upload
imageUpload.addEventListener('change', function () {
    if (this.files.length > 0) {
        uploadImage(this.files[0]);
    }
});

function setInput(text) {
    userInput.value = text;
    userInput.style.height = 'auto'; // Reset
    sendBtn.disabled = false;
    userInput.focus();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // 1. Add User Message
    addMessage(text, 'user');
    userInput.value = "";
    userInput.style.height = 'auto';
    sendBtn.disabled = true;

    // 2. Show Typing Indicator
    const typingId = showTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('message', text);
        formData.append('context', context);

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // 3. Remove Indicator & Add AI Response
        removeTypingIndicator(typingId);
        addMessage(data.response, 'ai');

    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage("Sorry, something went wrong. Please try again.", 'ai');
        console.error(error);
    }
}

async function uploadImage(file) {
    // 1. Add User Message (Image Preview)
    const reader = new FileReader();
    reader.onload = function (e) {
        const imgHtml = `<img src="${e.target.result}" style="max-width: 200px; border-radius: 8px;">`;
        addMessage(imgHtml, 'user');
    };
    reader.readAsDataURL(file);

    // 2. Show Processing Indicator
    const typingId = showTypingIndicator("Analyzing medical image...");

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        removeTypingIndicator(typingId);

        // Update Context
        context = `Previous diagnosis: ${data.classification.class} (${(data.classification.confidence * 100).toFixed(1)}%). Advice given: ${data.advice}`;

        // 3. Display Result Card
        const resultHtml = `
            <div class="result-card">
                <h5>Analysis Result</h5>
                <p><strong>Detected Class:</strong> ${data.classification.class}</p>
                <p><strong>Confidence:</strong> ${(data.classification.confidence * 100).toFixed(2)}%</p>
                
                <div style="position: relative; max-width: 300px; margin-top: 10px;">
                    <img src="${data.original_url}" id="img-${data.heatmap_url}" style="width: 100%; border-radius: 8px;">
                    <img src="${data.heatmap_url}" id="heat-${data.heatmap_url}" style="width: 100%; border-radius: 8px; position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 0.5s;">
                </div>
                
                <div class="heatmap-toggle-btn" onclick="toggleHeatmap('${data.heatmap_url}')">
                    <i class="fa-solid fa-layer-group"></i> Show Gradient Heatmap (Segmentation)
                </div>

                <hr style="opacity: 0.1">
                <div class="markdown-body">${marked.parse(data.advice)}</div>
            </div>
        `;

        addMessage(resultHtml, 'ai', true);

    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage("Image analysis failed. Please ensure it is a valid medical image.", 'ai');
        console.error(error);
    }

    // Reset file input
    imageUpload.value = "";
}

function addMessage(content, sender, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${sender}-message`;

    const icon = sender === 'ai' ? '<i class="fa-solid fa-robot"></i>' : '<i class="fa-regular fa-user"></i>';

    let innerContent = isHtml ? content : `<p>${content}</p>`;
    // If text messages from AI, parse markdown
    if (sender === 'ai' && !isHtml) {
        innerContent = marked.parse(content);
    }

    div.innerHTML = `
        <div class="avatar-icon">${icon}</div>
        <div class="message-content">${innerContent}</div>
    `;

    messagesContainer.appendChild(div);
    scrollToBottom();
}

function showTypingIndicator(text = "Thinking...") {
    const id = "typing-" + Date.now();
    const div = document.createElement('div');
    div.className = `message ai-message`;
    div.id = id;
    div.innerHTML = `
        <div class="avatar-icon"><i class="fa-solid fa-robot"></i></div>
        <div class="message-content">
            <div class="typing-indicator"><i class="fa-solid fa-circle-notch fa-spin"></i> ${text}</div>
        </div>
    `;
    messagesContainer.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

function scrollToBottom() {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function toggleHeatmap(id) {
    const heatmap = document.getElementById(`heat-${id}`);
    const btn = event.currentTarget; // The clicked button

    if (heatmap.style.opacity === "0") {
        heatmap.style.opacity = "1";
        btn.innerHTML = '<i class="fa-solid fa-eye-slash"></i> Hide Heatmap';
    } else {
        heatmap.style.opacity = "0";
        btn.innerHTML = '<i class="fa-solid fa-layer-group"></i> Show Gradient Heatmap (Segmentation)';
    }
}
