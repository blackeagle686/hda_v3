const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const imageUpload = document.getElementById('image-upload');

// Generate Session ID (simple random string)
const sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
console.log("Session ID:", sessionId);

// Context Memory (List of short summaries)
let contextSummaries = [];
let stagedImage = null;

// Theme Toggle Logic
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';

    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    const icon = document.getElementById('theme-icon');
    if (theme === 'light') {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    } else {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    }
}

// Initialize Theme
(function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    document.addEventListener('DOMContentLoaded', () => {
        updateThemeIcon(savedTheme);
    });
})();

// Auto-resize textarea
userInput.addEventListener('input', function () {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
    sendBtn.disabled = this.value.trim() === "" && !stagedImage;
});

// Handle Send
sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Handle Image Selection (Staging)
imageUpload.addEventListener('change', function () {
    if (this.files.length > 0) {
        stagedImage = this.files[0];
        sendBtn.disabled = false;

        // Show preview in UI
        const reader = new FileReader();
        reader.onload = e => {
            const previewId = "preview-" + Date.now();
            const div = document.createElement('div');
            div.id = previewId;
            div.className = "message user-message staged-message";
            div.innerHTML = `
                <div class="avatar-icon"><i class="fa-regular fa-user"></i></div>
                <div class="message-content">
                    <div style="position:relative; display:inline-block">
                        <img src="${e.target.result}" style="max-width:150px;border-radius:12px; border: 2px solid var(--accent-color); box-shadow: 0 4px 12px rgba(0,0,0,0.3)">
                        <button onclick="clearStagedImage('${previewId}')" style="position:absolute; top:-10px; right:-10px; border-radius:50%; width:28px; height:28px; background:#ff4444; color:white; border:none; cursor:pointer; display:flex; align-items:center; justify-content:center; font-weight:bold; box-shadow: 0 2px 4px rgba(0,0,0,0.3)">&times;</button>
                    </div>
                    <p style="font-size:0.85rem; color:var(--text-muted); margin-top:8px; font-style: italic">Image attached. Type your specific question above and click send.</p>
                </div>
            `;
            messagesContainer.appendChild(div);
            scrollToBottom();
        };
        reader.readAsDataURL(stagedImage);
    }
});

function clearStagedImage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
    stagedImage = null;
    imageUpload.value = "";
    sendBtn.disabled = userInput.value.trim() === "";
}

function setInput(text) {
    userInput.value = text;
    userInput.style.height = 'auto'; // Reset
    sendBtn.disabled = false;
    userInput.focus();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text && !stagedImage) return;

    if (stagedImage) {
        uploadImageAndMessage(stagedImage, text);
        stagedImage = null;
        imageUpload.value = "";
        // Remove existing staged previews
        document.querySelectorAll('.staged-message').forEach(el => el.remove());
    } else {
        sendTextMessage(text);
    }

    userInput.value = "";
    userInput.style.height = 'auto';
    sendBtn.disabled = true;
}

async function sendTextMessage(text) {
    addMessage(text, 'user');
    const typingId = showTypingIndicator();

    try {
        const formData = new FormData();
        formData.append('message', text);
        formData.append('context_summaries', JSON.stringify(contextSummaries));
        formData.append('session_id', sessionId);

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        removeTypingIndicator(typingId);
        addMessage(data.response, 'ai');

        if (data.summary) {
            contextSummaries.push(data.summary);
        }

    } catch (error) {
        removeTypingIndicator(typingId);
        addMessage("Sorry, something went wrong. Please try again.", 'ai');
        console.error(error);
    }
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('collapsed');
}

async function uploadImageAndMessage(file, text) {
    // Show image in chat as part of user message
    const reader = new FileReader();
    reader.onload = e => {
        const content = `
            <div style="display:flex; flex-direction:column; gap:10px;">
                <img src="${e.target.result}" style="max-width:250px; border-radius:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.2)">
                ${text ? `<p style="margin:0; line-height:1.5;">${text}</p>` : ''}
            </div>
        `;
        addMessage(content, 'user', true);
    };
    reader.readAsDataURL(file);

    const typingId = showTypingIndicator("Analyzing image and clinical data...");

    try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("message", text || "");

        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        removeTypingIndicator(typingId);

        if (!response.ok) throw new Error(data.error || "Analysis failed");

        const safeId = data.original_url.replaceAll("/", "_");
        const reportContent = data.response || data.advice;

        const resultHtml = `
        <div class="result-card" style="border-left: 4px solid var(--accent-color); padding-left: 15px;">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom: 12px;">
                <h5 style="color: var(--accent-color); margin: 0;"><i class="fa-solid fa-file-medical"></i> Detailed Medical Report</h5>
                <div style="display:flex; gap:8px;">
                    <button onclick='downloadReport("pdf", ${JSON.stringify(reportContent).replace(/'/g, "&apos;")})' class="btn" style="padding: 4px 8px; font-size: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-main); cursor: pointer; display: flex; align-items: center; gap: 4px;">
                        <i class="fa-solid fa-file-pdf" style="color: #ff4444;"></i> PDF
                    </button>
                    <button onclick='downloadReport("docx", ${JSON.stringify(reportContent).replace(/'/g, "&apos;")})' class="btn" style="padding: 4px 8px; font-size: 0.75rem; background: var(--bg-secondary); border: 1px solid var(--border-color); color: var(--text-main); cursor: pointer; display: flex; align-items: center; gap: 4px;">
                        <i class="fa-solid fa-file-word" style="color: #2b579a;"></i> Word
                    </button>
                </div>
            </div>
            
            <p style="margin-bottom: 5px;"><b>Detected Class:</b> <span class="badge" style="background: var(--bg-secondary); color: var(--accent-color); padding: 2px 8px; border-radius: 4px;">${data.classification.class}</span></p>
            <p><b>Confidence:</b> ${(data.classification.confidence * 100).toFixed(2)}%</p>

            <div style="position:relative; max-width:100%; margin:15px 0;">
                <img src="${data.original_url}" id="img-${safeId}" style="width:100%; max-width:400px; border-radius:12px; border: 1px solid var(--border-color);">
            </div>

            <hr style="border: 0; border-top: 1px solid var(--border-color); margin: 15px 0;">
            <div class="markdown-body" style="font-size: 0.95rem; line-height: 1.6;">
                ${marked.parse(reportContent)}
            </div>
        </div>
        `;

        addMessage(resultHtml, "ai", true);

        if (data.summary) {
            contextSummaries.push(data.summary);
        }

    } catch (err) {
        removeTypingIndicator(typingId);
        console.error(err);
        addMessage("❌ Analysis failed: " + err.message, "ai");
    }
}

async function downloadReport(format, content) {
    try {
        const formData = new FormData();
        formData.append("content", content);
        formData.append("format", format);

        const response = await fetch("/api/download-report", {
            method: "POST",
            body: formData
        });

        if (!response.ok) throw new Error("Download server error");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `HDA_Medical_Report.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (err) {
        console.error(err);
        alert("Download failed: " + err.message);
    }
}

// toggleHeatmap functions removed


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


