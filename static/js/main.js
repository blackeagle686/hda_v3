const messagesContainer = document.getElementById('messages-container');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const imageUpload = document.getElementById('image-upload');

// Generate Session ID (simple random string)
const sessionId = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
console.log("Session ID:", sessionId);

// Context Memory (List of short summaries)
let contextSummaries = [];

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
        // Send JSON string of summaries
        formData.append('context_summaries', JSON.stringify(contextSummaries));
        formData.append('session_id', sessionId);

        const response = await fetch('/api/chat', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        // 3. Remove Indicator & Add AI Response
        removeTypingIndicator(typingId);
        addMessage(data.response, 'ai');

        // 4. Update Context Memory
        if (data.summary) {
            contextSummaries.push(data.summary);
            console.log("Updated Context Summaries:", contextSummaries);
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

async function uploadImage(file) {
    const reader = new FileReader();
    reader.onload = e => {
        addMessage(
            `<img src="${e.target.result}" style="max-width:200px;border-radius:8px;">`,
            'user'
        );
    };
    reader.readAsDataURL(file);

    const typingId = showTypingIndicator("Analyzing medical image...");

    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch("/api/analyze", {
            method: "POST",
            body: formData
        });

        const text = await response.text();

        if (!response.ok) {
            console.error("Backend error:", text);
            throw new Error("API error");
        }

        const data = JSON.parse(text);

        removeTypingIndicator(typingId);

        const safeId = data.original_url.replaceAll("/", "_");

        const resultHtml = `
        <div class="result-card">
            <h5>Analysis Result</h5>
            <p><b>Detected Class:</b> ${data.classification.class}</p>
            <p><b>Confidence:</b> ${(data.classification.confidence * 100).toFixed(2)}%</p>

            <div style="position:relative;max-width:300px">
                <img src="${data.original_url}" id="img-${safeId}" style="width:100%">
            </div>

            <hr>
            <div class="markdown-body">${marked.parse(data.response || data.advice)}</div>
        </div>
        `;

        addMessage(resultHtml, "ai", true);

        // Update Context Memory from Image Analysis
        if (data.summary) {
            contextSummaries.push(data.summary);
            console.log("Updated Context Summaries (Image):", contextSummaries);
        }

    } catch (err) {
        removeTypingIndicator(typingId);
        console.error(err);
        addMessage("❌ Image analysis failed.", "ai");
    }

    imageUpload.value = "";
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


