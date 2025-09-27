const chatbox = document.getElementById("chatbox");
const userInput = document.getElementById("userInput");

function appendMessage(sender, message){
    const div = document.createElement("div");
    div.className = sender;
    div.innerHTML = `<b>${sender}:</b> ${message}`;
    chatbox.appendChild(div);
    chatbox.scrollTop = chatbox.scrollHeight;
}

function sendMessage(){
    const message = userInput.value;
    if(message.trim() === "") return;
    appendMessage("User", message);
    userInput.value = "";

    fetch("/get_response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(res => res.json())
    .then(data => {
        appendMessage("Bot", data.response);
        speak(data.response);
    });
}

// Voice recognition
function startVoice(){
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = function(event){
        const voiceMessage = event.results[0][0].transcript;
        userInput.value = voiceMessage;
        sendMessage();
    }
}

// Text-to-Speech
function speak(text){
    const utterance = new SpeechSynthesisUtterance(text);
    speechSynthesis.speak(utterance);
}
