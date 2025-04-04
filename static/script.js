document.getElementById("theme-toggle").addEventListener("click", function () {
    document.body.classList.toggle("dark-mode");
});

async function predictFakeNews() {
    let inputText = document.getElementById("newsText").value;
    let resultDiv = document.getElementById("result");
    let loadingDiv = document.getElementById("loading");

    if (!inputText) {
        alert("Please enter news text!");
        return;
    }

    resultDiv.classList.remove("show-result");
    resultDiv.innerText = "";
    loadingDiv.style.display = "block";

    let response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText })
    });

    let result = await response.json();

    loadingDiv.style.display = "none";
    resultDiv.innerText = "Prediction: " + result.prediction;
    resultDiv.classList.add("show-result");
}

function chatWithAI() {
    let userMessage = document.getElementById("chatInput").value;
    let responseDiv = document.getElementById("chatResponse");

    if (!userMessage) {
        responseDiv.innerText = "Ask me something!";
        return;
    }

    let responses = {
        "what is fake news": "Fake news is false or misleading information presented as news.",
        "how to detect fake news": "Check the source, verify facts, and look for bias.",
        "is social media reliable": "Social media is not always reliable. Always verify from trusted sources."
    };

    responseDiv.innerText = responses[userMessage.toLowerCase()] || "I'm not sure, but always fact-check!";
}

let quotes = [
    "Think before you share – Fake news spreads like wildfire! 🔥",
    "A lie can travel halfway around the world before the truth puts on its shoes. 🌍",
    "Verify before you amplify! 📢",
    "Stop. Read. Verify. 🔍"
];

setInterval(() => {
    document.getElementById("awareness-quote").innerText = quotes[Math.floor(Math.random() * quotes.length)];
}, 5000);
