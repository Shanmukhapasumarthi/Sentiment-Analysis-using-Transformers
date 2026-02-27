* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Segoe UI', sans-serif;
}

body {
    height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}

.background {
    position: absolute;
    width: 100%;
    height: 100%;
    backdrop-filter: blur(6px);
}

.container {
    position: relative;
    width: 500px;
    padding: 40px;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(15px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    text-align: center;
    color: white;
}

h1 {
    margin-bottom: 10px;
    font-size: 28px;
}

.subtitle {
    margin-bottom: 25px;
    font-size: 14px;
    opacity: 0.8;
}

textarea {
    width: 100%;
    height: 120px;
    padding: 12px;
    border-radius: 10px;
    border: none;
    outline: none;
    resize: none;
    font-size: 14px;
    margin-bottom: 20px;
}

button {
    width: 100%;
    padding: 12px;
    border-radius: 10px;
    border: none;
    background: #ffffff;
    color: #1e3c72;
    font-weight: bold;
    cursor: pointer;
    transition: 0.3s ease;
}

button:hover {
    background: #f0f0f0;
    transform: scale(1.03);
}

.result {
    margin-top: 25px;
    padding: 15px;
    border-radius: 12px;
    animation: fadeIn 0.5s ease-in-out;
}

.result h2 {
    margin-bottom: 5px;
}

.positive {
    background: rgba(46, 204, 113, 0.8);
}

.negative {
    background: rgba(231, 76, 60, 0.8);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Responsive */
@media (max-width: 600px) {
    .container {
        width: 90%;
        padding: 25px;
    }
}