import "./style.css";

document.querySelector("#app").innerHTML = `
  <main class="container">
    <h1>HM Fit - Prototipo Tauri</h1>
    <p class="subtitle">Fase 1: conectar con FastAPI manualmente</p>
    <button id="health-check">Probar conexión</button>
    <pre id="health-result" aria-live="polite">Esperando...</pre>
  </main>
`;

const button = document.getElementById("health-check");
const result = document.getElementById("health-result");

button.addEventListener("click", async () => {
  result.textContent = "Consultando http://127.0.0.1:8000/health ...";
  try {
    const response = await fetch("http://127.0.0.1:8000/health");
    const data = await response.json();
    result.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    result.textContent = `Error: ${error}`;
  }
});
