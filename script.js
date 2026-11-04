/* ==========================================================
   CONFIGURATION : À MODIFIER PAR L'ÉQUIPE
   ========================================================== */
// >>> Remplacez cette URL par celle de votre API FastAPI (route de prédiction) <<<
const API_URL = "http://127.0.0.1:8000/predict";

// Nom de la clé contenant le prix dans la réponse JSON de l'API (ex : {"prediction": 24.5})
const RESPONSE_KEY = "prediction";

// Dans Boston Housing, le prix (MEDV) est exprimé en milliers de dollars.
// Mettez 1 si votre API renvoie déjà un prix en dollars.
const PRICE_MULTIPLIER = 1000;

// Les 13 variables, dans l'ordre du dataset
const FEATURES = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT"];

/* ===== Références aux éléments HTML ===== */
const form = document.getElementById("prediction-form");
const predictBtn = document.getElementById("predict-btn");
const resetBtn = document.getElementById("reset-btn");
const resultBox = document.getElementById("result");
const resultTitle = document.getElementById("result-title");
const resultValue = document.getElementById("result-value");

/* ===== Affichage d'erreur / d'état ===== */
function setFieldError(input, message) {
  const field = input.closest(".field");
  field.querySelector(".error").textContent = message;
  field.classList.toggle("invalid", message !== "");
}

function showResult(type, title, text) {
  resultBox.hidden = false;
  resultBox.className = "result " + type;   // "success", "error-box" ou ""
  resultTitle.textContent = title;
  resultValue.textContent = text;
}

/* ===== Validation : retourne true si tout est correct ===== */
function validateForm() {
  let valid = true;
  FEATURES.forEach((name) => {
    const input = form.elements[name];
    const raw = input.value.trim();
    let message = "";

    if (raw === "") {
      message = "Ce champ est obligatoire.";
    } else if (name === "CHAS") {
      if (raw !== "0" && raw !== "1") message = "Choisissez 0 ou 1.";
    } else {
      const value = Number(raw);
      const min = Number(input.min), max = Number(input.max);
      if (!Number.isFinite(value)) message = "Saisissez un nombre valide.";
      else if (value < min || value > max) message = `La valeur doit être comprise entre ${min} et ${max}.`;
      else if (name === "RAD" && !Number.isInteger(value)) message = "RAD doit être un entier.";
    }
    setFieldError(input, message);
    if (message) valid = false;
  });
  return valid;
}

/* ===== Construction de l'objet JSON (valeurs converties en nombres) ===== */
function collectData() {
  const data = {};
  FEATURES.forEach((name) => { data[name] = Number(form.elements[name].value); });
  return data;
}

/* ===== Envoi à l'API FastAPI avec fetch() ===== */
async function sendToAPI(data) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  if (!response.ok) {
    throw new Error(`L'API a répondu avec une erreur (code ${response.status}).`);
  }
  return response.json();
}

/* ===== Clic sur « Prédire le prix » ===== */
form.addEventListener("submit", async (event) => {
  event.preventDefault();                       // empêche le rechargement de la page
  resultBox.hidden = true;

  if (!validateForm()) {
    showResult("error-box", "Formulaire incomplet", "Corrigez les champs signalés en rouge puis réessayez.");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Calcul en cours…";
  try {
    const json = await sendToAPI(collectData());
    const price = Number(json[RESPONSE_KEY]);
    if (!Number.isFinite(price)) {
      throw new Error(`Réponse inattendue : la clé « ${RESPONSE_KEY} » est absente ou n'est pas un nombre.`);
    }
    const formatted = (price * PRICE_MULTIPLIER).toLocaleString("fr-FR", { maximumFractionDigits: 0 });
    showResult("success", "Prix prédit par le modèle", `${formatted} $`);
  } catch (err) {
    const network = err instanceof TypeError;   // fetch échoue : API éteinte, mauvaise URL, CORS…
    showResult("error-box", "Prédiction impossible",
      network ? "Impossible de joindre l'API. Vérifiez qu'elle est lancée, que API_URL est correcte et que CORS est activé."
              : err.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Prédire le prix";
  }
});

/* ===== Clic sur « Réinitialiser » ===== */
resetBtn.addEventListener("click", () => {
  form.reset();
  FEATURES.forEach((name) => setFieldError(form.elements[name], ""));
  resultBox.hidden = true;
});