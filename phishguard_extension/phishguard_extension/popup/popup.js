const analyseBtn = document.getElementById("analyseBtn");
const resetBtn = document.getElementById("resetBtn");
const result = document.getElementById("result");

analyseBtn.addEventListener("click", () => {
  analyseBtn.textContent = "Analysing...";
  analyseBtn.disabled = true;

  setTimeout(() => {
    analyseBtn.classList.add("hidden");
    result.classList.remove("hidden");
  }, 700);
});

resetBtn.addEventListener("click", () => {
  result.classList.add("hidden");
  analyseBtn.classList.remove("hidden");
  analyseBtn.disabled = false;
  analyseBtn.textContent = "Analyse Email";
});
