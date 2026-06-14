const methodData = [
  {
    label: "Uniform 8-bit",
    mse: 0.0002064887,
    accuracy: 0.5362903226,
    bits: 8,
    note: "가장 낮은 오차를 보였지만 평균 비트 예산이 8bit라 6bit 공정 비교군은 아니다."
  },
  {
    label: "Uniform 6-bit",
    mse: 0.0035683309,
    accuracy: 0.5357862903,
    bits: 6,
    note: "모든 timestep에 같은 정밀도를 주는 기준선이며, raw Grönwall보다 훨씬 안정적이었다."
  },
  {
    label: "Local-error 6-bit",
    mse: 0.0035683309,
    accuracy: 0.5357862903,
    bits: 6,
    note: "국소 activation 오차만 보면 uniform 6-bit와 같은 스케줄이 선택되어 결과도 동일하다."
  },
  {
    label: "Low→High 6-bit",
    mse: 0.0180502398,
    accuracy: 0.5359375,
    bits: 6,
    note: "뒤쪽 timestep에 높은 비트를 주는 방식이다. 정확도는 비슷하지만 logit 오차가 커졌다."
  },
  {
    label: "High→Low 6-bit",
    mse: 0.0322770666,
    accuracy: 0.5351814516,
    bits: 6,
    note: "초기 단계에 높은 비트를 주지만 후반 4bit 구간에서 최종 logit 오차가 누적된다."
  },
  {
    label: "Empirical oracle",
    mse: 0.0535112903,
    accuracy: 0.5325604839,
    bits: 6,
    note: "작은 calibration subset에서 찾은 경험적 스케줄이며, 전체 평가에서는 안정적이지 않았다."
  },
  {
    label: "Raw Grönwall",
    mse: 0.063672742,
    accuracy: 0.5319556452,
    bits: 6,
    note: "최악 상한의 누적계수 G가 지나치게 커져 앞쪽 8bit, 뒤쪽 4bit로 양극화되었다."
  },
  {
    label: "Uniform 4-bit",
    mse: 0.1039405231,
    accuracy: 0.5296370968,
    bits: 4,
    note: "비트 폭을 낮추면 fake quantization 오차가 급격히 커진다는 H1의 대비군이다."
  }
];

const schedules = [
  {
    id: "uniform",
    label: "Uniform 6-bit",
    mse: 0.0035683309,
    bits: [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
  },
  {
    id: "raw",
    label: "Raw Grönwall",
    mse: 0.063672742,
    bits: [8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4]
  },
  {
    id: "lowHigh",
    label: "Low→High",
    mse: 0.0180502398,
    bits: [4, 4, 4, 6, 6, 6, 6, 6, 6, 8, 8, 8]
  },
  {
    id: "oracle",
    label: "Oracle",
    mse: 0.0535112903,
    bits: [6, 8, 8, 8, 8, 8, 4, 4, 6, 4, 4, 4]
  }
];

const tokenStates = [
  ["[MASK]", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "[MASK]"],
  ["AI", "[MASK]", "[MASK]", "[MASK]", "[MASK]", "[MASK]"],
  ["AI", "앱은", "[MASK]", "[MASK]", "[MASK]", "[MASK]"],
  ["AI", "앱은", "비트", "[MASK]", "[MASK]", "[MASK]"],
  ["AI", "앱은", "비트", "예산을", "[MASK]", "[MASK]"],
  ["AI", "앱은", "비트", "예산을", "조절해", "[MASK]"],
  ["AI", "앱은", "비트", "예산을", "조절해", "완성된다"]
];

let activeSchedule = schedules[0];
let activeStep = 0;
let labTimer = null;
let selectedMethodIndex = 1;

const screens = document.querySelectorAll(".screen");
const tabButtons = document.querySelectorAll(".tab-bar button");
const chart = document.querySelector("#metricChart");
const metricButtons = document.querySelectorAll("[data-metric]");
const schedulePicker = document.querySelector("#schedulePicker");
const labScheduleName = document.querySelector("#labScheduleName");
const labBitStrip = document.querySelector("#labBitStrip");
const scanLine = document.querySelector("#scanLine");
const labStepText = document.querySelector("#labStepText");
const labBitText = document.querySelector("#labBitText");
const tokenRow = document.querySelector("#tokenRow");
const labProgressText = document.querySelector("#labProgressText");
const labMseText = document.querySelector("#labMseText");
const errorMeter = document.querySelector("#errorMeter");
const runLabButton = document.querySelector("#runLab");
const labInsight = document.querySelector("#labInsight");
const methodInspector = document.querySelector("#methodInspector");
const bitSlider = document.querySelector("#bitSlider");
const bitValue = document.querySelector("#bitValue");
const errorValue = document.querySelector("#errorValue");
const derivativeValue = document.querySelector("#derivativeValue");
const gainValue = document.querySelector("#gainValue");
const curveField = document.querySelector("#curveField");
const bitMarker = document.querySelector("#bitMarker");

const labInsights = {
  uniform: "Uniform 6-bit은 모든 timestep을 같은 정밀도로 처리해 가장 안정적인 기준선이 된다.",
  raw: "Raw Grönwall은 초기 단계의 상한 가중치를 과하게 믿어 후반 4bit 구간에서 오차가 크게 남는다.",
  lowHigh: "Low→High는 후반부를 보호하지만, 초반 low-bit 오차가 이후 단계로 전달될 위험이 있다.",
  oracle: "Oracle은 calibration subset에 맞춘 스케줄이라 전체 평가에서는 일반화가 약했다."
};

function setScreen(name) {
  const currentIndex = [...screens].findIndex((screen) => screen.classList.contains("is-active"));
  const nextIndex = [...screens].findIndex((screen) => screen.dataset.screen === name);
  const direction = nextIndex >= currentIndex ? 1 : -1;

  screens.forEach((screen) => {
    screen.classList.toggle("is-active", screen.dataset.screen === name);
    screen.style.setProperty("--screen-x", `${direction * 18}px`);
    if (screen.dataset.screen === name) {
      screen.scrollTop = 0;
    }
  });

  tabButtons.forEach((button) => {
    button.classList.toggle("is-active", button.dataset.target === name);
  });

  history.replaceState(null, "", `#${name}`);
}

function formatMetric(item, metric) {
  if (metric === "mse") {
    return item.mse.toFixed(item.mse < 0.001 ? 6 : 4);
  }
  return `${(item.accuracy * 100).toFixed(2)}%`;
}

function renderChart(metric = "mse") {
  const values = methodData.map((item) => item[metric]);
  const max = Math.max(...values);
  const min = Math.min(...values);

  chart.innerHTML = methodData
    .map((item, index) => {
      const value = item[metric];
      const scale =
        metric === "mse"
          ? Math.max(0.03, value / max)
          : Math.max(0.05, (value - min) / (max - min));
      const isRisk = metric === "mse" ? value > 0.04 : value < 0.533;
      return `
        <article class="bar-row ${isRisk ? "is-risk" : ""} ${index === selectedMethodIndex ? "is-selected" : ""}" data-method-index="${index}" tabindex="0">
          <div class="bar-meta">
            <strong>${item.label}</strong>
            <span>${formatMetric(item, metric)} · ${item.bits}bit avg</span>
          </div>
          <div class="bar-track">
            <i class="bar-fill" style="--scale:${scale}"></i>
          </div>
        </article>
      `;
    })
    .join("");
  renderMethodInspector();
}

function renderMethodInspector() {
  const item = methodData[selectedMethodIndex];
  const relative = item.mse === 0 ? 0 : item.mse / methodData[1].mse;
  methodInspector.innerHTML = `
    <div>
      <span>선택한 방법</span>
      <strong>${item.label}</strong>
      <p>${item.note}</p>
    </div>
    <div class="inspector-stats">
      <article>
        <span>Uniform 6-bit 대비</span>
        <strong>${relative.toFixed(2)}배</strong>
      </article>
      <article>
        <span>정확도</span>
        <strong>${(item.accuracy * 100).toFixed(2)}%</strong>
      </article>
    </div>
  `;
}

function updateBitCalculator() {
  const b = Number(bitSlider.value);
  const error = 2 ** (-2 * b);
  const derivative = -2 * Math.log(2) * error;
  const gain = error - 2 ** (-2 * (b + 2));
  const normalized = (b - 3) / 5;
  const width = Math.max(0, curveField.clientWidth - 18);

  bitValue.textContent = `${b.toFixed(1)} bit`;
  errorValue.textContent = error.toExponential(3);
  derivativeValue.textContent = derivative.toExponential(3);
  gainValue.textContent = gain.toExponential(3);
  bitMarker.style.setProperty("--marker-x", `${normalized * width}px`);
}

function renderSchedulePicker() {
  schedulePicker.innerHTML = schedules
    .map(
      (schedule) => `
        <button type="button" class="${schedule.id === activeSchedule.id ? "is-active" : ""}" data-schedule="${schedule.id}">
          ${schedule.label}
        </button>
      `
    )
    .join("");
}

function renderLab() {
  const bit = activeSchedule.bits[activeStep];
  const progress = (activeStep + 1) / activeSchedule.bits.length;
  const tokenIndex = Math.min(tokenStates.length - 1, Math.floor(progress * tokenStates.length));
  const meterScale = Math.max(0.04, Math.min(1, (activeSchedule.mse / 0.1039405231) * (0.45 + progress * 0.55)));

  labScheduleName.textContent = activeSchedule.label;
  labStepText.textContent = `Step ${String(activeStep + 1).padStart(2, "0")} / 12`;
  labBitText.textContent = `${bit} bit`;
  labProgressText.textContent = `${Math.round(progress * 100)}%`;
  labMseText.textContent = activeSchedule.mse.toFixed(activeSchedule.mse < 0.01 ? 6 : 4);
  scanLine.style.setProperty("--step", activeStep);
  errorMeter.style.setProperty("--scale", meterScale);
  labInsight.textContent = labInsights[activeSchedule.id];

  labBitStrip.innerHTML = activeSchedule.bits
    .map((value, index) => `<span class="bit-cell bit-${value} ${index === activeStep ? "is-current" : ""}">${value}</span>`)
    .join("");

  tokenRow.innerHTML = tokenStates[tokenIndex]
    .map((token) => `<span class="${token === "[MASK]" ? "is-mask" : "is-token"}">${token}</span>`)
    .join("");
}

function runLab() {
  clearInterval(labTimer);
  activeStep = 0;
  renderLab();
  runLabButton.dataset.running = "true";

  labTimer = setInterval(() => {
    activeStep += 1;
    if (activeStep >= activeSchedule.bits.length) {
      clearInterval(labTimer);
      activeStep = activeSchedule.bits.length - 1;
      runLabButton.dataset.running = "false";
    }
    renderLab();
  }, 420);
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => setScreen(button.dataset.target));
});

metricButtons.forEach((button) => {
  button.addEventListener("click", () => {
    metricButtons.forEach((item) => item.classList.remove("is-active"));
    button.classList.add("is-active");
    renderChart(button.dataset.metric);
  });
});

chart.addEventListener("click", (event) => {
  const row = event.target.closest(".bar-row[data-method-index]");
  if (!row) return;
  selectedMethodIndex = Number(row.dataset.methodIndex);
  const activeMetric = document.querySelector("[data-metric].is-active")?.dataset.metric || "mse";
  renderChart(activeMetric);
});

chart.addEventListener("keydown", (event) => {
  if (event.key !== "Enter" && event.key !== " ") return;
  const row = event.target.closest(".bar-row[data-method-index]");
  if (!row) return;
  event.preventDefault();
  selectedMethodIndex = Number(row.dataset.methodIndex);
  const activeMetric = document.querySelector("[data-metric].is-active")?.dataset.metric || "mse";
  renderChart(activeMetric);
});

schedulePicker.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-schedule]");
  if (!button) return;
  activeSchedule = schedules.find((schedule) => schedule.id === button.dataset.schedule) || schedules[0];
  activeStep = 0;
  clearInterval(labTimer);
  runLabButton.dataset.running = "false";
  renderSchedulePicker();
  renderLab();
});

runLabButton.addEventListener("click", runLab);
bitSlider.addEventListener("input", updateBitCalculator);
window.addEventListener("resize", updateBitCalculator);
window.addEventListener("hashchange", () => {
  const target = location.hash.replace("#", "");
  if ([...screens].some((screen) => screen.dataset.screen === target)) {
    setScreen(target);
  }
});

const initialScreen = location.hash.replace("#", "");
if ([...screens].some((screen) => screen.dataset.screen === initialScreen)) {
  setScreen(initialScreen);
}

renderChart();
renderSchedulePicker();
renderLab();
updateBitCalculator();
