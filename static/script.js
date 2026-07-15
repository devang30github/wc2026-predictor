document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();

  const teamA = document.getElementById("team_a").value;
  const teamB = document.getElementById("team_b").value;
  const errorMsg = document.getElementById("error-msg");
  const result = document.getElementById("result");

  errorMsg.textContent = "";
  document.getElementById("placeholder").classList.add("hidden");
  

  if (teamA === teamB) {
    errorMsg.textContent = "Pick two different teams.";
    return;
  }

  const btn = document.querySelector(".predict-btn");
  btn.textContent = "Predicting…";
  btn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ team_a: teamA, team_b: teamB })
    });
    const data = await res.json();

    if (!res.ok) {
      errorMsg.textContent = data.error || "Something went wrong.";
      result.classList.add("hidden");
      return;
    }

    document.getElementById("result-team-a").textContent = data.team_a;
    document.getElementById("result-team-b").textContent = data.team_b;
    document.getElementById("score-a").textContent = data.predicted_score[0];
    document.getElementById("score-b").textContent = data.predicted_score[1];
    document.getElementById("winner-name").textContent = data.winner;

    const probA = (data.prob_a_win * 100).toFixed(1);
    const probB = (data.prob_b_win * 100).toFixed(1);

    document.getElementById("prob-label-a").textContent = data.team_a;
    document.getElementById("prob-label-b").textContent = data.team_b;
    document.getElementById("prob-pct-a").textContent = probA + "%";
    document.getElementById("prob-pct-b").textContent = probB + "%";

    result.classList.remove("hidden");
    document.getElementById("placeholder").classList.add("hidden");

    // Trigger bar fill animation on next frame
    requestAnimationFrame(() => {
      document.getElementById("bar-a").style.width = probA + "%";
      document.getElementById("bar-b").style.width = probB + "%";
    });

  } catch (err) {
    errorMsg.textContent = "Could not reach the prediction server.";
  } finally {
    btn.textContent = "Predict";
    btn.disabled = false;
  }
});