document.getElementById("btnTranscribe").addEventListener("click", async () => {
  const input = document.getElementById("audioFile");
  if (input.files.length === 0) {
    alert("Por favor, selecciona un archivo de audio.");
    return;
  }

  const file = input.files[0];
  const form = new FormData();
  form.append("file", file);

  const resultEl = document.getElementById("result");
  resultEl.textContent = "Transcribiendo…";

  try {
    const res = await fetch("/transcribe", {
      method: "POST",
      body: form,
    });
    if (!res.ok) throw new Error(`Error ${res.status}`);
    const data = await res.json();
    resultEl.textContent = data.text;
  } catch (err) {
    resultEl.textContent = "❌ Falló la transcripción.";
    console.error(err);
  }
});
