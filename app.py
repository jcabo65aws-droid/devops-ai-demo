<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DevOps + AI Demo</title>
  <style>
    body { font-family: Arial, sans-serif; max-width: 720px; margin: 4rem auto; }
    h1 { color: #2d6cdf; }
    input, button { padding: 0.6rem; }
  </style>
</head>
<body>
  <h1>🚀 DevOps + AI (Free Tier)</h1>
  <p>Escribe un texto y te digo el sentimiento usando VADER.</p>
  <input id="txt" placeholder="Type something..." style="width:70%"/>
  <button onclick="go()">Predict</button>
  <pre id="out"></pre>
  <script>
    async function go(){
      const text = document.getElementById('txt').value;
      const r = await fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
      document.getElementById('out').textContent = await r.text();
    }
  </script>
</body>
</html>


