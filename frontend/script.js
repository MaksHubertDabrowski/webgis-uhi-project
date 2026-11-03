async function loadStats() {
            try {
                const response = await fetch('http://127.0.0.1:8000/api/stats');
                const data = await response.json();
                
                document.getElementById('val-mean').innerText = data.mean_temp + " °C";
                document.getElementById('val-max').innerText = data.max_temp + " °C";
                document.getElementById('val-min').innerText = data.min_temp + " °C";
                
                // Opcjonalnie: pokoloruj wyniki
                document.getElementById('val-max').style.color = 'red';
                document.getElementById('val-min').style.color = 'blue';
                
            } catch (error) {
                console.error("Błąd pobierania statystyk:", error);
                document.querySelector('.stats-panel').innerHTML += "<br><small style='color:red'>Błąd danych</small>";
            }
        }

        loadHeatmap();
        loadStats(); 