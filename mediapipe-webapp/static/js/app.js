document.getElementById('fps').addEventListener('change', function() {
    const fps = this.value;
    fetch('/set_fps', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `fps=${fps}`,
    });
});

document.getElementById('blur_type').addEventListener('change', function() {
    const blur_type = this.value;
    fetch('/set_blur_type', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `blur_type=${blur_type}`,
    });
});

function updatePerformanceStats() {
    fetch('/performance_stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('cpu_usage').innerText = 'CPU Usage: ' + data.cpu_usage + '%';
            document.getElementById('memory_usage').innerText = 'Memory Usage: ' + data.memory_usage + '%';
            document.getElementById('blur_time').innerText = 'Blur Time: ' + data.blur_time_ms + ' ms';
            document.getElementById('cpu_response_time').innerText = 'CPU Response Time: ' + data.cpu_response_time_ms + ' ms';
        })
        .catch(error => console.error('Error fetching performance stats:', error));
}

function updateAverageMetrics() {
    fetch('/performance_stats')
        .then(response => response.json())
        .then(data => {
            document.getElementById('avg_blur_time').innerText = 'Average Blur Time (last 30s): ' + data.blur_time_ms + ' ms';
            document.getElementById('avg_cpu_response_time').innerText = 'Average CPU Response Time (last 30s): ' + data.cpu_response_time_ms + ' ms';
            document.getElementById('avg_cpu_usage').innerText = 'Average CPU Usage (last 30s): ' + data.avg_cpu_usage + ' %';
        })
        .catch(error => console.error('Error fetching average metrics:', error));
}

// Update performance stats every 2 seconds
setInterval(updatePerformanceStats, 2000);

// Update average metrics every 30 seconds
setInterval(updateAverageMetrics, 30000);
