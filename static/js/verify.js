const captureBtn = document.getElementById('captureBtn');
const canvas = document.getElementById('canvas');
const video = document.getElementById('video');
const resultBox = document.getElementById('result');

captureBtn.addEventListener('click', async () => {
    captureBtn.disabled = true;
    resultBox.className = 'result-box';
    resultBox.innerText = 'Verifying...';

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        try {
            const response = await fetch('/verify/', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.verified) {
                resultBox.className = 'result-box result-success';
                resultBox.innerText = `Verified: ${data.name} (${data.employee_id}) — Score: ${data.score.toFixed(2)} — Type: ${data.occlusion_type}`;
            } else {
                resultBox.className = 'result-box result-failure';
                resultBox.innerText = `Not verified. Reason: ${data.reason || 'No match found'} (Score: ${data.score ? data.score.toFixed(2) : 'N/A'})`;
            }
        } catch (err) {
            resultBox.className = 'result-box result-failure';
            resultBox.innerText = 'Error: ' + err.message;
        }

        captureBtn.disabled = false;
    }, 'image/jpeg', 0.9);
});