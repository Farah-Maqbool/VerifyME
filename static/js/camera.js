const video = document.getElementById('video');

async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (err) {
        document.getElementById('result').innerText = 'Could not access webcam: ' + err.message;
    }
}

startCamera();