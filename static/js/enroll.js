const enrollBtn = document.getElementById('enrollBtn');
const canvas = document.getElementById('canvas');
const resultBox = document.getElementById('result');
const employeeIdInput = document.getElementById('employeeId');
const employeeNameInput = document.getElementById('employeeName');

enrollBtn.addEventListener('click', async () => {
    const employeeId = employeeIdInput.value.trim();
    const employeeName = employeeNameInput.value.trim();

    if (!employeeId || !employeeName) {
        resultBox.className = 'result-box result-failure';
        resultBox.innerText = 'Please enter both Employee ID and Name.';
        return;
    }

    enrollBtn.disabled = true;
    resultBox.className = 'result-box';
    resultBox.innerText = 'Enrolling...';

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('employee_id', employeeId);
        formData.append('name', employeeName);
        formData.append('file', blob, 'enroll.jpg');

        try {
            const response = await fetch('/enroll/', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                resultBox.className = 'result-box result-success';
                resultBox.innerText = `Enrolled successfully: ${data.employee_id}`;
                employeeIdInput.value = '';
                employeeNameInput.value = '';
            } else {
                resultBox.className = 'result-box result-failure';
                resultBox.innerText = `Error: ${data.detail || 'Enrollment failed'}`;
            }
        } catch (err) {
            resultBox.className = 'result-box result-failure';
            resultBox.innerText = 'Error: ' + err.message;
        }

        enrollBtn.disabled = false;
    }, 'image/jpeg', 0.9);
});