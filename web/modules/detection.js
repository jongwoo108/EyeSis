import { state } from "./state.js";
import { initUI } from "./ui.js";
import { getAngleDisplayText } from "./utils.js";

const UI = initUI();

// 박스를 캔버스에 그리기
export function drawDetections(detections, videoWidth, videoHeight) {
    // AI 감지가 비활성화되어 있으면 캔버스 클리어하고 리턴
    if (!state.isDetectionActive) {
        if (state.detectionCtx && state.detectionCanvas) {
            state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
        }
        return;
    }

    if (!state.detectionCtx || !detections || detections.length === 0) {
        // 박스가 없으면 캔버스 클리어
        if (state.detectionCtx) {
            state.detectionCtx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);
        }
        return;
    }

    const ctx = state.detectionCtx;

    // 비디오와 캔버스의 실제 표시 영역 가져오기
    const videoRect = UI.video.getBoundingClientRect();
    const containerRect = UI.video.parentElement.getBoundingClientRect();

    // 캔버스 크기를 컨테이너와 정확히 일치시키기
    state.detectionCanvas.width = containerRect.width;
    state.detectionCanvas.height = containerRect.height;

    // 캔버스 클리어
    ctx.clearRect(0, 0, state.detectionCanvas.width, state.detectionCanvas.height);

    // 비디오의 실제 표시 영역 계산 (object-contain 스타일 고려)
    // 비디오 요소의 실제 렌더링 크기와 위치를 정확히 계산
    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerRect.width / containerRect.height;

    let displayWidth, displayHeight, offsetX, offsetY;

    if (videoAspect > containerAspect) {
        // 비디오가 더 넓음 - 컨테이너 높이에 맞춤
        displayHeight = containerRect.height;
        displayWidth = videoWidth * (containerRect.height / videoHeight);
        offsetX = (containerRect.width - displayWidth) / 2;
        offsetY = 0;
    } else {
        // 비디오가 더 높음 - 컨테이너 너비에 맞춤
        displayWidth = containerRect.width;
        displayHeight = videoHeight * (containerRect.width / videoWidth);
        offsetX = 0;
        offsetY = (containerRect.height - displayHeight) / 2;
    }

    // 디버깅용 (개발 중에만 사용)
    if (window.DEBUG_DETECTIONS) {
        console.log('박스 위치 계산:', {
            videoSize: `${videoWidth}x${videoHeight}`,
            containerSize: `${containerRect.width}x${containerRect.height}`,
            displaySize: `${displayWidth}x${displayHeight}`,
            offset: `(${offsetX}, ${offsetY})`,
            scale: `(${displayWidth / videoWidth}, ${displayHeight / videoHeight})`
        });
    }

    // 각 박스 그리기
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;

        // 원본 비디오 좌표를 표시 영역 좌표로 정확히 변환
        const scaleX = displayWidth / videoWidth;
        const scaleY = displayHeight / videoHeight;

        const scaledX1 = offsetX + x1 * scaleX;
        const scaledY1 = offsetY + y1 * scaleY;
        const scaledX2 = offsetX + x2 * scaleX;
        const scaledY2 = offsetY + y2 * scaleY;

        // 색상 설정
        let color;
        switch (detection.color) {
            case 'red':
                color = '#ef4444'; // 빨간색 (범죄자)
                break;
            case 'green':
                color = '#10b981'; // 초록색 (일반인)
                break;
            case 'yellow':
            default:
                color = '#eab308'; // 노란색 (미확인)
                break;
        }

        // 투명도 설정 (부드러운 전환용)
        ctx.globalAlpha = detection.opacity !== undefined ? detection.opacity : 1.0;

        // 박스 그리기 (더 두꺼운 선으로 강조)
        ctx.strokeStyle = color;
        ctx.lineWidth = 4;
        ctx.strokeRect(scaledX1, scaledY1, scaledX2 - scaledX1, scaledY2 - scaledY1);

        // 박스 모서리 강조 (선택적)
        const cornerSize = 8;
        ctx.lineWidth = 3;
        // 왼쪽 위
        ctx.beginPath();
        ctx.moveTo(scaledX1, scaledY1 + cornerSize);
        ctx.lineTo(scaledX1, scaledY1);
        ctx.lineTo(scaledX1 + cornerSize, scaledY1);
        ctx.stroke();
        // 오른쪽 위
        ctx.beginPath();
        ctx.moveTo(scaledX2 - cornerSize, scaledY1);
        ctx.lineTo(scaledX2, scaledY1);
        ctx.lineTo(scaledX2, scaledY1 + cornerSize);
        ctx.stroke();
        // 왼쪽 아래
        ctx.beginPath();
        ctx.moveTo(scaledX1, scaledY2 - cornerSize);
        ctx.lineTo(scaledX1, scaledY2);
        ctx.lineTo(scaledX1 + cornerSize, scaledY2);
        ctx.stroke();
        // 오른쪽 아래
        ctx.beginPath();
        ctx.moveTo(scaledX2 - cornerSize, scaledY2);
        ctx.lineTo(scaledX2, scaledY2);
        ctx.lineTo(scaledX2, scaledY2 - cornerSize);
        ctx.stroke();

        // 텍스트 정보 준비
        const angleText = detection.angle_type && detection.angle_type !== 'front' && detection.angle_type !== 'unknown'
            ? ` [${getAngleDisplayText(detection.angle_type)}]`
            : '';
        const nameText = `${detection.name} (${detection.confidence}%)`;
        const fullText = nameText + angleText;

        // 범죄자인 경우 경고 텍스트 추가
        let warningText = '';
        if (detection.status === 'criminal') {
            warningText = '⚠️ WARNING';
        }

        // 텍스트 위치 계산 (박스 위쪽에 배치)
        ctx.font = 'bold 16px Arial';
        const nameMetrics = ctx.measureText(nameText);
        const fullMetrics = ctx.measureText(fullText);
        const warningMetrics = warningText ? ctx.measureText(warningText) : { width: 0 };

        const textPadding = 6;
        const lineHeight = 22;
        const maxTextWidth = Math.max(fullMetrics.width, warningMetrics.width);
        const textBoxWidth = maxTextWidth + (textPadding * 2);
        const textBoxHeight = warningText ? lineHeight * 2 + textPadding : lineHeight + textPadding;

        // 텍스트가 화면 밖으로 나가지 않도록 조정
        let textX = scaledX1;
        if (textX + textBoxWidth > state.detectionCanvas.width) {
            textX = state.detectionCanvas.width - textBoxWidth;
        }
        if (textX < 0) {
            textX = 0;
        }

        let textY = scaledY1 - textBoxHeight - 4;
        // 텍스트가 화면 위로 나가면 박스 아래에 배치
        if (textY < 0) {
            textY = scaledY2 + 4;
        }

        // 텍스트 배경 그리기 (반투명 배경)
        ctx.fillStyle = color + 'CC'; // 80% 투명도 (기본)
        // globalAlpha가 이미 적용되어 있으므로 배경색의 알파값은 굳이 조절 안 해도 되지만, 
        // color + 'CC'는 이미 알파가 있는 hex string일 수 있음.
        // 하지만 ctx.globalAlpha가 전체 투명도를 조절하므로 괜찮음.
        ctx.fillRect(textX, textY, textBoxWidth, textBoxHeight);

        // 텍스트 그리기
        ctx.fillStyle = '#ffffff';
        let currentY = textY + lineHeight;

        // 경고 텍스트 먼저 (있는 경우)
        if (warningText) {
            ctx.font = 'bold 18px Arial';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(warningText, textX + textPadding, currentY - 4);
            currentY += lineHeight;
        }

        // 이름과 신뢰도
        ctx.font = 'bold 16px Arial';
        ctx.fillStyle = '#ffffff';
        ctx.fillText(nameText, textX + textPadding, currentY);

        // 각도 정보 (있는 경우, 같은 줄에)
        if (angleText) {
            ctx.font = '14px Arial';
            ctx.fillStyle = '#f0f0f0';
            const angleX = textX + textPadding + nameMetrics.width + 4;
            ctx.fillText(angleText, angleX, currentY);
        }

        // 투명도 초기화 (다음 루프를 위해)
        ctx.globalAlpha = 1.0;
    });
}

// 현재 비디오 프레임 캡처 (Base64)
export function captureVideoFrame() {
    if (!UI.video) {
        console.warn("⚠️ 비디오 요소를 찾을 수 없습니다");
        return null;
    }

    // 비디오가 종료된 경우는 processRealtimeDetection에서 처리하므로 여기서는 조용히 반환
    if (UI.video.ended) {
        return null;
    }

    // 일시정지된 경우도 조용히 반환 (메시지는 processRealtimeDetection에서 처리)
    if (UI.video.paused) {
        return null;
    }

    if (UI.video.videoWidth === 0 || UI.video.videoHeight === 0) {
        console.warn("⚠️ 비디오 크기가 0입니다. 비디오가 로드되지 않았을 수 있습니다");
        return null;
    }

    const ctx = state.videoCanvas.getContext('2d');
    state.videoCanvas.width = UI.video.videoWidth;
    state.videoCanvas.height = UI.video.videoHeight;

    ctx.drawImage(UI.video, 0, 0);
    return state.videoCanvas.toDataURL('image/jpeg', 0.7);
}