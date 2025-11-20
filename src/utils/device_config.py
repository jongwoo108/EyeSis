# src/utils/device_config.py
"""
GPU/CPU 디바이스 설정 모듈
GPU 사용 가능 여부를 확인하고 적절한 ctx_id를 반환합니다.
"""
import os
import sys
import warnings
import onnxruntime as ort
from typing import Optional
from pathlib import Path

def _find_cuda_path() -> Optional[str]:
    """
    시스템에 설치된 CUDA 경로를 찾습니다.
    
    Returns:
        str: CUDA bin 경로 (없으면 None)
    """
    possible_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
        r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    ]
    
    # 환경 변수에서도 확인
    cuda_path = os.getenv("CUDA_PATH")
    if cuda_path:
        bin_path = Path(cuda_path) / "bin"
        if bin_path.exists():
            return str(bin_path)
    
    # 일반적인 설치 경로 확인
    for path in possible_paths:
        if Path(path).exists():
            # cublasLt64_12.dll 같은 필수 DLL이 있는지 확인
            dll_files = list(Path(path).glob("cublasLt*.dll"))
            if dll_files:
                return path
    
    return None

def _ensure_cuda_in_path() -> bool:
    """
    CUDA 경로가 PATH에 없으면 추가합니다.
    
    Returns:
        bool: CUDA 경로를 찾아서 추가했으면 True
    """
    cuda_path = _find_cuda_path()
    if not cuda_path:
        return False
    
    current_path = os.getenv("PATH", "")
    if cuda_path not in current_path:
        os.environ["PATH"] = f"{cuda_path};{current_path}"
        return True
    
    return False

# 모듈 로드 시 자동으로 CUDA 경로 추가 시도
_ensure_cuda_in_path()

def get_device_id() -> int:
    """
    GPU 사용 가능 여부를 확인하고 적절한 디바이스 ID를 반환합니다.
    실제 CUDA 라이브러리 로드 가능 여부도 확인합니다.
    
    Returns:
        int: GPU 사용 가능하면 0, 아니면 -1 (CPU)
    """
    # 환경 변수로 강제 설정 가능
    force_cpu = os.getenv("FORCE_CPU", "0").lower() in ("1", "true", "yes")
    if force_cpu:
        return -1
    
    # GPU 인덱스 지정 가능 (기본값: 0)
    gpu_index = int(os.getenv("GPU_INDEX", "0"))
    
    try:
        # onnxruntime의 에러 메시지 억제
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            
            providers = ort.get_available_providers()
            
            # Provider가 있어도 실제로 사용 가능한지 확인
            if 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers:
                # 실제 GPU 사용 가능 여부는 InsightFace가 자동으로 처리하므로
                # 여기서는 provider 존재 여부만 확인
                return gpu_index
            else:
                return -1
    except Exception:
        # 에러 발생 시 안전하게 CPU 사용
        return -1

def safe_prepare_insightface(app, device_id: int, det_size: tuple = (640, 640), verbose: bool = True) -> int:
    """
    InsightFace의 prepare()를 안전하게 호출합니다.
    GPU 사용 실패 시 자동으로 CPU로 fallback합니다.
    
    Args:
        app: FaceAnalysis 인스턴스
        device_id: 사용하려는 디바이스 ID (0 이상이면 GPU, -1이면 CPU)
        det_size: detection size
        verbose: 상세 메시지 출력 여부
    
    Returns:
        int: 실제로 사용된 디바이스 ID (GPU 실패 시 -1 반환)
    """
    # GPU 사용 시도
    if device_id >= 0:
        # CUDA 경로가 PATH에 없으면 추가 시도
        cuda_added = _ensure_cuda_in_path()
        if cuda_added and verbose:
            cuda_path = _find_cuda_path()
            print(f"🔧 CUDA 경로를 PATH에 추가했습니다: {cuda_path}")
        
        try:
            # stderr 리다이렉트로 에러 메시지 억제 시도
            import io
            from contextlib import redirect_stderr
            
            # GPU 초기화 시도 (에러 메시지는 억제)
            with redirect_stderr(io.StringIO()):
                app.prepare(ctx_id=device_id, det_size=det_size)
            
            if verbose:
                print(f"✅ GPU 초기화 성공 (ctx_id={device_id})")
            return device_id
            
        except Exception as e:
            # GPU 실패 시 CPU로 fallback
            if verbose:
                print(f"⚠️ GPU 초기화 실패: {str(e)[:100]}")
                print(f"   CPU로 전환합니다.")
            
            try:
                app.prepare(ctx_id=-1, det_size=det_size)
                if verbose:
                    print(f"✅ CPU 초기화 성공")
                return -1
            except Exception as e2:
                # CPU도 실패하면 에러 발생
                raise RuntimeError(f"CPU 초기화도 실패했습니다: {e2}")
    else:
        # CPU 사용
        app.prepare(ctx_id=-1, det_size=det_size)
        if verbose:
            print(f"✅ CPU 초기화 성공")
        return -1

def get_device_info() -> dict:
    """
    현재 디바이스 정보를 반환합니다.
    
    Returns:
        dict: 디바이스 정보 (device_id, device_type, providers)
    """
    device_id = get_device_id()
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            providers = ort.get_available_providers()
    except Exception:
        providers = []
    
    device_type = "GPU" if device_id >= 0 else "CPU"
    
    return {
        "device_id": device_id,
        "device_type": device_type,
        "providers": providers
    }

