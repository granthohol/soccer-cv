# src/soccer_cv/devices.py

def pick_device(user_device: str | None = None) -> str:
    if user_device:
        return user_device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
