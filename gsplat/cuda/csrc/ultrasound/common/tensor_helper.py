import torch
import torch.nn.functional as F

def ensure_tensor(t: torch.Tensor, dtype=torch.float32):
    if t is None: return None
    if not t.is_cuda: t = t.cuda()
    if t.dtype != dtype: t = t.to(dtype=dtype)
    return t.contiguous()

def pack_vec3(t: torch.Tensor):
    if t.shape[-1] == 3:
        t = F.pad(t, (0, 1), mode='constant', value=0.0)
    return t.contiguous()