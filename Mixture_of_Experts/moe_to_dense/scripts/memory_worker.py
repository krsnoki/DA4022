"""Measure native resident memory in an isolated inference process."""
import json
import sys
import torch
import psutil
from ..synthetic import TinyDenseLM, TinyMoELM
from ..concatenation import DenseSwiGLU


def main():
    path, warmup, repeats = sys.argv[1:]
    payload = torch.load(path, map_location='cpu', weights_only=True)
    torch.set_num_threads(payload['threads'])
    info = payload['info']
    teacher = TinyMoELM(**info)
    if payload['dense']:
        model = TinyDenseLM(teacher, DenseSwiGLU(info['hidden_size'], info['intermediate_size']))
        del teacher
    else:
        model = teacher
    model.load_state_dict(payload['state'])
    sample = payload['sample']
    del payload
    model.eval()
    with torch.no_grad():
        for _ in range(int(warmup) + int(repeats)):
            model(sample)
    if sys.platform == 'win32':
        peak = psutil.Process().memory_info().peak_wset
    else:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
    print(json.dumps({'peak_memory_mib': peak / 2**20, 'peak_memory_scope': 'isolated process lifetime peak RSS/working set (includes startup, loading, warmup and inference)'}))


if __name__ == '__main__':
    main()
