import sys
from pathlib import Path
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def skip_gpu_tests_without_cuda(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("gpu") is None:
        return
    if torch.cuda.is_available():
        return

    print("Skipping GPU test: no GPU detected.")
    pytest.skip("no GPU detected")