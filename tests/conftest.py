"""Shared pytest fixtures and configuration for the RCG project."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch
import numpy as np
from omegaconf import OmegaConf, DictConfig


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_config() -> DictConfig:
    """Provide a mock configuration for testing."""
    config = {
        "model": {
            "name": "test_model",
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "seed": 42,
        },
        "data": {
            "dataset": "test_dataset",
            "num_workers": 2,
            "pin_memory": True,
        },
        "logging": {
            "log_dir": "logs",
            "save_interval": 100,
            "val_interval": 50,
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Provide a sample tensor for testing."""
    return torch.randn(4, 3, 256, 256)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Provide a sample batch of data for testing."""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 256, 256),
        "labels": torch.randint(0, 10, (batch_size,)),
        "masks": torch.ones(batch_size, 1, 256, 256),
    }


@pytest.fixture
def numpy_random_state():
    """Fixture to ensure reproducible numpy random state."""
    original_state = np.random.get_state()
    np.random.seed(42)
    yield
    np.random.set_state(original_state)


@pytest.fixture
def torch_random_state():
    """Fixture to ensure reproducible torch random state."""
    original_state = torch.get_rng_state()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        original_cuda_state = torch.cuda.get_rng_state()
        torch.cuda.manual_seed(42)
    yield
    torch.set_rng_state(original_state)
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(original_cuda_state)


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pt"
    checkpoint = {
        "epoch": 10,
        "model_state_dict": {"layer1.weight": torch.randn(10, 10)},
        "optimizer_state_dict": {"param_groups": [{"lr": 0.001}]},
        "loss": 0.5,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def mock_image_file(temp_dir: Path) -> Path:
    """Create a mock image file for testing."""
    import numpy as np
    from PIL import Image
    
    image_path = temp_dir / "test_image.png"
    image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    image.save(image_path)
    return image_path


@pytest.fixture
def device() -> torch.device:
    """Return the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(autouse=True)
def cleanup_cuda():
    """Automatically clean up CUDA cache after each test."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture
def mock_environment_variables() -> Generator[Dict[str, str], None, None]:
    """Temporarily set environment variables for testing."""
    original_env = os.environ.copy()
    test_env = {
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).parent.parent),
    }
    os.environ.update(test_env)
    yield test_env
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def capture_logs(caplog):
    """Fixture to capture and assert on log messages."""
    with caplog.at_level("DEBUG"):
        yield caplog


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Skip GPU tests if CUDA is not available
        if "gpu" in item.keywords and not torch.cuda.is_available():
            skip_gpu = pytest.mark.skip(reason="GPU not available")
            item.add_marker(skip_gpu)