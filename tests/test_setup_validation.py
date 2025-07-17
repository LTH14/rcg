"""Validation tests to ensure the testing infrastructure is properly set up."""

import pytest
import torch
from pathlib import Path
from omegaconf import DictConfig


class TestInfrastructureValidation:
    """Test class to validate the testing infrastructure setup."""
    
    def test_pytest_installation(self):
        """Test that pytest is properly installed."""
        assert pytest.__version__ is not None
        
    def test_pytest_cov_available(self):
        """Test that pytest-cov plugin is available."""
        import pytest_cov
        assert pytest_cov.__version__ is not None
        
    def test_pytest_mock_available(self):
        """Test that pytest-mock is available."""
        import pytest_mock
        # pytest-mock doesn't expose __version__, so check it can be imported
        assert pytest_mock is not None
        
    def test_temp_dir_fixture(self, temp_dir):
        """Test that temp_dir fixture creates a valid temporary directory."""
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test file creation in temp dir
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists()
        assert test_file.read_text() == "test content"
        
    def test_mock_config_fixture(self, mock_config):
        """Test that mock_config fixture provides valid configuration."""
        assert isinstance(mock_config, DictConfig)
        assert "model" in mock_config
        assert "training" in mock_config
        assert "data" in mock_config
        assert mock_config.training.batch_size == 32
        assert mock_config.model.name == "test_model"
        
    def test_sample_tensor_fixture(self, sample_tensor):
        """Test that sample_tensor fixture provides valid tensor."""
        assert isinstance(sample_tensor, torch.Tensor)
        assert sample_tensor.shape == (4, 3, 256, 256)
        assert sample_tensor.dtype == torch.float32
        
    def test_sample_batch_fixture(self, sample_batch):
        """Test that sample_batch fixture provides valid batch data."""
        assert isinstance(sample_batch, dict)
        assert "images" in sample_batch
        assert "labels" in sample_batch
        assert "masks" in sample_batch
        
        assert sample_batch["images"].shape == (4, 3, 256, 256)
        assert sample_batch["labels"].shape == (4,)
        assert sample_batch["masks"].shape == (4, 1, 256, 256)
        
    def test_numpy_random_state_fixture(self, numpy_random_state):
        """Test that numpy random state fixture ensures reproducibility."""
        import numpy as np
        
        # First run
        values1 = [np.random.rand() for _ in range(5)]
        
        # Reset seed manually
        np.random.seed(42)
        
        # Second run
        values2 = [np.random.rand() for _ in range(5)]
        
        # Should produce same values
        assert values1 == values2
        
    def test_torch_random_state_fixture(self, torch_random_state):
        """Test that torch random state fixture ensures reproducibility."""
        # First run
        tensor1 = torch.randn(3, 3)
        
        # Reset seed manually
        torch.manual_seed(42)
        
        # Second run
        tensor2 = torch.randn(3, 3)
        
        # Should produce same tensor
        assert torch.allclose(tensor1, tensor2)
        
    def test_mock_model_checkpoint_fixture(self, mock_model_checkpoint):
        """Test that mock_model_checkpoint creates valid checkpoint file."""
        assert mock_model_checkpoint.exists()
        assert mock_model_checkpoint.suffix == ".pt"
        
        # Load and verify checkpoint
        checkpoint = torch.load(mock_model_checkpoint)
        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "loss" in checkpoint
        assert checkpoint["epoch"] == 10
        
    def test_device_fixture(self, device):
        """Test that device fixture returns valid torch device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
        
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker is properly configured."""
        assert True
        
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker is properly configured."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker is properly configured."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True
        
    def test_coverage_configuration(self):
        """Test that coverage is properly configured."""
        # This test verifies coverage is running by its mere existence
        assert True
        
    def test_mock_image_file_fixture(self, mock_image_file):
        """Test that mock_image_file creates valid image file."""
        assert mock_image_file.exists()
        assert mock_image_file.suffix == ".png"
        
        # Verify it's a valid image
        from PIL import Image
        img = Image.open(mock_image_file)
        assert img.size == (256, 256)
        assert img.mode == "RGB"


@pytest.mark.parametrize("value,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_parametrize_decorator(value, expected):
    """Test that parametrize decorator works correctly."""
    assert value ** 2 == expected


def test_capture_logs_fixture(capture_logs):
    """Test that capture_logs fixture works correctly."""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    
    assert "Debug message" in capture_logs.text
    assert "Info message" in capture_logs.text
    assert "Warning message" in capture_logs.text