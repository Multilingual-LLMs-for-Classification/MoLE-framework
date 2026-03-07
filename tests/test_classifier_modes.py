"""
Unit tests for MOEClassifier deployment mode support.

These tests mock the heavy ML components (PromptRoutingSystem, httpx)
so they run without GPU or model weights.

Run:
    cd C:\\FYP\\MoLE-framework
    python -m pytest tests/test_classifier_modes.py -v
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from moe_classifier import MOEClassifier, DeploymentMode, ClassificationResult
from moe_classifier import PipelineConfig, MOETrainer
from moe_classifier.backends.local import LocalBackend
from moe_classifier.backends.remote import RemoteBackend
from moe_classifier.backends.distributed import DistributedBackend


# ======================================================================
# DeploymentMode enum
# ======================================================================

class TestDeploymentMode:
    def test_enum_values(self):
        assert DeploymentMode.LOCAL == "local"
        assert DeploymentMode.REMOTE == "remote"
        assert DeploymentMode.DISTRIBUTED == "distributed"

    def test_string_coercion(self):
        """DeploymentMode('local') should produce DeploymentMode.LOCAL."""
        assert DeploymentMode("local") is DeploymentMode.LOCAL
        assert DeploymentMode("remote") is DeploymentMode.REMOTE
        assert DeploymentMode("distributed") is DeploymentMode.DISTRIBUTED

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            DeploymentMode("invalid")


# ======================================================================
# MOEClassifier — constructor validation
# ======================================================================

class TestMOEClassifierInit:
    def test_default_is_local(self):
        clf = MOEClassifier()
        assert clf.deployment_mode == DeploymentMode.LOCAL

    def test_string_deployment(self):
        clf = MOEClassifier(deployment="remote", coordinator_url="http://x")
        assert clf.deployment_mode == DeploymentMode.REMOTE

    def test_enum_deployment(self):
        clf = MOEClassifier(deployment=DeploymentMode.DISTRIBUTED)
        assert clf.deployment_mode == DeploymentMode.DISTRIBUTED

    def test_invalid_deployment_raises(self):
        with pytest.raises(ValueError, match="Invalid deployment mode"):
            MOEClassifier(deployment="gpu_cluster")

    def test_not_ready_before_initialize(self):
        clf = MOEClassifier()
        assert not clf.is_ready

    def test_classify_before_init_raises(self):
        clf = MOEClassifier()
        with pytest.raises(RuntimeError, match="not initialized"):
            clf.classify(text="hello")

    def test_empty_text_raises(self):
        clf = MOEClassifier()
        clf._backend = MagicMock()
        clf._initialized = True
        with pytest.raises(ValueError, match="non-empty"):
            clf.classify(text="   ")

    def test_repr_not_initialized(self):
        clf = MOEClassifier()
        assert "not initialized" in repr(clf)
        assert "local" in repr(clf)

    def test_repr_remote(self):
        clf = MOEClassifier(deployment="remote", coordinator_url="http://x")
        assert "remote" in repr(clf)


# ======================================================================
# MOEClassifier — remote mode requires coordinator_url
# ======================================================================

class TestRemoteModeValidation:
    def test_remote_without_url_raises(self):
        clf = MOEClassifier(deployment="remote")
        with pytest.raises(ValueError, match="coordinator_url"):
            clf.initialize()

    def test_remote_with_url_creates_backend(self):
        clf = MOEClassifier(
            deployment="remote",
            coordinator_url="http://localhost:8000",
        )
        # Don't actually initialize (would try to connect)
        backend = clf._create_backend()
        assert backend is not None
        assert backend.__class__.__name__ == "RemoteBackend"


# ======================================================================
# LocalBackend
# ======================================================================

class TestLocalBackend:
    def test_initialize_and_classify(self):
        """LocalBackend should call PromptRoutingSystem.route_prompt()."""
        mock_system = MagicMock()
        mock_system.route_prompt.return_value = {
            "language": "english",
            "domain": "finance",
            "task": "rating",
            "result": "4",
            "routing_path": "english -> finance -> rating",
            "expert_confidence": 0.92,
            "domain_probabilities": {"finance": 0.95},
            "raw_response": "4 stars",
        }

        backend = LocalBackend()
        # Directly inject mock system (bypass heavy PromptRoutingSystem init)
        backend._system = mock_system
        backend._initialized = True

        assert backend.is_ready

        result = backend.classify(text="Great product!", description="Rate 1-5.")
        assert isinstance(result, ClassificationResult)
        assert result.result == "4"
        assert result.language == "english"
        assert result.domain == "finance"
        assert result.task == "rating"
        assert result.confidence == 0.92
        assert result.processing_time_ms > 0

        # Verify route_prompt was called with the right prompt
        call_args = mock_system.route_prompt.call_args
        assert "Great product!" in call_args[1]["prompt"] or "Great product!" in str(call_args)

    def test_get_stats(self):
        mock_system = MagicMock()
        mock_system.get_system_stats.return_value = {"total_domains": 1}

        backend = LocalBackend()
        backend._system = mock_system
        backend._initialized = True

        stats = backend.get_stats()
        assert stats["total_domains"] == 1


# ======================================================================
# RemoteBackend
# ======================================================================

class TestRemoteBackend:
    @patch("moe_classifier.backends.remote.httpx.Client")
    def test_initialize_with_token(self, MockClient):
        """Token should be set directly without login call."""
        mock_client = MagicMock()
        mock_client.get.return_value = MagicMock(status_code=200)
        mock_client.headers = {}
        MockClient.return_value = mock_client

        backend = RemoteBackend(
            coordinator_url="http://localhost:8000",
            token="my-jwt-token",
        )
        backend.initialize()

        assert backend.is_ready
        assert mock_client.headers["Authorization"] == "Bearer my-jwt-token"
        # Should NOT have called /auth/token
        mock_client.post.assert_not_called()

    @patch("moe_classifier.backends.remote.httpx.Client")
    def test_initialize_with_credentials(self, MockClient):
        """Credentials should trigger a login call to /api/v1/auth/token."""
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=MagicMock(return_value={"access_token": "jwt-from-login"}),
        )
        mock_client.get.return_value = MagicMock(status_code=200)
        MockClient.return_value = mock_client

        backend = RemoteBackend(
            coordinator_url="http://localhost:8000",
            credentials={"username": "alice", "password": "secret"},
        )
        backend.initialize()

        assert backend.is_ready
        # Should have called /auth/token
        mock_client.post.assert_called_once_with(
            "/api/v1/auth/token",
            data={"username": "alice", "password": "secret"},
        )
        assert mock_client.headers["Authorization"] == "Bearer jwt-from-login"

    @patch("moe_classifier.backends.remote.httpx.Client")
    def test_classify(self, MockClient):
        """classify() should POST to /api/v1/classify."""
        mock_response = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "language": "english",
                "domain": "finance",
                "task": "rating",
                "result": "5",
                "routing_path": "english -> finance -> rating",
                "confidence": 0.95,
            }),
        )
        mock_client = MagicMock()
        mock_client.headers = {}
        mock_client.get.return_value = MagicMock(status_code=200)
        mock_client.post.return_value = mock_response
        MockClient.return_value = mock_client

        backend = RemoteBackend(
            coordinator_url="http://localhost:8000",
            token="test-token",
        )
        backend.initialize()

        # Reset post mock after login-related calls
        mock_client.post.reset_mock()
        mock_client.post.return_value = mock_response

        result = backend.classify(text="Amazing!", description="Rate 1-5.")
        assert isinstance(result, ClassificationResult)
        assert result.result == "5"
        assert result.confidence == 0.95

        # Verify the POST payload structure
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/api/v1/classify"
        payload = call_args[1]["json"]
        assert payload["text"] == "Amazing!"
        assert payload["description"] == "Rate 1-5."


# ======================================================================
# DistributedBackend
# ======================================================================

class TestDistributedBackend:
    @patch("moe_classifier.backends.distributed.httpx.Client")
    def test_gating_then_dispatch(self, MockHttpClient):
        """Distributed should run gating locally then HTTP-dispatch to worker."""
        # Mock the gating result
        mock_gating = MagicMock()
        mock_gating.language = "english"
        mock_gating.domain = "finance"
        mock_gating.task = "rating"
        mock_gating.base_model_key = "llama-2-7b-hf"
        mock_gating.adapter_name = "sentiment_en"
        mock_gating.routing_path = "english -> finance -> rating"

        mock_system = MagicMock()
        mock_system.run_gating.return_value = mock_gating
        mock_system.get_system_stats.return_value = {"total_domains": 1}

        # Mock the HTTP dispatch to worker
        mock_worker_response = MagicMock(
            status_code=200,
            json=MagicMock(return_value={
                "result": "4",
                "confidence": 0.91,
                "raw_response": "4",
            }),
        )
        mock_worker_response.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.return_value = mock_worker_response
        MockHttpClient.return_value = mock_http

        # Create a temp mapping file
        import tempfile
        import os
        mapping = {
            "model_to_worker": {"llama-2-7b-hf": "worker-0"},
            "workers": {
                "worker-0": {
                    "url": "http://10.8.100.21:8001",
                    "base_model_key": "llama-2-7b-hf",
                }
            },
        }
        fd, mapping_path = tempfile.mkstemp(suffix=".json")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(mapping, f)

            backend = DistributedBackend(expert_mapping=mapping_path)
            # Directly inject mock system (bypass heavy PromptRoutingSystem init)
            backend._system = mock_system
            backend._initialized = True
            backend._load_mapping(mapping_path)

            assert backend.is_ready

            result = backend.classify(text="Revenue up 20%.", description="Rate 1-5.")
            assert isinstance(result, ClassificationResult)
            assert result.result == "4"
            assert "gateway:llama-2-7b-hf" in result.routing_path

            # Verify gating was called
            mock_system.run_gating.assert_called_once()

            # Verify HTTP dispatch was made
            mock_http.post.assert_called_once()
            call_url = mock_http.post.call_args[0][0]
            assert "10.8.100.21:8001" in call_url
            assert "/api/v1/expert/classify" in call_url
        finally:
            os.unlink(mapping_path)


# ======================================================================
# End-to-end: MOEClassifier delegates to backend
# ======================================================================

class TestMOEClassifierDelegation:
    def test_local_delegates_to_local_backend(self):
        clf = MOEClassifier(deployment="local")
        backend = clf._create_backend()
        assert backend.__class__.__name__ == "LocalBackend"

    def test_remote_delegates_to_remote_backend(self):
        clf = MOEClassifier(deployment="remote", coordinator_url="http://x")
        backend = clf._create_backend()
        assert backend.__class__.__name__ == "RemoteBackend"

    def test_distributed_delegates_to_distributed_backend(self):
        clf = MOEClassifier(deployment="distributed")
        backend = clf._create_backend()
        assert backend.__class__.__name__ == "DistributedBackend"

    def test_batch_classify_delegates(self):
        """classify_batch should call classify() for each item."""
        clf = MOEClassifier()
        mock_backend = MagicMock()
        mock_backend.is_ready = True
        mock_backend.classify.return_value = ClassificationResult(
            language="en", domain="finance", task="rating",
            result="4", routing_path="en -> finance -> rating",
        )
        clf._backend = mock_backend
        clf._initialized = True

        batch = clf.classify_batch([
            {"text": "Good", "description": "Rate."},
            {"text": "Bad", "description": "Rate."},
        ])
        assert batch.successful == 2
        assert batch.failed == 0
        assert len(batch.items) == 2

    def test_batch_classify_skip_errors(self):
        """Errors should be captured when skip_errors=True."""
        clf = MOEClassifier()
        mock_backend = MagicMock()
        mock_backend.is_ready = True
        mock_backend.classify.side_effect = [
            ClassificationResult(
                language="en", domain="finance", task="rating",
                result="4", routing_path="test",
            ),
            RuntimeError("Model error"),
        ]
        clf._backend = mock_backend
        clf._initialized = True

        batch = clf.classify_batch([
            {"text": "Good"},
            {"text": "Bad"},
        ])
        assert batch.successful == 1
        assert batch.failed == 1
        assert batch.items[1].error == "Model error"


# ======================================================================
# PipelineConfig
# ======================================================================

class TestPipelineConfig:
    def test_defaults_are_none(self):
        config = PipelineConfig()
        assert config.language_model is None
        assert config.expert_registry is None
        assert config.domain_model_dir is None
        assert config.domain_model_name == "xlm-roberta-base"
        assert config.task_router_dir is None
        assert config.task_encoder_name == "xlm-roberta-base"

    def test_custom_values(self):
        config = PipelineConfig(
            domain_model_dir="models/my_domain/",
            expert_registry="config/my_registry.json",
            language_model="models/lid.custom.bin",
        )
        assert config.domain_model_dir == "models/my_domain/"
        assert config.expert_registry == "config/my_registry.json"
        assert config.language_model == "models/lid.custom.bin"
        # Others remain default
        assert config.task_router_dir is None

    def test_pipeline_config_passed_to_local_backend(self):
        """Config should propagate from MOEClassifier -> LocalBackend."""
        config = PipelineConfig(domain_model_dir="models/custom/")
        clf = MOEClassifier(pipeline_config=config)
        backend = clf._create_backend()
        assert isinstance(backend, LocalBackend)
        assert backend._config is config
        assert backend._config.domain_model_dir == "models/custom/"

    def test_pipeline_config_passed_to_distributed_backend(self):
        """Config should propagate from MOEClassifier -> DistributedBackend."""
        config = PipelineConfig(expert_registry="config/custom.json")
        clf = MOEClassifier(deployment="distributed", pipeline_config=config)
        backend = clf._create_backend()
        assert isinstance(backend, DistributedBackend)
        assert backend._config is config
        assert backend._config.expert_registry == "config/custom.json"

    def test_no_config_means_none_on_backend(self):
        """Without PipelineConfig, backend._config should be None."""
        clf = MOEClassifier()
        backend = clf._create_backend()
        assert backend._config is None


# ======================================================================
# MOETrainer
# ======================================================================

class TestMOETrainer:
    def test_default_construction(self):
        trainer = MOETrainer()
        assert isinstance(trainer._config, PipelineConfig)
        assert trainer._system is None

    def test_construction_with_config(self):
        config = PipelineConfig(domain_model_dir="models/custom/")
        trainer = MOETrainer(pipeline_config=config)
        assert trainer._config is config

    def test_train_domain_classifier_delegates(self):
        """train_domain_classifier should call through to PromptRoutingSystem."""
        trainer = MOETrainer()

        # Inject a mock system to bypass heavy initialization
        mock_system = MagicMock()
        mock_system.train_domain_classifier.return_value = {"accuracy": 0.95}
        mock_system.domain_classifier = MagicMock()
        trainer._system = mock_system

        data = [
            {"prompt": "Test text", "domain": "finance"},
            {"prompt": "Another text", "domain": "health"},
        ]
        result = trainer.train_domain_classifier(data, epochs=3)

        # Verify delegation
        mock_system.train_domain_classifier.assert_called_once()
        call_args = mock_system.train_domain_classifier.call_args
        assert call_args[0][0] == data
        assert call_args[1]["epochs"] == 3

        # Verify model was saved
        mock_system.domain_classifier.save_model.assert_called_once()
        assert result["accuracy"] == 0.95

    def test_train_task_routers_delegates(self):
        """train_task_routers should call through to PromptRoutingSystem."""
        trainer = MOETrainer()

        mock_system = MagicMock()
        mock_system.task_classifier = MagicMock()
        trainer._system = mock_system

        data = [
            {"prompt": "Rate this.", "domain": "finance", "task": "rating"},
        ]
        trainer.train_task_routers(data)

        mock_system.train_q_routers.assert_called_once_with(data)
        mock_system.task_classifier.save_models.assert_called_once()

    def test_train_domain_classifier_custom_output(self, tmp_path):
        """When output_dir is specified, model should be saved there."""
        trainer = MOETrainer()

        mock_system = MagicMock()
        mock_system.train_domain_classifier.return_value = {}
        mock_system.domain_classifier = MagicMock()
        trainer._system = mock_system

        out = str(tmp_path / "my_model")
        trainer.train_domain_classifier(
            [{"prompt": "test", "domain": "d"}],
            output_dir=out,
        )

        # Verify save was called with the custom path
        mock_system.domain_classifier.save_model.assert_called_once_with(
            filepath=out,
        )
