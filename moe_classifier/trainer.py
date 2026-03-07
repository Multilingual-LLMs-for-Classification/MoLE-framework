"""
MOETrainer — training utilities for MoLE pipeline components.

Allows users to train the domain classifier and task routers on their
own labeled data, then save checkpoints for use with :class:`PipelineConfig`.

Example::

    from moe_classifier import MOETrainer

    trainer = MOETrainer()

    # Train the domain classifier on labeled prompts
    trainer.train_domain_classifier(
        training_data=[
            {"prompt": "Revenue missed expectations.", "domain": "finance"},
            {"prompt": "Patient showed improvement.", "domain": "health"},
        ],
        epochs=5,
        output_dir="models/my_domain_classifier/",
    )

    # Train the task routers
    trainer.train_task_routers(
        training_data=[
            {"prompt": "Rate this product 1-5.", "domain": "finance", "task": "rating"},
            {"prompt": "Classify this article.", "domain": "finance", "task": "news"},
        ],
        output_dir="models/my_task_routers/",
    )
"""

from pathlib import Path
from typing import Dict, List, Optional

from .pipeline_config import PipelineConfig


class MOETrainer:
    """
    Train MoLE pipeline components on custom data.

    Creates a ``PromptRoutingSystem`` in training mode and delegates
    to its built-in training methods.

    Parameters
    ----------
    pipeline_config : PipelineConfig, optional
        Override component paths (custom registry, encoder name, etc.).
        If ``None``, uses all framework defaults.
    """

    def __init__(self, pipeline_config: Optional[PipelineConfig] = None) -> None:
        self._config = pipeline_config or PipelineConfig()
        self._system = None

    def _ensure_system(self) -> None:
        """Lazily create PromptRoutingSystem in training mode."""
        if self._system is not None:
            return

        try:
            from moe_router.gating.components.routing_system import PromptRoutingSystem
        except ImportError as exc:
            raise RuntimeError(
                "moe_router package not found.  Make sure you are running "
                "from the MoLE-framework directory and the package is "
                "installed (pip install -e .)."
            ) from exc

        kwargs = {}
        if self._config.language_model:
            kwargs["language_model"] = self._config.language_model
        if self._config.expert_registry:
            kwargs["expert_registry_override"] = self._config.expert_registry
        if self._config.domain_model_dir:
            kwargs["domain_model_dir"] = self._config.domain_model_dir
        if self._config.domain_model_name != "xlm-roberta-base":
            kwargs["domain_model_name"] = self._config.domain_model_name
        if self._config.task_router_dir:
            kwargs["task_router_dir"] = self._config.task_router_dir
        if self._config.task_encoder_name != "xlm-roberta-base":
            kwargs["task_encoder_name"] = self._config.task_encoder_name

        self._system = PromptRoutingSystem(
            training_mode=True,
            coordinator_only=True,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Domain classifier training
    # ------------------------------------------------------------------

    def train_domain_classifier(
        self,
        training_data: List[Dict],
        *,
        epochs: int = 3,
        lr: float = 2e-5,
        batch_size: int = 32,
        val_split: float = 0.1,
        freeze_encoder: bool = True,
        output_dir: Optional[str] = None,
        **kwargs,
    ) -> Dict:
        """
        Train the XLM-RoBERTa domain classifier on labeled prompts.

        Parameters
        ----------
        training_data : list of dict
            Each dict must have ``"prompt"`` and ``"domain"`` keys.
            Example: ``[{"prompt": "Q3 earnings...", "domain": "finance"}]``
        epochs : int
            Number of training epochs (default: 3).
        lr : float
            Learning rate (default: 2e-5).
        batch_size : int
            Batch size (default: 32).
        val_split : float
            Fraction of data to use for validation (default: 0.1).
        freeze_encoder : bool
            If ``True``, freeze the XLM-R encoder and only train the
            classification head (faster, default).
        output_dir : str, optional
            Directory to save the trained model.  If ``None``, saves to
            the default framework directory.

        Returns
        -------
        dict
            Training summary with loss and accuracy information.
        """
        self._ensure_system()

        result = self._system.train_domain_classifier(
            training_data,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            val_split=val_split,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

        # Save to custom directory if specified
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            self._system.domain_classifier.save_model(filepath=str(out))
            print(f"[MOETrainer] Domain classifier saved to {out}")
        else:
            self._system.domain_classifier.save_model()
            print("[MOETrainer] Domain classifier saved to default location.")

        return result

    # ------------------------------------------------------------------
    # Task router training
    # ------------------------------------------------------------------

    def train_task_routers(
        self,
        training_data: List[Dict],
        *,
        val_split: float = 0.1,
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Train Q-learning task routers on labeled prompts.

        Parameters
        ----------
        training_data : list of dict
            Each dict must have ``"prompt"``, ``"domain"``, and ``"task"`` keys.
            Example: ``[{"prompt": "Rate 1-5.", "domain": "finance", "task": "rating"}]``
        val_split : float
            Fraction of data for validation (default: 0.1).
        output_dir : str, optional
            Directory to save trained routers.  If ``None``, saves to
            the default framework directory.
        """
        self._ensure_system()

        self._system.train_q_routers(training_data)

        # Save models
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            # Override the model_dir and save
            self._system.task_classifier.model_dir = out
            self._system.task_classifier.save_models()
            print(f"[MOETrainer] Task routers saved to {out}")
        else:
            self._system.task_classifier.save_models()
            print("[MOETrainer] Task routers saved to default location.")
