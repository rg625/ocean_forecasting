# models/config_classes.py
from omegaconf import MISSING
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field
from .cnn import (
    TransformerConfig as ModelTransformerConfig,
)  # Alias to avoid naming conflict


@dataclass
class NormalizationConfig:
    type: str = "MeanStdNormalizer"  # Default to a sensible choice
    sim: int = 0


@dataclass
class DataConfig:
    dataset_type: str = MISSING
    data_dir: str = MISSING
    train_file: str = MISSING
    val_file: str = MISSING
    test_file: str = MISSING
    input_sequence_length: int = MISSING
    max_sequence_length: int = MISSING
    # A dictionary mapping variable names to their channel counts is crucial.
    variables: Dict[str, int] = field(default_factory=dict)
    static_variables: Dict[str, int] = field(default_factory=dict)
    quantile_range: Tuple[float, float] = (2.5, 97.5)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


@dataclass
class ModelConfig:
    height: int = MISSING
    width: int = MISSING
    input_channels: int = MISSING
    hidden_dims: List[int] = field(default_factory=list)
    block_size: int = MISSING
    kernel_size: int = MISSING
    conv_kwargs: Dict[str, Any] = field(default_factory=dict)
    latent_dim: int = MISSING
    # Use the specific TransformerConfig dataclass for type safety
    transformer: ModelTransformerConfig = field(default_factory=ModelTransformerConfig)
    predict_re: bool = False
    # Explicitly control if the Reynolds loss regularizes the main model.
    re_grad_enabled: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    use_checkpoint: bool = False
    num_epochs: int = MISSING
    patience: int = 10
    random_sequence_length: bool = True
    save_latest_every: int = 1
    num_visual_batches: int = 1


@dataclass
class LossConfig:
    alpha: float = MISSING
    beta: float = MISSING
    re_weight: float = MISSING
    weighting_type: str = MISSING
    sigma_blur: Optional[float] = None


@dataclass
class LRSchedulerConfig:
    lr: float = MISSING
    warmup: int = MISSING
    decay: int = MISSING
    final_lr: float = MISSING


@dataclass
class MetricConfig:
    mode: str = MISSING
    variable_mode: str = MISSING
    variable_name: Optional[str] = None


@dataclass
class Config:
    output_dir: str = MISSING
    ckpt: Optional[str] = None
    log_epoch: int = 1

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    metric: Optional[MetricConfig] = None
