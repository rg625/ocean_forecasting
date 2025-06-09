from omegaconf import MISSING
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass, field  # Import field for default_factory


@dataclass
class NormalizationConfig:
    type: str = MISSING
    sim: int = 0  # Default to 0, or MISSING if always required


@dataclass
class DataConfig:
    dataset_type: str = MISSING
    data_dir: str = MISSING
    train_file: str = MISSING
    val_file: str = MISSING
    test_file: str = MISSING
    input_sequence_length: int = MISSING
    max_sequence_length: int = MISSING
    variables: Optional[List[str]] = None
    quantile_range: Tuple[float, float] = (2.5, 97.5)  # Default value
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
    transformer: Dict[str, Any] = field(default_factory=dict)
    predict_re: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = MISSING
    use_checkpoint: bool = False
    num_epochs: int = MISSING
    patience: int = 10  # Default patience
    random_sequence_length: bool = True


@dataclass
class LossConfig:
    alpha: float = MISSING
    beta: float = MISSING
    re_weight: float = MISSING
    weighting_type: str = MISSING
    sigma_blur: Optional[float] = None  # Changed from ~ to None directly


@dataclass
class LRSchedulerConfig:
    lr: float = MISSING
    warmup: int = MISSING
    decay: int = MISSING
    final_lr: float = MISSING


@dataclass
class MetricConfig:
    type: str = MISSING
    variable_mode: str = MISSING
    variable_name: Optional[str] = None


@dataclass
class Config:
    output_dir: str = MISSING
    ckpt: Optional[str] = None  # Checkpoint path is optional
    log_epoch: int = 1  # Default logging frequency

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    metric: MetricConfig = field(default_factory=MetricConfig)
