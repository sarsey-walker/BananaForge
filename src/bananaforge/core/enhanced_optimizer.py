"""Enhanced optimization engine with Feature 3 capabilities.

This module implements the enhanced optimization features:
- Discrete validation tracking
- Advanced learning rate scheduling
- Enhanced early stopping
- Mixed precision support
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR

from .optimizer import LayerOptimizer, OptimizationConfig


@dataclass
class EnhancedOptimizationConfig(OptimizationConfig):
    """Enhanced configuration with Feature 3 capabilities."""

    # Discrete validation tracking
    enable_discrete_tracking: bool = True
    validation_interval: int = 10
    discrete_loss_weight: float = 1.0

    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    warmup_steps: int = 0
    warmup_factor: float = 0.1
    decay_schedule: str = "cosine"  # linear, cosine, exponential, step
    lr_decay_factor: float = 0.1
    lr_decay_steps: int = 1000
    enable_adaptive_lr: bool = False
    lr_patience: int = 100
    lr_reduction_factor: float = 0.5

    # Enhanced early stopping
    enable_enhanced_early_stopping: bool = True
    discrete_patience: int = 200
    continuous_patience: int = 200
    min_improvement_threshold: float = 1e-6
    early_stopping_metric: str = "discrete_loss"  # discrete_loss, continuous_loss, both

    # Mixed precision support
    enable_mixed_precision: bool = False
    autocast_dtype: torch.dtype = torch.float16
    grad_scaling: bool = True
    scale_factor: float = 2.0**16


class DiscreteValidator:
    """Handles discrete validation tracking during optimization."""

    def __init__(self, config: EnhancedOptimizationConfig):
        self.config = config
        self.best_discrete_loss = float("inf")
        self.best_discrete_state = None
        self.discrete_history = []
        self.validation_count = 0

    def should_validate(self, step: int) -> bool:
        """Check if we should run discrete validation at this step."""
        return step % self.config.validation_interval == 0

    def compute_discrete_loss(
        self,
        optimizer: nn.Module,
        target_image: torch.Tensor,
        material_colors: torch.Tensor,
    ) -> float:
        """Compute discrete loss using hard material assignments."""
        with torch.no_grad():
            # Get discrete material assignments from global logits
            discrete_global_logits = torch.zeros_like(optimizer.global_logits)
            discrete_assignments = torch.argmax(optimizer.global_logits, dim=1)

            # Create one-hot assignments
            for i, assignment in enumerate(discrete_assignments):
                discrete_global_logits[i, assignment] = 1.0

            # Temporarily replace soft assignments with hard ones
            original_logits = optimizer.global_logits.data.clone()
            optimizer.global_logits.data = discrete_global_logits

            # Forward pass with discrete assignments
            pred_image, height_map = optimizer.forward(material_colors)

            # Compute loss
            discrete_loss = torch.nn.functional.mse_loss(pred_image, target_image)

            # Restore original logits
            optimizer.global_logits.data = original_logits

            return discrete_loss.item()

    def validate(
        self,
        step: int,
        optimizer: nn.Module,
        target_image: torch.Tensor,
        material_colors: torch.Tensor,
    ) -> Tuple[float, bool]:
        """Run discrete validation and return loss and improvement flag."""
        discrete_loss = self.compute_discrete_loss(
            optimizer, target_image, material_colors
        )

        self.discrete_history.append((step, discrete_loss))
        self.validation_count += 1

        # Check for improvement
        improved = False
        if (
            discrete_loss
            < self.best_discrete_loss - self.config.min_improvement_threshold
        ):
            self.best_discrete_loss = discrete_loss
            improved = True

            # Save best discrete state
            self.best_discrete_state = {
                "global_logits": optimizer.global_logits.data.clone(),
                "height_logits": optimizer.height_logits.data.clone(),
                "step": step,
                "discrete_loss": discrete_loss,
            }

        return discrete_loss, improved

    def get_best_state(self) -> Optional[Dict[str, Any]]:
        """Get the best discrete state found so far."""
        return self.best_discrete_state

    def get_history(self) -> List[Tuple[int, float]]:
        """Get the discrete validation history."""
        return self.discrete_history.copy()


class LearningRateScheduler:
    """Advanced learning rate scheduling with warmup and decay."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: EnhancedOptimizationConfig,
        total_steps: int,
    ):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.warmup_steps = min(config.warmup_steps, total_steps // 4)

        # Initialize scheduler based on decay schedule
        self.scheduler = self._create_scheduler()

        # Adaptive LR components
        if config.enable_adaptive_lr:
            self.plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.lr_reduction_factor,
                patience=config.lr_patience,
                threshold=config.min_improvement_threshold,
            )
        else:
            self.plateau_scheduler = None

        self.lr_history = []

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create the appropriate learning rate scheduler."""
        if not self.config.enable_lr_scheduling:
            return None

        post_warmup_steps = max(1, self.total_steps - self.warmup_steps)

        if self.config.decay_schedule == "cosine":
            return CosineAnnealingLR(
                self.optimizer, T_max=post_warmup_steps, eta_min=self.base_lr * 0.01
            )
        elif self.config.decay_schedule == "exponential":
            gamma = (self.config.lr_decay_factor) ** (1.0 / post_warmup_steps)
            return ExponentialLR(self.optimizer, gamma=gamma)
        elif self.config.decay_schedule == "step":
            step_size = max(1, post_warmup_steps // 3)
            return StepLR(
                self.optimizer, step_size=step_size, gamma=self.config.lr_decay_factor
            )
        else:  # linear
            return None  # We'll handle linear decay manually

    def step(self, loss: Optional[float] = None) -> None:
        """Update learning rate for current step."""
        if self.current_step < self.warmup_steps:
            # Warmup phase
            warmup_factor = self.config.warmup_factor
            progress = self.current_step / self.warmup_steps
            lr_multiplier = warmup_factor + (1.0 - warmup_factor) * progress
            current_lr = self.base_lr * lr_multiplier

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        elif self.scheduler is not None:
            # Use built-in scheduler
            self.scheduler.step()

        elif self.config.decay_schedule == "linear":
            # Manual linear decay
            decay_steps = self.total_steps - self.warmup_steps
            progress = (self.current_step - self.warmup_steps) / decay_steps
            lr_multiplier = 1.0 - progress * (1.0 - self.config.lr_decay_factor)
            current_lr = self.base_lr * lr_multiplier

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr

        # Adaptive LR adjustment
        if self.plateau_scheduler is not None and loss is not None:
            self.plateau_scheduler.step(loss)

        self.current_step += 1
        self.lr_history.append(self.get_current_lr())

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def get_history(self) -> List[float]:
        """Get learning rate history."""
        return self.lr_history.copy()


class EnhancedEarlyStopping:
    """Enhanced early stopping with discrete metrics and configurable patience."""

    def __init__(self, config: EnhancedOptimizationConfig):
        self.config = config

        # Tracking state
        self.best_continuous_loss = float("inf")
        self.best_discrete_loss = float("inf")
        self.continuous_patience_counter = 0
        self.discrete_patience_counter = 0

        # Stopping state
        self.should_stop = False
        self.stopping_reason = None
        self.stopping_message = ""
        self.stopped_at_step = None

        # History
        self.loss_history = []

    def check_stopping(
        self, step: int, continuous_loss: float, discrete_loss: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check if optimization should stop."""

        # Track continuous loss
        if (
            continuous_loss
            < self.best_continuous_loss - self.config.min_improvement_threshold
        ):
            self.best_continuous_loss = continuous_loss
            self.continuous_patience_counter = 0
        else:
            self.continuous_patience_counter += 1

        # Track discrete loss if available
        if discrete_loss is not None:
            if (
                discrete_loss
                < self.best_discrete_loss - self.config.min_improvement_threshold
            ):
                self.best_discrete_loss = discrete_loss
                self.discrete_patience_counter = 0
            else:
                self.discrete_patience_counter += 1

        # Record history
        self.loss_history.append(
            {
                "step": step,
                "continuous_loss": continuous_loss,
                "discrete_loss": discrete_loss,
                "continuous_patience": self.continuous_patience_counter,
                "discrete_patience": self.discrete_patience_counter,
            }
        )

        # Check stopping conditions
        should_stop, reason = self._evaluate_stopping_conditions()

        if should_stop:
            self.should_stop = True
            self.stopping_reason = reason
            self.stopped_at_step = step
            self.stopping_message = self._generate_stopping_message(step, reason)

        return should_stop, reason

    def _evaluate_stopping_conditions(self) -> Tuple[bool, Optional[str]]:
        """Evaluate whether to stop based on current state."""
        metric = self.config.early_stopping_metric

        if metric == "discrete_loss":
            if self.discrete_patience_counter >= self.config.discrete_patience:
                return True, "discrete_loss_plateau"

        elif metric == "continuous_loss":
            if self.continuous_patience_counter >= self.config.continuous_patience:
                return True, "continuous_loss_plateau"

        elif metric == "both":
            if (
                self.discrete_patience_counter >= self.config.discrete_patience
                and self.continuous_patience_counter >= self.config.continuous_patience
            ):
                return True, "both_losses_plateau"

        return False, None

    def _generate_stopping_message(self, step: int, reason: str) -> str:
        """Generate a clear stopping message."""
        messages = {
            "discrete_loss_plateau": f"Discrete validation loss plateaued for {self.config.discrete_patience} validations",
            "continuous_loss_plateau": f"Continuous loss plateaued for {self.config.continuous_patience} steps",
            "both_losses_plateau": "Both discrete and continuous losses plateaued",
        }

        base_message = messages.get(reason, f"Unknown stopping reason: {reason}")
        return f"Early stopping at step {step}: {base_message}"

    def get_stopping_info(self) -> Dict[str, Any]:
        """Get comprehensive stopping information."""
        return {
            "stopped": self.should_stop,
            "reason": self.stopping_reason,
            "message": self.stopping_message,
            "step": self.stopped_at_step,
            "best_continuous_loss": self.best_continuous_loss,
            "best_discrete_loss": self.best_discrete_loss,
            "continuous_patience": self.continuous_patience_counter,
            "discrete_patience": self.discrete_patience_counter,
        }


class MixedPrecisionManager:
    """Manages mixed precision training for optimization."""

    def __init__(self, config: EnhancedOptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.enabled = False
        self.scaler = None
        self.autocast_enabled = False

        # Initialize mixed precision if requested and supported
        if config.enable_mixed_precision:
            self._initialize_mixed_precision()

    def _initialize_mixed_precision(self) -> None:
        """Initialize mixed precision components."""
        try:
            # Check device and capability support
            if self.device.type == "cuda":
                # Check if CUDA supports mixed precision
                if torch.cuda.is_available():
                    # Initialize gradient scaler
                    if self.config.grad_scaling:
                        try:
                            # Try the newer API first
                            self.scaler = torch.amp.GradScaler(
                                "cuda",
                                init_scale=self.config.scale_factor,
                                growth_factor=2.0,
                                backoff_factor=0.5,
                                growth_interval=2000,
                            )
                        except AttributeError:
                            # Fall back to older API for compatibility
                            self.scaler = torch.cuda.amp.GradScaler(
                                init_scale=self.config.scale_factor,
                                growth_factor=2.0,
                                backoff_factor=0.5,
                                growth_interval=2000,
                            )

                    self.autocast_enabled = True
                    self.enabled = True

            elif self.device.type == "cpu":
                # CPU mixed precision (limited support)
                warnings.warn(
                    "Mixed precision training on CPU has limited support. "
                    "Falling back to float32.",
                    UserWarning,
                )
                self.enabled = False

        except Exception as e:
            warnings.warn(
                f"Failed to initialize mixed precision: {e}. "
                "Falling back to float32.",
                UserWarning,
            )
            self.enabled = False

    def get_autocast_context(self):
        """Get autocast context manager."""
        if self.autocast_enabled and self.device.type == "cuda":
            return torch.autocast(
                device_type="cuda", dtype=self.config.autocast_dtype, enabled=True
            )
        else:
            # Return a no-op context manager
            return torch.no_grad().__class__()

    def scale_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        parameters: List[torch.nn.Parameter],
    ) -> None:
        """Handle scaled backward pass and optimizer step."""
        if self.scaler is not None:
            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Unscale before gradient clipping
            self.scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)

            # Optimizer step with scaling
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=0.5)
            optimizer.step()

    def is_enabled(self) -> bool:
        """Check if mixed precision is enabled."""
        return self.enabled

    def get_scale(self) -> float:
        """Get current gradient scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0


class EnhancedLayerOptimizer(LayerOptimizer):
    """Enhanced layer optimizer with Feature 3 capabilities."""

    def __init__(
        self,
        image_size: Tuple[int, int],
        num_materials: int,
        config: Union[OptimizationConfig, EnhancedOptimizationConfig],
        target_image: Optional[torch.Tensor] = None,
        initial_height_logits: Optional[torch.Tensor] = None,
        initial_global_logits: Optional[torch.Tensor] = None,
    ):
        # Convert to enhanced config if needed
        if not isinstance(config, EnhancedOptimizationConfig):
            enhanced_config = EnhancedOptimizationConfig(**config.__dict__)
        else:
            enhanced_config = config

        self.enhanced_config = enhanced_config

        # Initialize base optimizer
        super().__init__(
            image_size=image_size,
            num_materials=num_materials,
            config=enhanced_config,
            target_image=target_image,
            initial_height_logits=initial_height_logits,
            initial_global_logits=initial_global_logits,
        )

        # Initialize Feature 3 components
        self.discrete_validator = None
        self.lr_scheduler = None
        self.early_stopping = None
        self.mixed_precision = None

        self._initialize_enhanced_features()

    def _initialize_enhanced_features(self) -> None:
        """Initialize all Feature 3 components."""
        # Discrete validation tracking
        if self.enhanced_config.enable_discrete_tracking:
            self.discrete_validator = DiscreteValidator(self.enhanced_config)

        # Enhanced early stopping
        if self.enhanced_config.enable_enhanced_early_stopping:
            self.early_stopping = EnhancedEarlyStopping(self.enhanced_config)

        # Mixed precision support
        if self.enhanced_config.enable_mixed_precision:
            self.mixed_precision = MixedPrecisionManager(
                self.enhanced_config, self.device
            )

    def optimize(
        self,
        target_image: torch.Tensor,
        material_colors: torch.Tensor,
        callback: Optional[callable] = None,
    ) -> Dict[str, List[float]]:
        """Enhanced optimization with Feature 3 capabilities."""
        target_image = target_image.to(self.device)
        material_colors = material_colors.to(self.device)

        # Initialize learning rate scheduler
        if self.enhanced_config.enable_lr_scheduling:
            self.lr_scheduler = LearningRateScheduler(
                self.optimizer, self.enhanced_config, self.enhanced_config.iterations
            )

        # Initialize enhanced loss history
        loss_history = {
            "total": [],
            "perceptual": [],
            "color": [],
            "smoothness": [],
            "consistency": [],
            "discrete_loss": [],
            "learning_rate": [],
        }

        self.train()

        for step in range(self.enhanced_config.iterations):
            self.current_step = step

            # Update temperature
            self._update_temperature_schedule(step)

            # Forward and backward pass with mixed precision support
            total_loss, loss_dict = self._enhanced_forward_backward(
                target_image, material_colors
            )

            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(total_loss.item())

            # Record losses
            for key, value in loss_dict.items():
                if key in loss_history:
                    loss_history[key].append(
                        value.item() if torch.is_tensor(value) else value
                    )

            if self.lr_scheduler is not None:
                loss_history["learning_rate"].append(self.lr_scheduler.get_current_lr())

            # Discrete validation
            discrete_loss = None
            if (
                self.discrete_validator is not None
                and self.discrete_validator.should_validate(step)
            ):
                discrete_loss, improved = self.discrete_validator.validate(
                    step, self, target_image, material_colors
                )
                loss_history["discrete_loss"].append(discrete_loss)
                loss_dict["discrete_loss"] = discrete_loss

            # Enhanced early stopping
            if self.early_stopping is not None:
                should_stop, reason = self.early_stopping.check_stopping(
                    step, total_loss.item(), discrete_loss
                )
                if should_stop:
                    print(
                        f"Enhanced early stopping: {self.early_stopping.stopping_message}"
                    )
                    break
            else:
                # Fall back to basic early stopping
                if total_loss.item() < self.best_loss:
                    self.best_loss = total_loss.item()
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                if (
                    self.patience_counter
                    >= self.enhanced_config.early_stopping_patience
                ):
                    print(f"Basic early stopping at step {step}")
                    break

            # Progress callback
            if callback and step % 10 == 0:
                callback(step, loss_dict, None, None)  # Simplified for now

        return loss_history

    def _enhanced_forward_backward(
        self, target_image: torch.Tensor, material_colors: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhanced forward/backward pass with mixed precision support."""

        if self.mixed_precision is not None and self.mixed_precision.is_enabled():
            # Mixed precision forward pass
            with self.mixed_precision.get_autocast_context():
                pred_image, height_map = self.forward(material_colors)
                material_probs = torch.softmax(self.global_logits, dim=1)
                total_loss, loss_dict = self.loss_fn(
                    pred_image, target_image, height_map, material_probs
                )

            # Mixed precision backward pass
            self.optimizer.zero_grad()
            self.mixed_precision.scale_and_step(
                total_loss, self.optimizer, list(self.parameters())
            )
        else:
            # Standard precision forward/backward pass
            pred_image, height_map = self.forward(material_colors)
            material_probs = torch.softmax(self.global_logits, dim=1)
            total_loss, loss_dict = self.loss_fn(
                pred_image, target_image, height_map, material_probs
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()

        return total_loss, loss_dict

    def _update_temperature_schedule(self, step: int) -> None:
        """Update temperature scheduling (enhanced version)."""
        if self.lr_scheduler is not None:
            # Align temperature scheduling with learning rate warmup
            warmup_steps = self.lr_scheduler.warmup_steps
            if step < warmup_steps:
                # Keep initial temperature during warmup
                self.current_temperature = self.enhanced_config.initial_temperature
            else:
                # Apply decay after warmup
                progress = (step - warmup_steps) / max(
                    1, self.enhanced_config.iterations - warmup_steps
                )
                self.current_temperature = (
                    self.enhanced_config.initial_temperature
                    + progress
                    * (
                        self.enhanced_config.final_temperature
                        - self.enhanced_config.initial_temperature
                    )
                )
        else:
            # Use original temperature scheduling
            super()._update_temperature_schedule(step)

        self.gumbel_softmax.set_temperature(self.current_temperature)

    def get_enhancement_info(self) -> Dict[str, Any]:
        """Get information about enabled enhancements."""
        info = {
            "discrete_tracking": self.discrete_validator is not None,
            "lr_scheduling": self.lr_scheduler is not None,
            "enhanced_early_stopping": self.early_stopping is not None,
            "mixed_precision": self.mixed_precision is not None
            and self.mixed_precision.is_enabled(),
        }

        # Add component-specific info
        if self.discrete_validator is not None:
            info["best_discrete_loss"] = self.discrete_validator.best_discrete_loss
            info["validation_count"] = self.discrete_validator.validation_count

        if self.lr_scheduler is not None:
            info["current_lr"] = self.lr_scheduler.get_current_lr()
            info["warmup_steps"] = self.lr_scheduler.warmup_steps

        if self.early_stopping is not None:
            info["early_stopping_info"] = self.early_stopping.get_stopping_info()

        if self.mixed_precision is not None:
            info["mixed_precision_scale"] = self.mixed_precision.get_scale()

        return info

    def get_best_discrete_state(self) -> Optional[Dict[str, Any]]:
        """Get the best discrete state if available."""
        if self.discrete_validator is not None:
            return self.discrete_validator.get_best_state()
        return None

    def restore_best_discrete_state(self) -> bool:
        """Restore the best discrete state if available."""
        best_state = self.get_best_discrete_state()
        if best_state is not None:
            self.global_logits.data = best_state["global_logits"]
            self.height_logits.data = best_state["height_logits"]
            return True
        return False
