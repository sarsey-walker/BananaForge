"""Transparency-based optimization for filament savings and material swap reduction.

This module implements optimization algorithms that use transparency mixing to
reduce material swaps by 30% or more while maintaining visual quality.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from .base_layer_optimizer import BaseLayerOptimizer
from .transparency_mixer import TransparencyColorMixer


@dataclass
class TransparencyOptimizationConfig:
    """Configuration for transparency-based optimization."""

    min_savings_threshold: float = 0.3  # 30% minimum savings
    quality_preservation_weight: float = 0.7
    cost_reduction_weight: float = 0.3
    max_iterations: int = 1000
    convergence_threshold: float = 1e-4
    enable_parallel_processing: bool = True
    chunk_size: int = 64


class TransparencyOptimizer:
    """Optimizer for reducing material swaps through transparency mixing.

    This class implements Story 4.5.4: Filament Savings Through Transparency,
    focusing on achieving 30%+ reduction in material swaps while maintaining
    visual quality.
    """

    def __init__(
        self,
        min_savings_threshold: float = 0.3,
        quality_preservation_weight: float = 0.7,
        cost_reduction_weight: float = 0.3,
        device: str = "cuda",
    ):
        """Initialize transparency optimizer.

        Args:
            min_savings_threshold: Minimum savings rate to achieve (0-1)
            quality_preservation_weight: Weight for quality preservation (0-1)
            cost_reduction_weight: Weight for cost reduction (0-1)
            device: Device for computations
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.min_savings_threshold = min_savings_threshold
        self.quality_preservation_weight = quality_preservation_weight
        self.cost_reduction_weight = cost_reduction_weight

        # Validate weights
        total_weight = quality_preservation_weight + cost_reduction_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")

        # Initialize sub-components
        self.transparency_mixer = TransparencyColorMixer(device=str(self.device))
        self.base_optimizer = BaseLayerOptimizer(device=str(self.device))

        # Optimization state
        self._optimization_cache = {}
        self._cache_lock = threading.Lock()

    def calculate_material_swaps(
        self, material_assignments: torch.Tensor, method: str = "standard"
    ) -> int:
        """Calculate number of material swaps in layer assignments.

        Args:
            material_assignments: Material assignments tensor (layers, H, W)
            method: Calculation method ('standard', 'weighted')

        Returns:
            Number of material swaps
        """
        num_layers, height, width = material_assignments.shape

        if num_layers <= 1:
            return 0

        total_swaps = 0

        if method == "standard":
            # Count transitions between adjacent layers
            for layer in range(num_layers - 1):
                current_layer = material_assignments[layer]
                next_layer = material_assignments[layer + 1]

                # Count positions where material changes
                swaps = torch.sum(current_layer != next_layer).item()
                total_swaps += swaps

        elif method == "weighted":
            # Weight swaps by frequency and complexity
            for layer in range(num_layers - 1):
                current_layer = material_assignments[layer]
                next_layer = material_assignments[layer + 1]

                # Calculate swap complexity
                unique_swaps = self._count_unique_swaps(current_layer, next_layer)
                complexity_weight = self._calculate_swap_complexity_weight(
                    current_layer, next_layer
                )

                weighted_swaps = unique_swaps * complexity_weight
                total_swaps += int(weighted_swaps)

        return total_swaps

    def optimize_with_transparency(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        target_savings: float = 0.3,
    ) -> Dict:
        """Optimize material assignments using transparency to reduce swaps.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Original material assignments (layers, H, W)
            materials: List of material dictionaries
            target_savings: Target savings rate (0-1)

        Returns:
            Dictionary with optimization results
        """
        # Calculate baseline swaps
        baseline_swaps = self.calculate_material_swaps(material_assignments, "standard")

        # Optimize base layer selection
        base_colors = [torch.tensor(mat["color"]) for mat in materials]
        base_optimization = self.base_optimizer.optimize_for_palette_expansion(
            height_map, base_colors, base_colors, max_base_colors=2
        )

        # Apply transparency-based optimization
        optimized_assignments = self._apply_transparency_optimization(
            material_assignments, materials, base_optimization, target_savings
        )

        # Calculate optimized swaps
        optimized_swaps = self.calculate_material_swaps(
            optimized_assignments, "standard"
        )

        # Calculate swap reduction
        if baseline_swaps > 0:
            swap_reduction = (baseline_swaps - optimized_swaps) / baseline_swaps
        else:
            swap_reduction = 0.0

        return {
            "optimized_assignments": optimized_assignments,
            "swap_reduction": swap_reduction,
            "baseline_swaps": baseline_swaps,
            "optimized_swaps": optimized_swaps,
            "target_achieved": swap_reduction >= target_savings,
            "optimization_details": {
                "base_optimization": base_optimization,
                "transparency_strategy": self._get_transparency_strategy_info(),
            },
        }

    def calculate_cost_savings(
        self,
        height_map: torch.Tensor,
        baseline_assignments: torch.Tensor,
        materials: List[Dict],
        transparency_enabled: bool = True,
    ) -> Dict:
        """Calculate cost savings from transparency optimization.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            baseline_assignments: Baseline material assignments
            materials: List of material dictionaries
            transparency_enabled: Whether transparency optimization is enabled

        Returns:
            Dictionary with detailed cost analysis
        """
        # Calculate baseline costs
        baseline_cost_breakdown = self._calculate_material_costs(
            baseline_assignments, materials, height_map
        )
        baseline_cost = baseline_cost_breakdown["total_cost"]

        if transparency_enabled:
            # Optimize with transparency
            optimization_result = self.optimize_with_transparency(
                height_map, baseline_assignments, materials
            )
            optimized_assignments = optimization_result["optimized_assignments"]

            # Calculate optimized costs
            optimized_cost_breakdown = self._calculate_material_costs(
                optimized_assignments, materials, height_map
            )
            optimized_cost = optimized_cost_breakdown["total_cost"]
        else:
            optimized_cost_breakdown = baseline_cost_breakdown
            optimized_cost = baseline_cost

        # Calculate savings
        total_savings = baseline_cost - optimized_cost

        # Calculate material usage reduction
        material_usage_reduction = self._calculate_material_usage_reduction(
            baseline_assignments,
            optimized_assignments if transparency_enabled else baseline_assignments,
            materials,
        )

        return {
            "baseline_cost": baseline_cost,
            "optimized_cost": optimized_cost,
            "total_savings": total_savings,
            "savings_percentage": total_savings / max(baseline_cost, 1e-6),
            "cost_breakdown": {
                "baseline": baseline_cost_breakdown,
                "optimized": optimized_cost_breakdown,
            },
            "material_usage_reduction": material_usage_reduction,
        }

    def optimize_for_quality_preservation(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        quality_threshold: float = 0.85,
    ) -> Dict:
        """Optimize while preserving visual quality above threshold.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material assignments tensor
            materials: List of material dictionaries
            quality_threshold: Minimum quality score to maintain

        Returns:
            Dictionary with quality-preserving optimization results
        """
        # Calculate baseline quality metrics
        baseline_quality = self._calculate_visual_quality_metrics(
            material_assignments, materials
        )

        # Iterative optimization with quality constraints
        best_assignments = material_assignments.clone()
        best_quality = baseline_quality
        best_swap_reduction = 0.0

        # Try different transparency strategies
        strategies = [
            {"opacity_focus": "conservative", "base_strategy": "single_dark"},
            {"opacity_focus": "moderate", "base_strategy": "dual_contrast"},
            {"opacity_focus": "aggressive", "base_strategy": "optimal_expansion"},
        ]

        for strategy in strategies:
            # Apply strategy
            strategy_result = self._apply_transparency_strategy(
                height_map, material_assignments, materials, strategy
            )

            # Check quality
            strategy_quality = self._calculate_visual_quality_metrics(
                strategy_result["assignments"], materials
            )

            # Check if quality threshold is met
            if strategy_quality["visual_quality_score"] >= quality_threshold:
                # Calculate swap reduction
                swap_reduction = self._calculate_swap_reduction(
                    material_assignments, strategy_result["assignments"]
                )

                # Update best if better swap reduction
                if swap_reduction > best_swap_reduction:
                    best_assignments = strategy_result["assignments"]
                    best_quality = strategy_quality
                    best_swap_reduction = swap_reduction

        # Calculate color change reduction
        color_change_reduction = self._calculate_color_change_reduction(
            material_assignments, best_assignments
        )

        return {
            "optimized_assignments": best_assignments,
            "quality_metrics": best_quality,
            "color_change_reduction": color_change_reduction,
            "swap_reduction_achieved": best_swap_reduction,
            "quality_threshold_met": best_quality["visual_quality_score"]
            >= quality_threshold,
        }

    def generate_detailed_savings_report(
        self,
        height_map: torch.Tensor,
        baseline_assignments: torch.Tensor,
        materials: List[Dict],
        include_time_analysis: bool = True,
        include_material_analysis: bool = True,
        include_swap_analysis: bool = True,
    ) -> Dict:
        """Generate comprehensive savings report.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            baseline_assignments: Baseline material assignments
            materials: List of material dictionaries
            include_time_analysis: Whether to include time analysis
            include_material_analysis: Whether to include material analysis
            include_swap_analysis: Whether to include swap analysis

        Returns:
            Comprehensive savings report dictionary
        """
        # Optimize with transparency
        optimization_result = self.optimize_with_transparency(
            height_map, baseline_assignments, materials
        )

        # Calculate cost savings
        cost_analysis = self.calculate_cost_savings(
            height_map, baseline_assignments, materials, transparency_enabled=True
        )

        # Build report
        report = {
            "summary": {
                "total_cost_savings": cost_analysis["total_savings"],
                "total_time_savings": 0.0,  # Will be calculated if requested
                "total_material_savings": 0.0,  # Will be calculated if requested
                "swap_reduction_percentage": optimization_result["swap_reduction"],
            }
        }

        # Time savings analysis
        if include_time_analysis:
            time_savings = self._calculate_time_savings(
                baseline_assignments, optimization_result["optimized_assignments"]
            )
            report["time_savings"] = time_savings
            report["summary"]["total_time_savings"] = time_savings["time_saved_minutes"]

        # Material savings analysis
        if include_material_analysis:
            material_savings = self._calculate_detailed_material_savings(
                baseline_assignments,
                optimization_result["optimized_assignments"],
                materials,
            )
            report["material_savings"] = material_savings
            report["summary"]["total_material_savings"] = material_savings[
                "material_saved_grams"
            ]

        # Swap analysis
        if include_swap_analysis:
            swap_savings = self._calculate_detailed_swap_savings(
                baseline_assignments, optimization_result["optimized_assignments"]
            )
            report["swap_savings"] = swap_savings

        # Cost analysis
        report["cost_analysis"] = cost_analysis

        return report

    def optimize_with_constraints(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        min_quality_score: float = 0.9,
        max_color_error: float = 0.1,
        min_savings_rate: float = 0.25,
    ) -> Dict:
        """Optimize with strict quality and savings constraints.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material assignments tensor
            materials: List of material dictionaries
            min_quality_score: Minimum quality score required
            max_color_error: Maximum allowable color error
            min_savings_rate: Minimum savings rate required

        Returns:
            Dictionary with constraint-compliant optimization results
        """
        # Initialize constraint tracking
        constraints_met = {
            "quality_constraint_met": False,
            "color_error_constraint_met": False,
            "savings_constraint_met": False,
        }

        best_result = None
        constraint_violations = []

        # Try multiple optimization approaches
        approaches = [
            {"quality_weight": 0.8, "savings_weight": 0.2, "name": "quality_focused"},
            {"quality_weight": 0.6, "savings_weight": 0.4, "name": "balanced"},
            {"quality_weight": 0.4, "savings_weight": 0.6, "name": "savings_focused"},
        ]

        for approach in approaches:
            # Configure optimizer weights
            temp_optimizer = TransparencyOptimizer(
                quality_preservation_weight=approach["quality_weight"],
                cost_reduction_weight=approach["savings_weight"],
                device=str(self.device),
            )

            # Run optimization
            result = temp_optimizer.optimize_for_quality_preservation(
                height_map, material_assignments, materials, min_quality_score
            )

            # Check constraints
            quality_score = result["quality_metrics"]["visual_quality_score"]
            color_error = result["quality_metrics"].get("color_accuracy", 1.0)
            color_error = 1.0 - color_error  # Convert accuracy to error
            savings_rate = result["swap_reduction_achieved"]

            current_constraints = {
                "quality_constraint_met": quality_score >= min_quality_score,
                "color_error_constraint_met": color_error <= max_color_error,
                "savings_constraint_met": savings_rate >= min_savings_rate,
            }

            # Check if all constraints are met
            all_constraints_met = all(current_constraints.values())

            if all_constraints_met:
                constraints_met = current_constraints
                best_result = result
                best_result["achieved_metrics"] = {
                    "quality_score": quality_score,
                    "color_error": color_error,
                    "savings_rate": savings_rate,
                }
                break
            else:
                # Track constraint violations
                violations = [k for k, v in current_constraints.items() if not v]
                constraint_violations.extend(violations)

                # Keep best partial result
                if best_result is None:
                    best_result = result
                    best_result["achieved_metrics"] = {
                        "quality_score": quality_score,
                        "color_error": color_error,
                        "savings_rate": savings_rate,
                    }

        # Return results with constraint compliance info
        optimization_success = all(constraints_met.values())

        return {
            "optimization_success": optimization_success,
            "constraint_compliance": constraints_met,
            "achieved_metrics": best_result["achieved_metrics"] if best_result else {},
            "optimization_result": best_result,
            "constraint_violations": (
                list(set(constraint_violations)) if constraint_violations else []
            ),
        }

    def optimize_large_model(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        chunk_size: int = 64,
        parallel_processing: bool = True,
    ) -> Dict:
        """Optimize large models efficiently using chunking and parallelization.

        Args:
            height_map: Large height map tensor
            material_assignments: Large material assignments tensor
            materials: List of material dictionaries
            chunk_size: Size of chunks for processing
            parallel_processing: Whether to use parallel processing

        Returns:
            Dictionary with optimization results for large model
        """
        start_time = time.time()

        # Divide into chunks
        chunks = self._divide_into_chunks(height_map, material_assignments, chunk_size)

        if parallel_processing and len(chunks) > 1:
            # Process chunks in parallel
            optimized_chunks = self._process_chunks_parallel(chunks, materials)
        else:
            # Process chunks sequentially
            optimized_chunks = self._process_chunks_sequential(chunks, materials)

        # Reassemble optimized result
        optimized_assignments = self._reassemble_chunks(optimized_chunks)

        # Calculate overall metrics
        baseline_swaps = self.calculate_material_swaps(material_assignments)
        optimized_swaps = self.calculate_material_swaps(optimized_assignments)

        swap_reduction = (
            ((baseline_swaps - optimized_swaps) / max(baseline_swaps, 1))
            if baseline_swaps > 0
            else 0
        )

        # Calculate cost savings
        cost_analysis = self.calculate_cost_savings(
            height_map, material_assignments, materials, transparency_enabled=True
        )

        processing_time = time.time() - start_time

        return {
            "optimized_assignments": optimized_assignments,
            "swap_reduction": swap_reduction,
            "cost_savings": cost_analysis["total_savings"],
            "processing_time_seconds": processing_time,
            "chunks_processed": len(chunks),
            "parallel_processing_used": parallel_processing and len(chunks) > 1,
        }

    def _apply_transparency_optimization(
        self,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        base_optimization: Dict,
        target_savings: float,
    ) -> torch.Tensor:
        """Apply transparency-based optimization to material assignments."""
        optimized_assignments = material_assignments.clone()

        # Extract base colors for optimization
        base_colors = [torch.tensor(mat["color"]) for mat in materials]

        # Get optimal base layers from base optimization
        if "selected_base_colors" in base_optimization:
            optimal_bases = base_optimization["selected_base_colors"]
        else:
            # Fallback: select darkest colors
            brightness_scores = [torch.mean(color).item() for color in base_colors]
            darkest_idx = np.argmin(brightness_scores)
            optimal_bases = [{"base_color_index": darkest_idx}]

        # Apply transparency mixing strategy
        for layer_idx in range(material_assignments.shape[0]):
            current_layer = material_assignments[layer_idx]

            # Identify regions for transparency optimization
            optimization_regions = self._identify_optimization_regions(
                current_layer, layer_idx, material_assignments.shape[0]
            )

            # Apply transparency mixing to identified regions
            for region in optimization_regions:
                optimized_assignments = self._apply_transparency_to_region(
                    optimized_assignments, region, optimal_bases, base_colors, layer_idx
                )

        return optimized_assignments

    def _calculate_material_costs(
        self,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        height_map: torch.Tensor,
    ) -> Dict:
        """Calculate detailed material costs."""
        layer_height = 0.08  # mm, typical layer height
        # Calculate volume per pixel
        pixel_area = 1.0  # Normalized pixel area

        total_cost = 0.0
        material_costs = {}
        swap_costs = 0.0
        time_costs = 0.0

        # Calculate material usage costs
        for mat_idx, material in enumerate(materials):
            cost_per_kg = material.get("cost_per_kg", 25.0)
            density = material.get("density", 1.24)  # g/cm³

            # Count pixels using this material
            material_mask = material_assignments == mat_idx
            material_pixels = torch.sum(material_mask).item()

            # Calculate volume (simplified)
            volume_cm3 = (
                material_pixels * pixel_area * layer_height / 10.0
            )  # Convert to cm³
            weight_g = volume_cm3 * density
            weight_kg = weight_g / 1000.0

            material_cost = weight_kg * cost_per_kg
            total_cost += material_cost

            material_costs[material["id"]] = {
                "weight_g": weight_g,
                "cost": material_cost,
                "pixels_used": material_pixels,
            }

        # Calculate swap costs (time and material waste)
        num_swaps = self.calculate_material_swaps(material_assignments)
        swap_time_minutes = num_swaps * 2.0  # 2 minutes per swap
        swap_cost_per_minute = 0.5  # $0.50 per minute of time
        swap_costs = swap_time_minutes * swap_cost_per_minute

        # Calculate time costs
        estimated_print_time_hours = (
            material_assignments.shape[0] * layer_height / 10.0
        )  # Simplified
        time_costs = estimated_print_time_hours * 1.0  # $1 per hour electricity/wear

        total_cost += swap_costs + time_costs

        return {
            "total_cost": total_cost,
            "material_costs": material_costs,
            "swap_costs": swap_costs,
            "time_costs": time_costs,
            "num_swaps": num_swaps,
        }

    def _calculate_material_usage_reduction(
        self,
        baseline_assignments: torch.Tensor,
        optimized_assignments: torch.Tensor,
        materials: List[Dict],
    ) -> Dict:
        """Calculate reduction in material usage."""
        usage_reduction = {}

        for mat_idx, material in enumerate(materials):
            # Calculate baseline usage
            baseline_pixels = torch.sum(baseline_assignments == mat_idx).item()

            # Calculate optimized usage
            optimized_pixels = torch.sum(optimized_assignments == mat_idx).item()

            # Calculate reduction
            reduction = baseline_pixels - optimized_pixels
            reduction_percentage = reduction / max(baseline_pixels, 1)

            usage_reduction[material["id"]] = {
                "baseline_usage": baseline_pixels,
                "optimized_usage": optimized_pixels,
                "reduction_pixels": reduction,
                "reduction_percentage": reduction_percentage,
            }

        return usage_reduction

    def _calculate_visual_quality_metrics(
        self, material_assignments: torch.Tensor, materials: List[Dict]
    ) -> Dict:
        """Calculate visual quality metrics for material assignments."""
        # Reconstruct color image from assignments
        base_colors = torch.stack([torch.tensor(mat["color"]) for mat in materials])

        # Create color reconstruction
        num_layers, height, width = material_assignments.shape
        reconstructed = torch.zeros(3, height, width)

        # Simple reconstruction by averaging layers
        for layer in range(num_layers):
            layer_assignments = material_assignments[layer]
            for mat_idx in range(len(materials)):
                mask = (layer_assignments == mat_idx).float()
                color = base_colors[mat_idx].view(3, 1, 1)
                reconstructed += mask.unsqueeze(0) * color

        reconstructed /= max(num_layers, 1)

        # Calculate quality metrics
        color_variance = torch.var(reconstructed, dim=(1, 2)).mean().item()
        layer_consistency = self._calculate_layer_consistency(material_assignments)
        spatial_coherence = self._calculate_spatial_coherence(material_assignments)

        # Composite quality score
        visual_quality_score = (
            0.4 * min(color_variance / 0.1, 1.0)  # Normalize color variance
            + 0.4 * layer_consistency
            + 0.2 * spatial_coherence
        )

        return {
            "visual_quality_score": visual_quality_score,
            "color_accuracy": min(color_variance / 0.1, 1.0),
            "layer_consistency": layer_consistency,
            "spatial_coherence": spatial_coherence,
        }

    def _apply_transparency_strategy(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        materials: List[Dict],
        strategy: Dict,
    ) -> Dict:
        """Apply specific transparency strategy."""
        # Clone assignments for modification
        optimized_assignments = material_assignments.clone()

        # Apply strategy based on configuration
        opacity_focus = strategy["opacity_focus"]
        base_strategy = strategy["base_strategy"]

        # Select opacity levels based on focus
        if opacity_focus == "conservative":
            opacity_levels = [0.25, 0.5, 0.75]
        elif opacity_focus == "moderate":
            opacity_levels = [0.33, 0.67, 1.0]
        else:  # aggressive
            opacity_levels = [0.2, 0.4, 0.6, 0.8, 1.0]

        # Apply base color strategy
        base_colors = [torch.tensor(mat["color"]) for mat in materials]

        if base_strategy == "single_dark":
            # Use single darkest color as base
            brightness_scores = [torch.mean(color).item() for color in base_colors]
            darkest_idx = np.argmin(brightness_scores)
            base_indices = [darkest_idx]
        elif base_strategy == "dual_contrast":
            # Use two contrasting colors
            base_indices = self._select_contrasting_bases(base_colors, 2)
        else:  # optimal_expansion
            # Use base optimizer for selection
            base_opt_result = self.base_optimizer.optimize_for_palette_expansion(
                height_map, base_colors, base_colors, max_base_colors=2
            )
            base_indices = [
                info["base_color_index"]
                for info in base_opt_result["selected_base_colors"]
            ]

        # Apply transparency mixing
        for layer_idx in range(material_assignments.shape[0]):
            optimized_assignments = self._apply_layer_transparency(
                optimized_assignments, layer_idx, base_indices, opacity_levels
            )

        return {
            "assignments": optimized_assignments,
            "strategy_used": strategy,
            "base_indices_used": base_indices,
            "opacity_levels_used": opacity_levels,
        }

    def _calculate_swap_reduction(
        self, baseline_assignments: torch.Tensor, optimized_assignments: torch.Tensor
    ) -> float:
        """Calculate swap reduction between baseline and optimized assignments."""
        baseline_swaps = self.calculate_material_swaps(baseline_assignments)
        optimized_swaps = self.calculate_material_swaps(optimized_assignments)

        if baseline_swaps == 0:
            return 0.0

        return (baseline_swaps - optimized_swaps) / baseline_swaps

    def _calculate_color_change_reduction(
        self, baseline_assignments: torch.Tensor, optimized_assignments: torch.Tensor
    ) -> float:
        """Calculate reduction in color changes."""
        # Count unique color transitions in baseline
        baseline_transitions = self._count_color_transitions(baseline_assignments)

        # Count unique color transitions in optimized
        optimized_transitions = self._count_color_transitions(optimized_assignments)

        if baseline_transitions == 0:
            return 0.0

        return (baseline_transitions - optimized_transitions) / baseline_transitions

    def _calculate_time_savings(
        self, baseline_assignments: torch.Tensor, optimized_assignments: torch.Tensor
    ) -> Dict:
        """Calculate time savings from reduced material swaps."""
        baseline_swaps = self.calculate_material_swaps(baseline_assignments)
        optimized_swaps = self.calculate_material_swaps(optimized_assignments)

        # Time per swap (including purge and stabilization)
        time_per_swap_minutes = 2.5

        baseline_print_time = (
            baseline_assignments.shape[0] * 0.08 / 10.0 * 60
        )  # Layer time estimate in minutes
        optimized_print_time = optimized_assignments.shape[0] * 0.08 / 10.0 * 60

        swap_time_saved = (baseline_swaps - optimized_swaps) * time_per_swap_minutes

        return {
            "baseline_print_time": baseline_print_time,
            "optimized_print_time": optimized_print_time
            + optimized_swaps * time_per_swap_minutes,
            "time_saved_minutes": swap_time_saved,
            "swap_time_reduction": swap_time_saved,
        }

    def _calculate_detailed_material_savings(
        self,
        baseline_assignments: torch.Tensor,
        optimized_assignments: torch.Tensor,
        materials: List[Dict],
    ) -> Dict:
        """Calculate detailed material savings."""
        baseline_usage = self._calculate_total_material_usage(
            baseline_assignments, materials
        )
        optimized_usage = self._calculate_total_material_usage(
            optimized_assignments, materials
        )

        material_saved = baseline_usage - optimized_usage
        waste_reduction = self._calculate_waste_reduction(
            baseline_assignments, optimized_assignments
        )

        return {
            "baseline_material_usage": baseline_usage,
            "optimized_material_usage": optimized_usage,
            "material_saved_grams": material_saved,
            "waste_reduction": waste_reduction,
        }

    def _calculate_detailed_swap_savings(
        self, baseline_assignments: torch.Tensor, optimized_assignments: torch.Tensor
    ) -> Dict:
        """Calculate detailed swap savings."""
        baseline_swaps = self.calculate_material_swaps(baseline_assignments)
        optimized_swaps = self.calculate_material_swaps(optimized_assignments)

        swaps_eliminated = baseline_swaps - optimized_swaps

        # Calculate complexity reduction
        baseline_complexity = self._calculate_swap_complexity(baseline_assignments)
        optimized_complexity = self._calculate_swap_complexity(optimized_assignments)
        complexity_reduction = baseline_complexity - optimized_complexity

        return {
            "baseline_swap_count": baseline_swaps,
            "optimized_swap_count": optimized_swaps,
            "swaps_eliminated": swaps_eliminated,
            "swap_complexity_reduction": complexity_reduction,
        }

    # Helper methods for implementation
    def _count_unique_swaps(
        self, current_layer: torch.Tensor, next_layer: torch.Tensor
    ) -> int:
        """Count unique material swaps between layers."""
        unique_changes = torch.unique(
            torch.stack([current_layer, next_layer], dim=0), dim=0
        )
        return len(unique_changes) - 1  # Subtract one for no-change case

    def _calculate_swap_complexity_weight(
        self, current_layer: torch.Tensor, next_layer: torch.Tensor
    ) -> float:
        """Calculate complexity weight for material swaps."""
        # More complex patterns get higher weight
        num_unique_current = len(torch.unique(current_layer))
        num_unique_next = len(torch.unique(next_layer))
        complexity = (num_unique_current + num_unique_next) / 10.0  # Normalize
        return min(complexity, 2.0)

    def _get_transparency_strategy_info(self) -> Dict:
        """Get information about transparency strategy used."""
        return {
            "opacity_levels": self.transparency_mixer.opacity_levels,
            "blending_method": self.transparency_mixer.blending_method,
            "max_layers": self.transparency_mixer.max_layers,
        }

    def _identify_optimization_regions(
        self, current_layer: torch.Tensor, layer_idx: int, total_layers: int
    ) -> List[Dict]:
        """Identify regions suitable for transparency optimization."""
        # Simple region identification - areas with frequent material changes
        height, width = current_layer.shape
        regions = []

        # Divide into quadrants for analysis
        mid_h, mid_w = height // 2, width // 2
        quadrants = [
            (0, mid_h, 0, mid_w),
            (0, mid_h, mid_w, width),
            (mid_h, height, 0, mid_w),
            (mid_h, height, mid_w, width),
        ]

        for i, (start_h, end_h, start_w, end_w) in enumerate(quadrants):
            region_data = current_layer[start_h:end_h, start_w:end_w]
            unique_materials = len(torch.unique(region_data))

            if unique_materials > 1:  # Region has material diversity
                regions.append(
                    {
                        "bounds": (start_h, end_h, start_w, end_w),
                        "material_diversity": unique_materials,
                        "priority": unique_materials / 5.0,  # Normalize priority
                    }
                )

        return regions

    def _apply_transparency_to_region(
        self,
        assignments: torch.Tensor,
        region: Dict,
        optimal_bases: List[Dict],
        base_colors: List[torch.Tensor],
        layer_idx: int,
    ) -> torch.Tensor:
        """Apply transparency optimization to a specific region."""
        start_h, end_h, start_w, end_w = region["bounds"]

        if not optimal_bases:
            return assignments

        # Select optimal base for this region
        base_idx = optimal_bases[0]["base_color_index"]

        # Apply transparency-based material assignment
        region_assignments = assignments[:, start_h:end_h, start_w:end_w]

        # Simple transparency strategy: use base color for lower layers
        if layer_idx < assignments.shape[0] // 2:
            # Lower layers: prefer base color
            region_assignments[layer_idx] = base_idx

        assignments[:, start_h:end_h, start_w:end_w] = region_assignments

        return assignments

    def _divide_into_chunks(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        chunk_size: int,
    ) -> List[Dict]:
        """Divide large tensors into processable chunks."""
        _, _, height, width = height_map.shape
        chunks = []

        for i in range(0, height, chunk_size):
            for j in range(0, width, chunk_size):
                end_i = min(i + chunk_size, height)
                end_j = min(j + chunk_size, width)

                chunk_height_map = height_map[:, :, i:end_i, j:end_j]
                chunk_assignments = material_assignments[:, i:end_i, j:end_j]

                chunks.append(
                    {
                        "height_map": chunk_height_map,
                        "assignments": chunk_assignments,
                        "bounds": (i, j, end_i, end_j),
                        "chunk_id": len(chunks),
                    }
                )

        return chunks

    def _process_chunks_parallel(
        self, chunks: List[Dict], materials: List[Dict]
    ) -> List[Dict]:
        """Process chunks in parallel."""
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for chunk in chunks:
                future = executor.submit(self._process_single_chunk, chunk, materials)
                futures.append(future)

            results = [future.result() for future in futures]

        return results

    def _process_chunks_sequential(
        self, chunks: List[Dict], materials: List[Dict]
    ) -> List[Dict]:
        """Process chunks sequentially."""
        return [self._process_single_chunk(chunk, materials) for chunk in chunks]

    def _process_single_chunk(self, chunk: Dict, materials: List[Dict]) -> Dict:
        """Process a single chunk for optimization."""
        chunk_result = self.optimize_with_transparency(
            chunk["height_map"], chunk["assignments"], materials, target_savings=0.3
        )

        return {
            "optimized_assignments": chunk_result["optimized_assignments"],
            "bounds": chunk["bounds"],
            "chunk_id": chunk["chunk_id"],
            "swap_reduction": chunk_result["swap_reduction"],
        }

    def _reassemble_chunks(self, optimized_chunks: List[Dict]) -> torch.Tensor:
        """Reassemble optimized chunks into complete tensor."""
        # Determine output size from chunks
        max_i = max(chunk["bounds"][2] for chunk in optimized_chunks)
        max_j = max(chunk["bounds"][3] for chunk in optimized_chunks)
        num_layers = optimized_chunks[0]["optimized_assignments"].shape[0]

        # Create output tensor
        reassembled = torch.zeros(num_layers, max_i, max_j, dtype=torch.long)

        # Place each chunk in correct position
        for chunk in optimized_chunks:
            i, j, end_i, end_j = chunk["bounds"]
            reassembled[:, i:end_i, j:end_j] = chunk["optimized_assignments"]

        return reassembled

    def _select_contrasting_bases(
        self, base_colors: List[torch.Tensor], num_bases: int
    ) -> List[int]:
        """Select base colors with maximum contrast."""
        if len(base_colors) <= num_bases:
            return list(range(len(base_colors)))

        selected_indices = []

        # Start with darkest color
        brightness_scores = [torch.mean(color).item() for color in base_colors]
        darkest_idx = np.argmin(brightness_scores)
        selected_indices.append(darkest_idx)

        # Add most contrasting colors
        for _ in range(num_bases - 1):
            best_contrast = -1
            best_idx = -1

            for i, color in enumerate(base_colors):
                if i in selected_indices:
                    continue

                # Calculate contrast with already selected colors
                min_contrast = float("inf")
                for selected_idx in selected_indices:
                    selected_color = base_colors[selected_idx]
                    contrast = torch.norm(color - selected_color).item()
                    min_contrast = min(min_contrast, contrast)

                if min_contrast > best_contrast:
                    best_contrast = min_contrast
                    best_idx = i

            if best_idx != -1:
                selected_indices.append(best_idx)

        return selected_indices

    def _apply_layer_transparency(
        self,
        assignments: torch.Tensor,
        layer_idx: int,
        base_indices: List[int],
        opacity_levels: List[float],
    ) -> torch.Tensor:
        """Apply transparency to specific layer."""
        if not base_indices:
            return assignments

        # Simple transparency application
        current_layer = assignments[layer_idx]

        # Use transparency to reduce material diversity
        unique_materials = torch.unique(current_layer)

        if len(unique_materials) > 2:  # If too many materials, apply transparency
            # Replace some materials with base materials using transparency
            base_idx = base_indices[0]

            # Replace materials with lowest frequency
            material_counts = [
                (mat.item(), torch.sum(current_layer == mat).item())
                for mat in unique_materials
            ]
            material_counts.sort(key=lambda x: x[1])  # Sort by frequency

            # Replace least frequent materials
            for mat_val, count in material_counts[: len(material_counts) // 2]:
                assignments[layer_idx][current_layer == mat_val] = base_idx

        return assignments

    def _calculate_layer_consistency(self, material_assignments: torch.Tensor) -> float:
        """Calculate consistency between layers."""
        if material_assignments.shape[0] <= 1:
            return 1.0

        consistency_scores = []

        for layer in range(material_assignments.shape[0] - 1):
            current = material_assignments[layer]
            next_layer = material_assignments[layer + 1]

            # Calculate similarity between adjacent layers
            same_assignments = torch.sum(current == next_layer).item()
            total_positions = current.numel()
            similarity = same_assignments / total_positions

            consistency_scores.append(similarity)

        return np.mean(consistency_scores)

    def _calculate_spatial_coherence(self, material_assignments: torch.Tensor) -> float:
        """Calculate spatial coherence of material assignments."""
        coherence_scores = []

        for layer in range(material_assignments.shape[0]):
            layer_data = material_assignments[layer]

            # Calculate spatial coherence using neighbor similarity
            height, width = layer_data.shape
            same_neighbors = 0
            total_neighbors = 0

            for i in range(height):
                for j in range(width):
                    current_mat = layer_data[i, j]

                    # Check 4-connected neighbors
                    for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if layer_data[ni, nj] == current_mat:
                                same_neighbors += 1
                            total_neighbors += 1

            if total_neighbors > 0:
                coherence_scores.append(same_neighbors / total_neighbors)

        return np.mean(coherence_scores) if coherence_scores else 0.0

    def _count_color_transitions(self, material_assignments: torch.Tensor) -> int:
        """Count color transitions in material assignments."""
        transitions = set()

        for layer in range(material_assignments.shape[0] - 1):
            current = material_assignments[layer]
            next_layer = material_assignments[layer + 1]

            unique_pairs = torch.unique(
                torch.stack([current.flatten(), next_layer.flatten()]), dim=1
            )

            for pair in unique_pairs.t():
                if pair[0] != pair[1]:  # Different materials
                    transitions.add((pair[0].item(), pair[1].item()))

        return len(transitions)

    def _calculate_total_material_usage(
        self, material_assignments: torch.Tensor, materials: List[Dict]
    ) -> float:
        """Calculate total material usage in grams."""
        total_usage = 0.0
        layer_height = 0.08  # mm
        pixel_area = 1.0  # mm²

        for mat_idx, material in enumerate(materials):
            density = material.get("density", 1.24)  # g/cm³

            # Count pixels using this material
            material_pixels = torch.sum(material_assignments == mat_idx).item()

            # Calculate volume and weight
            volume_mm3 = material_pixels * pixel_area * layer_height
            volume_cm3 = volume_mm3 / 1000.0
            weight_g = volume_cm3 * density

            total_usage += weight_g

        return total_usage

    def _calculate_waste_reduction(
        self, baseline_assignments: torch.Tensor, optimized_assignments: torch.Tensor
    ) -> float:
        """Calculate waste reduction from optimization."""
        # Estimate waste from purging during material changes
        baseline_swaps = self.calculate_material_swaps(baseline_assignments)
        optimized_swaps = self.calculate_material_swaps(optimized_assignments)

        # Assume 1g waste per swap (purging)
        waste_per_swap = 1.0  # grams

        waste_reduction = (baseline_swaps - optimized_swaps) * waste_per_swap
        return waste_reduction

    def _calculate_swap_complexity(self, material_assignments: torch.Tensor) -> float:
        """Calculate complexity of material swaps."""
        complexity = 0.0

        for layer in range(material_assignments.shape[0] - 1):
            current = material_assignments[layer]
            next_layer = material_assignments[layer + 1]

            # Count unique materials in each layer
            current_unique = len(torch.unique(current))
            next_unique = len(torch.unique(next_layer))

            # More unique materials = higher complexity
            layer_complexity = (current_unique + next_unique) / 10.0
            complexity += layer_complexity

        return complexity
