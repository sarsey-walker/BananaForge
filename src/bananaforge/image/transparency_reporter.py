"""
Transparency statistics reporting system.

This module provides detailed transparency statistics and reporting capabilities,
helping users understand their image's transparency characteristics and make
informed decisions about image preparation.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from PIL import Image
from .transparency_detector import TransparencyInfo


@dataclass
class TransparencyStatistics:
    """Detailed transparency statistics for an image."""
    
    # Basic statistics
    total_pixels: int
    transparent_pixel_count: int
    opaque_pixel_count: int
    transparency_percentage: float
    
    # Alpha distribution analysis
    alpha_histogram: Dict[int, int]  # alpha_value: pixel_count
    alpha_distribution_summary: Dict[str, float]  # mean, std, min, max
    
    # Regional analysis
    transparency_regions: List[Dict[str, any]]  # Connected transparency regions
    largest_transparent_region_size: int
    transparency_distribution: str  # "scattered", "clustered", "edge-focused", "uniform"
    
    # Recommendations
    complexity_score: float  # 0-1, higher = more complex transparency
    processing_recommendation: str  # "proceed", "review", "edit_required"
    specific_issues: List[str]  # Specific transparency issues found


class TransparencyReporter:
    """
    Advanced transparency statistics and reporting system.
    
    Provides detailed analysis of transparency patterns, distributions,
    and actionable recommendations for users.
    """
    
    def __init__(self):
        """Initialize the transparency reporter."""
        pass
    
    def generate_detailed_report(self, transparency_info: TransparencyInfo, 
                               image_path: str) -> TransparencyStatistics:
        """
        Generate comprehensive transparency statistics report.
        
        Args:
            transparency_info: Basic transparency detection results
            image_path: Path to the image file for detailed analysis
            
        Returns:
            TransparencyStatistics with detailed analysis
        """
        if not transparency_info.has_transparency:
            return self._create_no_transparency_stats(transparency_info)
        
        # Load image for detailed analysis
        with Image.open(image_path) as image:
            return self._analyze_transparency_patterns(image, transparency_info)
    
    def _create_no_transparency_stats(self, info: TransparencyInfo) -> TransparencyStatistics:
        """Create statistics for non-transparent images."""
        return TransparencyStatistics(
            total_pixels=info.total_pixels,
            transparent_pixel_count=0,
            opaque_pixel_count=info.total_pixels,
            transparency_percentage=0.0,
            alpha_histogram={255: info.total_pixels},
            alpha_distribution_summary={
                "mean": 255.0,
                "std": 0.0,
                "min": 255.0,
                "max": 255.0
            },
            transparency_regions=[],
            largest_transparent_region_size=0,
            transparency_distribution="none",
            complexity_score=0.0,
            processing_recommendation="proceed",
            specific_issues=[]
        )
    
    def _analyze_transparency_patterns(self, image: Image.Image, 
                                     info: TransparencyInfo) -> TransparencyStatistics:
        """Analyze detailed transparency patterns in the image."""
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        # Extract alpha channel
        if image.mode == 'RGBA':
            alpha_channel = img_array[:, :, 3]
        elif image.mode == 'LA':
            alpha_channel = img_array[:, :, 1]
        else:
            # No alpha channel - shouldn't happen for transparent images
            alpha_channel = np.full(img_array.shape[:2], 255, dtype=np.uint8)
        
        # Calculate alpha histogram
        alpha_histogram = self._calculate_alpha_histogram(alpha_channel)
        
        # Calculate alpha distribution statistics
        alpha_stats = self._calculate_alpha_statistics(alpha_channel)
        
        # Analyze transparency regions
        transparency_regions = self._analyze_transparency_regions(alpha_channel)
        
        # Determine transparency distribution pattern
        distribution_pattern = self._classify_transparency_distribution(
            alpha_channel, transparency_regions
        )
        
        # Calculate complexity score
        complexity_score = self._calculate_complexity_score(
            alpha_channel, alpha_histogram, transparency_regions
        )
        
        # Generate recommendations
        recommendation, issues = self._generate_recommendations(
            info, alpha_stats, complexity_score, distribution_pattern
        )
        
        return TransparencyStatistics(
            total_pixels=info.total_pixels,
            transparent_pixel_count=info.transparent_pixel_count,
            opaque_pixel_count=info.opaque_pixel_count,
            transparency_percentage=info.transparency_percentage,
            alpha_histogram=alpha_histogram,
            alpha_distribution_summary=alpha_stats,
            transparency_regions=transparency_regions,
            largest_transparent_region_size=max(
                (region['size'] for region in transparency_regions), 
                default=0
            ),
            transparency_distribution=distribution_pattern,
            complexity_score=complexity_score,
            processing_recommendation=recommendation,
            specific_issues=issues
        )
    
    def _calculate_alpha_histogram(self, alpha_channel: np.ndarray) -> Dict[int, int]:
        """Calculate histogram of alpha values."""
        unique_values, counts = np.unique(alpha_channel, return_counts=True)
        return {int(val): int(count) for val, count in zip(unique_values, counts)}
    
    def _calculate_alpha_statistics(self, alpha_channel: np.ndarray) -> Dict[str, float]:
        """Calculate statistical summary of alpha values."""
        return {
            "mean": float(np.mean(alpha_channel)),
            "std": float(np.std(alpha_channel)),
            "min": float(np.min(alpha_channel)),
            "max": float(np.max(alpha_channel))
        }
    
    def _analyze_transparency_regions(self, alpha_channel: np.ndarray) -> List[Dict[str, any]]:
        """Analyze connected regions of transparency."""
        # Find pixels with alpha < 255 (transparent)
        transparent_mask = alpha_channel < 255
        
        if not np.any(transparent_mask):
            return []
        
        # Use connected components analysis (simplified)
        # For now, we'll analyze basic region characteristics
        regions = []
        
        # Find contiguous transparent areas (simplified approach)
        height, width = alpha_channel.shape
        visited = np.zeros_like(transparent_mask, dtype=bool)
        
        def flood_fill(start_y, start_x):
            """Iterative flood fill to find connected region size."""
            if (start_y < 0 or start_y >= height or 
                start_x < 0 or start_x >= width or
                visited[start_y, start_x] or 
                not transparent_mask[start_y, start_x]):
                return 0
            
            # Use iterative approach to avoid recursion depth issues
            stack = [(start_y, start_x)]
            size = 0
            
            while stack:
                y, x = stack.pop()
                
                if (y < 0 or y >= height or x < 0 or x >= width or
                    visited[y, x] or not transparent_mask[y, x]):
                    continue
                
                visited[y, x] = True
                size += 1
                
                # Add 4-connected neighbors to stack
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    stack.append((y + dy, x + dx))
            
            return size
        
        # Find all connected regions
        for y in range(height):
            for x in range(width):
                if transparent_mask[y, x] and not visited[y, x]:
                    region_size = flood_fill(y, x)
                    if region_size > 10:  # Only report significant regions
                        regions.append({
                            "center_y": y,
                            "center_x": x,
                            "size": region_size,
                            "alpha_values": alpha_channel[transparent_mask].tolist()[:10]  # Sample
                        })
        
        # Sort by size (largest first)
        regions.sort(key=lambda r: r['size'], reverse=True)
        return regions[:10]  # Return top 10 regions
    
    def _classify_transparency_distribution(self, alpha_channel: np.ndarray, 
                                          regions: List[Dict]) -> str:
        """Classify the pattern of transparency distribution."""
        if len(regions) == 0:
            return "none"
        
        height, width = alpha_channel.shape
        total_transparent = np.sum(alpha_channel < 255)
        
        if total_transparent == 0:
            return "none"
        
        # Check for edge-focused transparency
        edge_pixels = self._count_edge_transparency(alpha_channel)
        edge_ratio = edge_pixels / total_transparent if total_transparent > 0 else 0
        
        if edge_ratio > 0.7:
            return "edge-focused"
        
        # Check for clustering vs scattering
        if len(regions) == 1 and regions[0]['size'] > total_transparent * 0.8:
            return "clustered"
        elif len(regions) > 5:
            return "scattered"
        elif self._check_uniform_distribution(alpha_channel):
            return "uniform"
        else:
            return "mixed"
    
    def _count_edge_transparency(self, alpha_channel: np.ndarray) -> int:
        """Count transparent pixels near image edges."""
        height, width = alpha_channel.shape
        edge_width = min(10, min(height, width) // 10)  # 10px or 10% of smallest dimension
        
        # Top and bottom edges
        top_edge = alpha_channel[:edge_width, :]
        bottom_edge = alpha_channel[-edge_width:, :]
        
        # Left and right edges
        left_edge = alpha_channel[:, :edge_width]
        right_edge = alpha_channel[:, -edge_width:]
        
        # Count transparent pixels in edge regions
        edge_transparent = (
            np.sum(top_edge < 255) +
            np.sum(bottom_edge < 255) +
            np.sum(left_edge < 255) +
            np.sum(right_edge < 255)
        )
        
        return edge_transparent
    
    def _check_uniform_distribution(self, alpha_channel: np.ndarray) -> bool:
        """Check if transparency is uniformly distributed."""
        # Divide image into grid and check transparency in each cell
        height, width = alpha_channel.shape
        grid_size = 4  # 4x4 grid
        
        cell_height = height // grid_size
        cell_width = width // grid_size
        
        transparencies = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y_start = i * cell_height
                y_end = min((i + 1) * cell_height, height)
                x_start = j * cell_width
                x_end = min((j + 1) * cell_width, width)
                
                cell = alpha_channel[y_start:y_end, x_start:x_end]
                cell_transparent = np.sum(cell < 255)
                cell_total = cell.size
                
                if cell_total > 0:
                    transparencies.append(cell_transparent / cell_total)
        
        # Check if transparency is relatively uniform across cells
        if len(transparencies) > 0:
            std_dev = np.std(transparencies)
            return std_dev < 0.2  # Low standard deviation indicates uniformity
        
        return False
    
    def _calculate_complexity_score(self, alpha_channel: np.ndarray, 
                                  alpha_histogram: Dict[int, int],
                                  regions: List[Dict]) -> float:
        """Calculate complexity score (0-1) for transparency patterns."""
        complexity_factors = []
        
        # Factor 1: Number of different alpha values (0-1)
        unique_alphas = len(alpha_histogram)
        alpha_complexity = min(unique_alphas / 50.0, 1.0)  # Normalize to 50 max
        complexity_factors.append(alpha_complexity)
        
        # Factor 2: Number of regions (0-1)  
        region_complexity = min(len(regions) / 20.0, 1.0)  # Normalize to 20 max
        complexity_factors.append(region_complexity)
        
        # Factor 3: Alpha value variation (0-1)
        if unique_alphas > 1:
            alpha_values = np.array(list(alpha_histogram.keys()))
            alpha_std = np.std(alpha_values)
            variation_complexity = min(alpha_std / 127.5, 1.0)  # Normalize to max std
            complexity_factors.append(variation_complexity)
        else:
            complexity_factors.append(0.0)
        
        # Factor 4: Edge complexity (0-1)
        edge_transparent = self._count_edge_transparency(alpha_channel)
        total_transparent = np.sum(alpha_channel < 255)
        if total_transparent > 0:
            edge_ratio = edge_transparent / total_transparent
            edge_complexity = 1.0 - edge_ratio  # Higher complexity if not edge-focused
            complexity_factors.append(edge_complexity)
        else:
            complexity_factors.append(0.0)
        
        # Return weighted average
        weights = [0.3, 0.3, 0.2, 0.2]  # Weight the factors
        return sum(f * w for f, w in zip(complexity_factors, weights))
    
    def _generate_recommendations(self, info: TransparencyInfo, 
                                alpha_stats: Dict[str, float],
                                complexity_score: float,
                                distribution_pattern: str) -> Tuple[str, List[str]]:
        """Generate processing recommendations and identify specific issues."""
        issues = []
        
        # Analyze transparency percentage
        if info.transparency_percentage > 75:
            issues.append("High transparency percentage may indicate background removal needed")
        elif info.transparency_percentage > 50:
            issues.append("Significant transparency may affect print quality")
        
        # Analyze alpha distribution
        if alpha_stats["std"] > 50:
            issues.append("Wide variation in transparency levels detected")
        
        if alpha_stats["mean"] < 128:
            issues.append("Overall low opacity may indicate transparency artifacts")
        
        # Analyze distribution pattern
        if distribution_pattern == "edge-focused":
            issues.append("Edge transparency detected - likely anti-aliasing or feathering")
        elif distribution_pattern == "scattered":
            issues.append("Scattered transparency may indicate compression artifacts")
        elif distribution_pattern == "uniform":
            issues.append("Uniform transparency suggests intentional alpha channel")
        
        # Generate overall recommendation
        if complexity_score > 0.7:
            recommendation = "edit_required"
            issues.append("Complex transparency pattern requires manual review")
        elif complexity_score > 0.4 or info.transparency_percentage > 25:
            recommendation = "review"
        else:
            recommendation = "proceed"
            
        return recommendation, issues
    
    def format_statistics_report(self, stats: TransparencyStatistics, 
                               verbose: bool = False) -> str:
        """
        Format statistics as human-readable report.
        
        Args:
            stats: Transparency statistics to format
            verbose: Whether to include detailed technical information
            
        Returns:
            Formatted statistics report
        """
        lines = []
        
        # Header
        lines.append("📊 Transparency Analysis Report")
        lines.append("=" * 35)
        lines.append("")
        
        # Basic statistics
        lines.append("🔢 Basic Statistics:")
        lines.append(f"  • Total pixels: {stats.total_pixels:,}")
        lines.append(f"  • Transparent pixels: {stats.transparent_pixel_count:,}")
        lines.append(f"  • Opaque pixels: {stats.opaque_pixel_count:,}")
        lines.append(f"  • Transparency: {stats.transparency_percentage:.1f}%")
        lines.append("")
        
        # Alpha distribution summary
        if stats.transparency_percentage > 0:
            lines.append("📈 Alpha Channel Analysis:")
            lines.append(f"  • Mean alpha: {stats.alpha_distribution_summary['mean']:.1f}")
            lines.append(f"  • Alpha range: {stats.alpha_distribution_summary['min']:.0f} - {stats.alpha_distribution_summary['max']:.0f}")
            lines.append(f"  • Standard deviation: {stats.alpha_distribution_summary['std']:.1f}")
            lines.append(f"  • Distribution pattern: {stats.transparency_distribution}")
            lines.append("")
        
        # Regional analysis
        if stats.transparency_regions:
            lines.append("🗺️ Transparency Regions:")
            lines.append(f"  • Number of regions: {len(stats.transparency_regions)}")
            lines.append(f"  • Largest region: {stats.largest_transparent_region_size:,} pixels")
            if verbose and len(stats.transparency_regions) > 0:
                lines.append("  • Top regions by size:")
                for i, region in enumerate(stats.transparency_regions[:3]):
                    lines.append(f"    {i+1}. Size: {region['size']:,} pixels at ({region['center_x']}, {region['center_y']})")
            lines.append("")
        
        # Complexity and recommendations
        lines.append("🎯 Analysis Summary:")
        lines.append(f"  • Complexity score: {stats.complexity_score:.2f} (0=simple, 1=complex)")
        lines.append(f"  • Recommendation: {stats.processing_recommendation.replace('_', ' ').title()}")
        lines.append("")
        
        # Specific issues
        if stats.specific_issues:
            lines.append("⚠️ Issues Identified:")
            for issue in stats.specific_issues:
                lines.append(f"  • {issue}")
            lines.append("")
        
        # Verbose details
        if verbose and stats.transparency_percentage > 0:
            lines.append("🔬 Detailed Technical Information:")
            
            # Alpha histogram (top values)
            if stats.alpha_histogram:
                lines.append("  Alpha value distribution (top 10):")
                sorted_histogram = sorted(stats.alpha_histogram.items(), 
                                        key=lambda x: x[1], reverse=True)
                for alpha_val, count in sorted_histogram[:10]:
                    percentage = (count / stats.total_pixels) * 100
                    lines.append(f"    Alpha {alpha_val}: {count:,} pixels ({percentage:.1f}%)")
            lines.append("")
        
        return "\n".join(lines)
    
    def export_statistics_json(self, stats: TransparencyStatistics) -> Dict:
        """
        Export statistics as JSON-serializable dictionary.
        
        Args:
            stats: Transparency statistics to export
            
        Returns:
            Dictionary suitable for JSON serialization
        """
        return {
            "basic_statistics": {
                "total_pixels": stats.total_pixels,
                "transparent_pixel_count": stats.transparent_pixel_count,
                "opaque_pixel_count": stats.opaque_pixel_count,
                "transparency_percentage": stats.transparency_percentage
            },
            "alpha_analysis": {
                "histogram": stats.alpha_histogram,
                "distribution_summary": stats.alpha_distribution_summary,
                "distribution_pattern": stats.transparency_distribution
            },
            "regional_analysis": {
                "regions": stats.transparency_regions,
                "largest_region_size": stats.largest_transparent_region_size,
                "region_count": len(stats.transparency_regions)
            },
            "assessment": {
                "complexity_score": stats.complexity_score,
                "processing_recommendation": stats.processing_recommendation,
                "specific_issues": stats.specific_issues
            }
        }