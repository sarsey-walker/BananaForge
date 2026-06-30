"""STL file generation from optimized height maps."""

import io
import logging
import struct
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import trimesh


class STLGenerator:
    """Generate STL files from height maps and material assignments."""

    def __init__(
        self,
        layer_height: float = 0.08,
        initial_layer_height: float = 0.16,
        nozzle_diameter: float = 0.4,
        base_height: float = 0.0,
        bottom_mode: str = "simplified",
    ):
        """Initialize STL generator.

        Args:
            layer_height: Height of each layer in mm (default: 0.08mm)
            initial_layer_height: Height of first layer in mm (default: 0.16mm)
            nozzle_diameter: Nozzle diameter in mm (default: 0.4mm)
            base_height: Deprecated extra Z offset in mm; kept for compatibility
            bottom_mode: Bottom face generation mode: full, simplified, or none
        """
        self.layer_height = layer_height
        self.initial_layer_height = initial_layer_height
        self.nozzle_diameter = nozzle_diameter
        self.base_height = base_height
        self.bottom_mode = self._validate_bottom_mode(bottom_mode)
        self.logger = logging.getLogger(__name__)

    def generate_stl(
        self,
        height_map: torch.Tensor,
        output_path: Union[str, Path],
        physical_size: float = 100.0,
        smooth_mesh: bool = True,
        add_base: bool = True,
        bottom_mode: Optional[str] = None,
    ) -> trimesh.Trimesh:
        """Generate STL file from height map.

        Args:
            height_map: Height map tensor (1, 1, H, W) in layer units
            output_path: Output STL file path
            physical_size: Physical size of longest dimension in mm
            physical_size: Scale factor for mesh resolution
            smooth_mesh: Whether to apply mesh smoothing
            add_base: Whether to add a base plate
            bottom_mode: Bottom face generation mode: full, simplified, or none

        Returns:
            Generated trimesh object
        """
        resolved_bottom_mode = self._resolve_bottom_mode(bottom_mode, add_base)

        # Convert height map to numpy
        height_array = height_map.squeeze().cpu().numpy()

        # Scale height map to physical dimensions
        h, w = height_array.shape
        max_dim = max(h, w)
        scale_factor = physical_size / max_dim

        # Physical dimensions
        physical_width = w * scale_factor
        physical_height = h * scale_factor

        # Convert layer units to physical height. The initial layer is the
        # printable base thickness; do not add another base offset later.
        physical_height_map = (
            height_array * self.layer_height + self.initial_layer_height
        )

        mesh = self._create_mesh_from_heightmap(
            physical_height_map,
            physical_width,
            physical_height,
            physical_size,
            bottom_mode=resolved_bottom_mode,
        )

        # Apply smoothing if requested
        if smooth_mesh:
            mesh = self._smooth_mesh(mesh)

        # Save STL file
        mesh.export(str(output_path))

        self.logger.info(f"STL file saved to {output_path}")
        self.logger.info(
            f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        return mesh

    def _create_mesh_from_heightmap(
        self,
        height_map: np.ndarray,
        width: float,
        height: float,
        physical_size: float,
        alpha_mask: Optional[np.ndarray] = None,
        bottom_mode: Optional[str] = None,
    ) -> trimesh.Trimesh:
        """Create 3D mesh from 2D height map with optional alpha mask support."""
        bottom_mode = self._validate_bottom_mode(bottom_mode or self.bottom_mode)
        H, W = height_map.shape
        has_alpha_mask = alpha_mask is not None
        if alpha_mask is None:
            alpha_mask = np.ones((H, W), dtype=bool)
        else:
            # Ensure alpha mask has correct shape and type
            assert alpha_mask.shape == (
                H,
                W,
            ), f"Alpha mask shape {alpha_mask.shape} doesn't match height map {(H, W)}"
            alpha_mask = alpha_mask.astype(bool)

        # Vectorized Creation of Vertices
        j, i = np.meshgrid(np.arange(W), np.arange(H))
        x = j.astype(np.float32)
        y = (H - 1 - i).astype(np.float32)
        z = height_map.astype(np.float32)

        top_vertices = np.stack([x, y, z], axis=2)
        bottom_vertices = top_vertices.copy()
        bottom_vertices[:, :, 2] = 0

        # Scale vertices
        original_max = max(W - 1, H - 1)
        if original_max > 0:
            scale = physical_size / original_max
            top_vertices[:, :, :2] *= scale
            bottom_vertices[:, :, :2] *= scale

        # --- Top and Bottom Surfaces ---
        quad_valid = (
            alpha_mask[:-1, :-1]
            & alpha_mask[:-1, 1:]
            & alpha_mask[1:, 1:]
            & alpha_mask[1:, :-1]
        )
        valid_i, valid_j = np.nonzero(quad_valid)

        if has_alpha_mask:
            top_triangles = self._create_full_top_triangles(top_vertices, quad_valid)
        else:
            top_triangles = self._create_greedy_top_triangles(top_vertices, quad_valid)
        if bottom_mode == "none":
            bottom_triangles = np.empty((0, 3, 3), dtype=np.float32)
        elif bottom_mode == "full":
            bottom_triangles = self._create_full_bottom_triangles(
                bottom_vertices, quad_valid
            )
        else:
            bottom_triangles = self._create_greedy_bottom_triangles(
                bottom_vertices, quad_valid
            )

        # --- Side Walls ---
        side_triangles_list = []

        # Left edges
        left_cond = np.zeros_like(quad_valid, dtype=bool)
        left_cond[:, 0] = quad_valid[:, 0]
        left_cond[:, 1:] = quad_valid[:, 1:] & (~quad_valid[:, :-1])
        li, lj = np.nonzero(left_cond)
        if li.size > 0:
            lv0 = top_vertices[li, lj]
            lv1 = top_vertices[li + 1, lj]
            lb0 = bottom_vertices[li, lj]
            lb1 = bottom_vertices[li + 1, lj]
            left_tris = np.concatenate(
                [np.stack([lv0, lv1, lb1], axis=1), np.stack([lv0, lb1, lb0], axis=1)],
                axis=0,
            )
            side_triangles_list.append(left_tris)

        # Right edges
        right_cond = np.zeros_like(quad_valid, dtype=bool)
        right_cond[:, -1] = quad_valid[:, -1]
        right_cond[:, :-1] = quad_valid[:, :-1] & (~quad_valid[:, 1:])
        ri, rj = np.nonzero(right_cond)
        if ri.size > 0:
            rv0 = top_vertices[ri, rj + 1]
            rv1 = top_vertices[ri + 1, rj + 1]
            rb0 = bottom_vertices[ri, rj + 1]
            rb1 = bottom_vertices[ri + 1, rj + 1]
            right_tris = np.concatenate(
                [np.stack([rv0, rv1, rb1], axis=1), np.stack([rv0, rb1, rb0], axis=1)],
                axis=0,
            )
            side_triangles_list.append(right_tris)

        # Top edges
        top_cond = np.zeros_like(quad_valid, dtype=bool)
        top_cond[0, :] = quad_valid[0, :]
        top_cond[1:, :] = quad_valid[1:, :] & (~quad_valid[:-1, :])
        ti, tj = np.nonzero(top_cond)
        if ti.size > 0:
            tv0 = top_vertices[ti, tj]
            tv1 = top_vertices[ti, tj + 1]
            tb0 = bottom_vertices[ti, tj]
            tb1 = bottom_vertices[ti, tj + 1]
            top_wall_tris = np.concatenate(
                [np.stack([tv0, tv1, tb1], axis=1), np.stack([tv0, tb1, tb0], axis=1)],
                axis=0,
            )
            side_triangles_list.append(top_wall_tris)

        # Bottom edges
        bottom_cond = np.zeros_like(quad_valid, dtype=bool)
        bottom_cond[-1, :] = quad_valid[-1, :]
        bottom_cond[:-1, :] = quad_valid[:-1, :] & (~quad_valid[1:, :])
        bi, bj = np.nonzero(bottom_cond)
        if bi.size > 0:
            bv0_edge = top_vertices[bi + 1, bj]
            bv1_edge = top_vertices[bi + 1, bj + 1]
            bb0 = bottom_vertices[bi + 1, bj]
            bb1 = bottom_vertices[bi + 1, bj + 1]
            bottom_wall_tris = np.concatenate(
                [
                    np.stack([bv0_edge, bv1_edge, bb1], axis=1),
                    np.stack([bv0_edge, bb1, bb0], axis=1),
                ],
                axis=0,
            )
            side_triangles_list.append(bottom_wall_tris)

        side_triangles = (
            np.concatenate(side_triangles_list, axis=0)
            if side_triangles_list
            else np.empty((0, 3, 3), dtype=np.float32)
        )

        # --- Combine All Triangles ---
        all_triangles = np.concatenate(
            [top_triangles, side_triangles, bottom_triangles], axis=0
        )

        # --- Create a Structured Array for Binary STL ---
        stl_dtype = np.dtype(
            [
                ("normal", np.float32, (3,)),
                ("v1", np.float32, (3,)),
                ("v2", np.float32, (3,)),
                ("v3", np.float32, (3,)),
                ("attr", np.uint16),
            ]
        )
        stl_data = np.empty(all_triangles.shape[0], dtype=stl_dtype)

        # --- Compute Normals Vectorized ---
        v1_arr = all_triangles[:, 0, :]
        v2_arr = all_triangles[:, 1, :]
        v3_arr = all_triangles[:, 2, :]
        normals = np.cross(v2_arr - v1_arr, v3_arr - v1_arr)
        norms = np.linalg.norm(normals, axis=1)
        norms[norms == 0] = 1
        normals /= norms[:, np.newaxis]

        stl_data["normal"] = normals
        stl_data["v1"] = all_triangles[:, 0, :]
        stl_data["v2"] = all_triangles[:, 1, :]
        stl_data["v3"] = all_triangles[:, 2, :]
        stl_data["attr"] = 0

        # Write to an in-memory buffer
        buffer = io.BytesIO()
        header = "Binary STL generated by BananaForge".encode("utf-8").ljust(80, b" ")
        buffer.write(header)
        buffer.write(struct.pack("<I", all_triangles.shape[0]))
        buffer.write(stl_data.tobytes())
        buffer.seek(0)

        mesh = trimesh.load(buffer, file_type="stl")
        mesh.merge_vertices()
        return mesh

    @staticmethod
    def _validate_bottom_mode(bottom_mode: str) -> str:
        """Validate and normalize the bottom face generation mode."""
        valid_modes = {"full", "simplified", "none"}
        if bottom_mode not in valid_modes:
            raise ValueError(
                f"bottom_mode must be one of {sorted(valid_modes)}, got {bottom_mode!r}"
            )
        return bottom_mode

    def _resolve_bottom_mode(self, bottom_mode: Optional[str], add_base: bool) -> str:
        """Resolve legacy add_base with the explicit bottom mode option."""
        if not add_base:
            return "none"
        return self._validate_bottom_mode(bottom_mode or self.bottom_mode)

    def _create_greedy_top_triangles(
        self, top_vertices: np.ndarray, quad_valid: np.ndarray
    ) -> np.ndarray:
        """Create top triangles, merging coplanar cell rectangles."""
        z = top_vertices[:, :, 2]
        flat_quads = (
            quad_valid
            & np.isclose(z[:-1, :-1], z[:-1, 1:])
            & np.isclose(z[:-1, :-1], z[1:, 1:])
            & np.isclose(z[:-1, :-1], z[1:, :-1])
        )

        triangles = []

        rounded_z = np.round(z[:-1, :-1], decimals=6)
        for i, j, rect_h, rect_w in self._greedy_rectangles(flat_quads, rounded_z):
            v0 = top_vertices[i, j]
            v1 = top_vertices[i, j + rect_w]
            v2 = top_vertices[i + rect_h, j + rect_w]
            v3 = top_vertices[i + rect_h, j]
            triangles.extend([[v2, v1, v0], [v3, v2, v0]])

        non_flat_i, non_flat_j = np.nonzero(quad_valid & ~flat_quads)
        if non_flat_i.size > 0:
            v0 = top_vertices[non_flat_i, non_flat_j]
            v1 = top_vertices[non_flat_i, non_flat_j + 1]
            v2 = top_vertices[non_flat_i + 1, non_flat_j + 1]
            v3 = top_vertices[non_flat_i + 1, non_flat_j]
            triangles.extend(
                np.concatenate(
                    [
                        np.stack([v2, v1, v0], axis=1),
                        np.stack([v3, v2, v0], axis=1),
                    ],
                    axis=0,
                )
            )

        if not triangles:
            return np.empty((0, 3, 3), dtype=np.float32)

        return np.asarray(triangles, dtype=np.float32)

    def _create_full_top_triangles(
        self, top_vertices: np.ndarray, quad_valid: np.ndarray
    ) -> np.ndarray:
        """Create one pair of top triangles per valid cell."""
        valid_i, valid_j = np.nonzero(quad_valid)
        if valid_i.size == 0:
            return np.empty((0, 3, 3), dtype=np.float32)

        v0 = top_vertices[valid_i, valid_j]
        v1 = top_vertices[valid_i, valid_j + 1]
        v2 = top_vertices[valid_i + 1, valid_j + 1]
        v3 = top_vertices[valid_i + 1, valid_j]

        return np.concatenate(
            [
                np.stack([v2, v1, v0], axis=1),
                np.stack([v3, v2, v0], axis=1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _create_greedy_bottom_triangles(
        self, bottom_vertices: np.ndarray, quad_valid: np.ndarray
    ) -> np.ndarray:
        """Create bottom triangles, merging adjacent valid rectangles."""
        triangles = []
        values = np.zeros_like(quad_valid, dtype=np.float32)

        for i, j, rect_h, rect_w in self._greedy_rectangles(quad_valid, values):
            v0 = bottom_vertices[i, j]
            v1 = bottom_vertices[i, j + rect_w]
            v2 = bottom_vertices[i + rect_h, j + rect_w]
            v3 = bottom_vertices[i + rect_h, j]
            triangles.extend([[v0, v1, v2], [v0, v2, v3]])

        if not triangles:
            return np.empty((0, 3, 3), dtype=np.float32)

        return np.asarray(triangles, dtype=np.float32)

    def _create_full_bottom_triangles(
        self, bottom_vertices: np.ndarray, quad_valid: np.ndarray
    ) -> np.ndarray:
        """Create one pair of bottom triangles per valid cell."""
        valid_i, valid_j = np.nonzero(quad_valid)
        if valid_i.size == 0:
            return np.empty((0, 3, 3), dtype=np.float32)

        v0 = bottom_vertices[valid_i, valid_j]
        v1 = bottom_vertices[valid_i, valid_j + 1]
        v2 = bottom_vertices[valid_i + 1, valid_j + 1]
        v3 = bottom_vertices[valid_i + 1, valid_j]

        return np.concatenate(
            [
                np.stack([v0, v1, v2], axis=1),
                np.stack([v0, v2, v3], axis=1),
            ],
            axis=0,
        ).astype(np.float32, copy=False)

    def _greedy_rectangles(
        self, mask: np.ndarray, values: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Group equal-value true cells into maximal rectangles."""
        rows, cols = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        rectangles = []

        for i in range(rows):
            for j in range(cols):
                if visited[i, j] or not mask[i, j]:
                    continue

                value = values[i, j]
                rect_w = 1
                while (
                    j + rect_w < cols
                    and not visited[i, j + rect_w]
                    and mask[i, j + rect_w]
                    and values[i, j + rect_w] == value
                ):
                    rect_w += 1

                rect_h = 1
                while i + rect_h < rows:
                    row_slice = slice(j, j + rect_w)
                    if np.any(visited[i + rect_h, row_slice]):
                        break
                    if not np.all(mask[i + rect_h, row_slice]):
                        break
                    if not np.all(values[i + rect_h, row_slice] == value):
                        break
                    rect_h += 1

                visited[i : i + rect_h, j : j + rect_w] = True
                rectangles.append((i, j, rect_h, rect_w))

        return rectangles

    def _smooth_mesh(
        self, mesh: trimesh.Trimesh, iterations: int = 2
    ) -> trimesh.Trimesh:
        """Apply Laplacian smoothing to mesh."""
        try:
            # Apply Laplacian smoothing
            try:
                smoothed = mesh.smoothed(iterations=iterations)
                return smoothed
            except AttributeError:
                # Use newer API - just return original mesh for now
                # Full implementation would use trimesh.smoothing.filter_laplacian
                self.logger.warning("Smoothing not available in this trimesh version")
                return mesh
        except Exception as e:
            self.logger.warning(f"Smoothing failed: {e}, returning original mesh")
            return mesh

    def generate_layer_stls(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_ids: list,
        output_dir: Union[str, Path],
        physical_size: float = 100.0,
    ) -> dict:
        """Generate separate STL files for each material layer.

        Args:
            height_map: Height map tensor (1, 1, H, W) in layer units
            material_assignments: Material assignments tensor (L, H, W)
            material_ids: List of material IDs corresponding to assignment indices
            output_dir: Directory to save STL files
            physical_size: Physical size of longest dimension in mm

        Returns:
            Dictionary of material IDs to STL file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        num_layers, h, w = material_assignments.shape
        stl_paths = {}

        for material_idx, material_id in enumerate(material_ids):
            # Create a combined height map for this material across all layers
            material_height_map = torch.zeros_like(height_map)

            for layer_idx in range(num_layers):
                # Mask for current material at current layer
                material_mask = material_assignments[layer_idx] == material_idx

                # Layer height for this material is the base height map where the material is present
                layer_height_values = height_map * material_mask.float()

                # Accumulate heights, taking the max height for each pixel
                material_height_map = torch.max(
                    material_height_map, layer_height_values
                )

            if torch.any(material_height_map > 0):
                stl_path = output_dir / f"{material_id}.stl"

                self.generate_stl(
                    height_map=material_height_map,
                    output_path=stl_path,
                    physical_size=physical_size,
                    smooth_mesh=False,  # Avoid smoothing individual layers
                )

                stl_paths[material_id] = str(stl_path)

        return stl_paths

    def create_preview_mesh(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_colors: torch.Tensor,
        physical_size: float = 100.0,
    ) -> trimesh.Trimesh:
        """Create a single mesh with vertex colors for preview.

        Args:
            height_map: Height map tensor (1, 1, H, W)
            material_assignments: Material assignments (L, H, W)
            material_colors: Colors for each material (N, 3)
            physical_size: Physical size of longest dimension in mm

        Returns:
            Colored trimesh object
        """
        # Generate base mesh
        mesh = self.generate_stl(
            height_map, "temp_preview.stl", physical_size, add_base=True
        )

        # Assign vertex colors
        vertex_colors = self._assign_vertex_colors(
            mesh, height_map, material_assignments, material_colors, physical_size
        )
        mesh.visual.vertex_colors = vertex_colors

        return mesh

    def _assign_vertex_colors(
        self,
        mesh: trimesh.Trimesh,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_colors: torch.Tensor,
        physical_size: float,
    ) -> np.ndarray:
        """Assign vertex colors based on material assignments."""
        num_layers, h, w = material_assignments.shape
        num_vertices = len(mesh.vertices)

        # Default to a gray color
        vertex_colors = np.full((num_vertices, 4), [128, 128, 128, 255], dtype=np.uint8)

        # Map vertex positions back to image coordinates
        scale_factor = physical_size / max(h, w)

        # Invert scaling to get back to pixel coordinates
        coords = mesh.vertices[:, :2] / scale_factor

        # Flip Y-axis
        coords[:, 1] = h - 1 - coords[:, 1]

        # Round to nearest pixel
        pixel_coords = np.round(coords).astype(int)
        pixel_coords = np.clip(pixel_coords, [0, 0], [w - 1, h - 1])

        # Get vertex heights in layer units
        vertex_heights_mm = mesh.vertices[:, 2]
        vertex_layers = (
            vertex_heights_mm - self.initial_layer_height
        ) / self.layer_height
        vertex_layers = np.round(vertex_layers).astype(int)
        vertex_layers = np.clip(vertex_layers, 0, num_layers - 1)

        # Get top-most material at each vertex location
        top_material_indices = material_assignments[
            vertex_layers, pixel_coords[:, 1], pixel_coords[:, 0]
        ]

        # Get corresponding colors
        colors_rgb = material_colors.cpu().numpy() * 255
        color_indices = top_material_indices.cpu().numpy()
        color_indices = np.clip(color_indices, 0, len(colors_rgb) - 1)
        assigned_colors = colors_rgb[color_indices]

        vertex_colors[:, :3] = assigned_colors.astype(np.uint8)

        return vertex_colors

    def analyze_mesh_quality(self, mesh: trimesh.Trimesh) -> dict:
        """Analyze and report mesh quality metrics."""
        # Check if mesh is manifold (no attribute in newer trimesh versions)
        try:
            is_manifold = mesh.is_manifold
        except AttributeError:
            # Closed 2-manifold meshes have exactly two incident faces per edge.
            if len(mesh.edges_unique) == 0:
                is_manifold = False
            else:
                edge_face_counts = np.bincount(
                    mesh.edges_unique_inverse,
                    minlength=len(mesh.edges_unique),
                )
                is_manifold = bool(np.all(edge_face_counts == 2))

        return {
            "watertight": mesh.is_watertight,
            "volume": mesh.volume,
            "euler_number": mesh.euler_number,
            "manifold": is_manifold,
            "face_count": len(mesh.faces),
            "vertex_count": len(mesh.vertices),
        }

    def generate_stl_with_alpha(
        self,
        height_map: torch.Tensor,
        alpha_mask: torch.Tensor,
        output_path: Union[str, Path],
        physical_size: float = 100.0,
        smooth_mesh: bool = True,
        create_boundaries: bool = True,
        ensure_manifold: bool = True,
        clean_edges: bool = True,
        bottom_mode: Optional[str] = None,
    ) -> trimesh.Trimesh:
        """Generate STL file from height map with alpha channel support.

        Args:
            height_map: Height map tensor (1, 1, H, W) in layer units
            alpha_mask: Alpha mask tensor (H, W) where True = opaque (alpha >= 0.5)
            output_path: Output STL file path
            physical_size: Physical size of longest dimension in mm
            smooth_mesh: Whether to apply mesh smoothing
            create_boundaries: Whether to create proper boundaries around transparent regions
            ensure_manifold: Whether to ensure the resulting mesh is manifold
            clean_edges: Whether to clean edges at transparency boundaries
            bottom_mode: Bottom face generation mode: full, simplified, or none

        Returns:
            Generated trimesh object with alpha channel support
        """
        # Convert tensors to numpy
        resolved_bottom_mode = self._validate_bottom_mode(
            bottom_mode or ("full" if ensure_manifold else self.bottom_mode)
        )
        height_array = height_map.squeeze().cpu().numpy()
        alpha_array = alpha_mask.cpu().numpy()

        # Scale height map to physical dimensions
        h, w = height_array.shape
        max_dim = max(h, w)
        scale_factor = physical_size / max_dim

        # Physical dimensions
        physical_width = w * scale_factor
        physical_height = h * scale_factor

        # Convert layer units to physical height. The initial layer is the
        # printable base thickness; do not add another base offset later.
        physical_height_map = (
            height_array * self.layer_height + self.initial_layer_height
        )

        # Create mesh with alpha mask
        mesh = self._create_mesh_from_heightmap(
            physical_height_map,
            physical_width,
            physical_height,
            physical_size,
            alpha_array,
            bottom_mode=resolved_bottom_mode,
        )

        # Post-process mesh for alpha channel support
        if create_boundaries:
            mesh = self._create_alpha_boundaries(mesh, alpha_array, physical_size)

        if ensure_manifold and resolved_bottom_mode != "none":
            mesh = self._ensure_manifold_mesh(mesh)

        if clean_edges:
            mesh = self._clean_alpha_edges(mesh, alpha_array)

        # Apply smoothing if requested
        if smooth_mesh:
            mesh = self._smooth_mesh(mesh)

        # Save STL file
        mesh.export(str(output_path))

        self.logger.info(f"Alpha-enabled STL file saved to {output_path}")
        self.logger.info(
            f"Mesh stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces"
        )

        return mesh

    def generate_layer_stls_with_alpha(
        self,
        height_map: torch.Tensor,
        material_assignments: torch.Tensor,
        material_ids: list,
        alpha_mask: torch.Tensor,
        output_dir: Union[str, Path],
        physical_size: float = 100.0,
    ) -> dict:
        """Generate separate STL files for each material layer with alpha channel support.

        Args:
            height_map: Height map tensor (1, 1, H, W) in layer units
            material_assignments: Material assignments tensor (L, H, W)
            material_ids: List of material IDs corresponding to assignment indices
            alpha_mask: Alpha mask tensor (H, W) where True = opaque
            output_dir: Directory to save STL files
            physical_size: Physical size of longest dimension in mm

        Returns:
            Dictionary of material IDs to STL file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        num_layers, h, w = material_assignments.shape
        stl_paths = {}

        for material_idx, material_id in enumerate(material_ids):
            # Create a combined height map for this material across all layers
            material_height_map = torch.zeros_like(height_map)

            for layer_idx in range(num_layers):
                # Mask for current material at current layer
                material_mask = material_assignments[layer_idx] == material_idx

                # Apply alpha mask - only include material where alpha is opaque
                combined_mask = material_mask & alpha_mask

                # Layer height for this material is the base height map where the material is present
                layer_height_values = height_map * combined_mask.float()

                # Accumulate heights, taking the max height for each pixel
                material_height_map = torch.max(
                    material_height_map, layer_height_values
                )

            # Only generate STL if material has non-zero height in opaque regions
            if torch.any(material_height_map > 0):
                stl_path = output_dir / f"{material_id}.stl"

                # Create material-specific alpha mask (where this material exists)
                material_alpha_mask = material_height_map.squeeze() > 0

                self.generate_stl_with_alpha(
                    height_map=material_height_map,
                    alpha_mask=material_alpha_mask,
                    output_path=stl_path,
                    physical_size=physical_size,
                    smooth_mesh=False,  # Avoid smoothing individual layers
                )

                stl_paths[material_id] = str(stl_path)

        return stl_paths

    def detect_alpha_boundaries(
        self, mesh: trimesh.Trimesh, alpha_mask: np.ndarray
    ) -> List[np.ndarray]:
        """Detect boundary edges at alpha channel transitions."""
        boundaries = []

        alpha_mask = np.asarray(alpha_mask, dtype=bool)
        h, w = alpha_mask.shape

        for i in range(h):
            for j in range(w):
                if not alpha_mask[i, j]:
                    continue

                if i == 0 or not alpha_mask[i - 1, j]:
                    boundaries.append(np.array([[i, j], [i, j + 1]]))
                if i == h - 1 or not alpha_mask[i + 1, j]:
                    boundaries.append(np.array([[i + 1, j], [i + 1, j + 1]]))
                if j == 0 or not alpha_mask[i, j - 1]:
                    boundaries.append(np.array([[i, j], [i + 1, j]]))
                if j == w - 1 or not alpha_mask[i, j + 1]:
                    boundaries.append(np.array([[i, j + 1], [i + 1, j + 1]]))

        return boundaries

    def trace_boundary_loops(
        self, boundary_edges: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Trace boundary edges to form closed loops."""
        if not boundary_edges:
            return []

        def point_key(point: np.ndarray) -> Tuple[int, int]:
            return (int(point[0]), int(point[1]))

        adjacency = {}
        edge_keys = set()
        for edge in boundary_edges:
            start = point_key(edge[0])
            end = point_key(edge[1])
            if start == end:
                continue
            adjacency.setdefault(start, set()).add(end)
            adjacency.setdefault(end, set()).add(start)
            edge_keys.add(tuple(sorted((start, end))))

        loops = []
        visited_edges = set()

        for start_edge in list(edge_keys):
            if start_edge in visited_edges:
                continue

            start, current = start_edge
            loop = [start, current]
            visited_edges.add(start_edge)
            previous = start

            while current != start:
                candidates = [
                    point
                    for point in adjacency.get(current, set())
                    if point != previous
                ]
                next_point = None
                for candidate in candidates:
                    candidate_edge = tuple(sorted((current, candidate)))
                    if candidate_edge not in visited_edges or candidate == start:
                        next_point = candidate
                        break

                if next_point is None:
                    break

                visited_edges.add(tuple(sorted((current, next_point))))
                loop.append(next_point)
                previous, current = current, next_point

            if loop[0] == loop[-1] and len(loop) >= 4:
                loops.append(np.array(loop, dtype=float))

        return loops

    def analyze_edge_quality(
        self, mesh: trimesh.Trimesh, alpha_mask: np.ndarray
    ) -> dict:
        """Analyze edge quality metrics for alpha channel boundaries."""
        # Get mesh edges
        edges = mesh.edges_unique
        edge_count = len(edges)

        # Count non-manifold edges (edges shared by more than 2 faces)
        non_manifold_edges = 0
        open_mesh_edges = 0

        for edge in edges:
            faces_with_edge = []
            for i, face in enumerate(mesh.faces):
                if edge[0] in face and edge[1] in face:
                    faces_with_edge.append(i)

            if len(faces_with_edge) == 1:
                open_mesh_edges += 1
            elif len(faces_with_edge) > 2:
                non_manifold_edges += 1

        # Calculate boundary smoothness (simplified metric)
        boundary_smoothness = max(0.0, 1.0 - (non_manifold_edges / max(edge_count, 1)))

        # Count degenerate faces (faces with very small area)
        degenerate_faces = 0
        for face in mesh.faces:
            vertices = mesh.vertices[face]
            area = 0.5 * np.linalg.norm(
                np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            )
            if area < 1e-10:
                degenerate_faces += 1

        return {
            "non_manifold_edges": non_manifold_edges,
            "boundary_edges": len(self.detect_alpha_boundaries(mesh, alpha_mask)),
            "open_mesh_edges": open_mesh_edges,
            "boundary_smoothness": boundary_smoothness,
            "degenerate_faces": degenerate_faces,
            "total_edges": edge_count,
        }

    def _create_alpha_boundaries(
        self, mesh: trimesh.Trimesh, alpha_mask: np.ndarray, physical_size: float
    ) -> trimesh.Trimesh:
        """Create proper boundaries around transparent regions."""
        # This is a simplified implementation
        # In a full implementation, this would analyze the alpha mask
        # and add appropriate boundary geometry

        boundary_edges = self.detect_alpha_boundaries(mesh, alpha_mask)
        if not boundary_edges:
            return mesh

        # For now, just return the original mesh
        # A full implementation would add boundary faces
        return mesh

    def _ensure_manifold_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Ensure the mesh is manifold by fixing common issues."""
        try:
            # Remove duplicate vertices
            mesh.merge_vertices()

            # Remove degenerate faces
            try:
                mesh.remove_degenerate_faces()
            except AttributeError:
                # Use newer API
                mesh.update_faces(mesh.nondegenerate_faces())

            # Fill holes if needed
            if not mesh.is_watertight:
                mesh.fill_holes()

            return mesh
        except Exception as e:
            self.logger.warning(f"Failed to ensure manifold mesh: {e}")
            return mesh

    def _clean_alpha_edges(
        self, mesh: trimesh.Trimesh, alpha_mask: np.ndarray
    ) -> trimesh.Trimesh:
        """Clean edges at transparency boundaries."""
        try:
            # Remove duplicate vertices and degenerate faces
            mesh.merge_vertices()
            try:
                mesh.remove_degenerate_faces()
            except AttributeError:
                # Use newer API
                mesh.update_faces(mesh.nondegenerate_faces())

            # Additional edge cleaning could be implemented here
            # This is a simplified version

            return mesh
        except Exception as e:
            self.logger.warning(f"Failed to clean alpha edges: {e}")
            return mesh
