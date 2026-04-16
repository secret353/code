import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.loss import mesh_laplacian_smoothing

from paper_hyperparams import get_dataset_hparams, normalize_dataset_name
from utils import get_each_mesh


class Paper3Loss(nn.Module):
    def __init__(self, dataset_name: str, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.dataset_name = normalize_dataset_name(dataset_name)
        weights = get_dataset_hparams(self.dataset_name)
        self.lambda_distance = weights["lambda_distance"]
        self.lambda_normal = weights["lambda_normal"]
        self.lambda_laplacian = weights["lambda_laplacian"]
        self.loss_weights = {
            "lambda_distance": self.lambda_distance,
            "lambda_normal": self.lambda_normal,
            "lambda_laplacian": self.lambda_laplacian,
        }
        self.gamma = gamma
        self.eps = eps

    def labelmesh(self, label: torch.Tensor):
        label_volume = label[0, 0].detach().cpu()
        device = label.device
        meshes = {}
        for cls in torch.unique(label_volume):
            cls = int(cls.item())
            if cls == 0:
                continue
            try:
                mesh = get_each_mesh(label_volume, cls)
            except Exception:
                continue
            if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                continue
            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
            normals = self.compute_vertex_normals(vertices, faces)
            meshes[cls] = {"vertices": vertices, "faces": faces, "normals": normals}
        return meshes

    def compute_vertex_normals(
        self, vertices: torch.Tensor, faces: torch.Tensor
    ) -> torch.Tensor:
        if vertices.numel() == 0:
            return vertices
        if faces.numel() == 0:
            return torch.zeros_like(vertices)

        face_vertices = vertices[faces]
        face_normals = torch.cross(
            face_vertices[:, 1] - face_vertices[:, 0],
            face_vertices[:, 2] - face_vertices[:, 0],
            dim=-1,
        )
        vertex_normals = torch.zeros_like(vertices)
        for face_idx in range(3):
            vertex_normals.index_add_(0, faces[:, face_idx], face_normals)
        return F.normalize(vertex_normals, dim=-1, eps=self.eps)

    def sample_binary_targets(
        self, label: torch.Tensor, vertices: torch.Tensor, cls: int
    ) -> torch.Tensor:
        cls_target = (label == cls).float()
        grid = 2 * vertices[None, :, None, None] - 1
        sampled = F.grid_sample(
            cls_target,
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        return sampled[0, 0, :, 0, 0]

    def spatial_consistency_loss(
        self, relative_position: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        relative_position = relative_position.clamp(self.eps, 1.0 - self.eps)
        positive_mask = targets > 0.5
        negative_mask = ~positive_mask

        positive_count = positive_mask.sum().float()
        negative_count = negative_mask.sum().float()
        total_count = positive_count + negative_count
        if total_count.item() == 0:
            return relative_position.new_zeros(())

        alpha = negative_count / (total_count + self.eps)
        loss = relative_position.new_zeros(())
        if positive_count.item() > 0:
            loss = loss + (
                -alpha
                * torch.pow(1.0 - relative_position[positive_mask], self.gamma)
                * torch.log(relative_position[positive_mask])
            ).sum()
        if negative_count.item() > 0:
            loss = loss + (
                -(1.0 - alpha)
                * torch.pow(relative_position[negative_mask], self.gamma)
                * torch.log(1.0 - relative_position[negative_mask])
            ).sum()
        return loss / (total_count + self.eps)

    def surface_distance_loss(
        self, pred_vertices: torch.Tensor, gt_vertices: torch.Tensor
    ) -> torch.Tensor:
        pairwise_dist = torch.cdist(pred_vertices[None], gt_vertices[None]).squeeze(0)
        return (
            pairwise_dist.min(dim=1).values.pow(2).mean()
            + pairwise_dist.min(dim=0).values.pow(2).mean()
        )

    def normal_consistency_loss(
        self,
        pred_vertices: torch.Tensor,
        pred_faces: torch.Tensor,
        gt_vertices: torch.Tensor,
        gt_normals: torch.Tensor,
    ) -> torch.Tensor:
        if pred_faces.numel() == 0 or gt_vertices.numel() == 0:
            return pred_vertices.new_zeros(())
        pred_normals = self.compute_vertex_normals(pred_vertices, pred_faces)
        pairwise_dist = torch.cdist(pred_vertices[None], gt_vertices[None]).squeeze(0)
        nearest_indices = pairwise_dist.argmin(dim=1)
        return F.mse_loss(pred_normals, gt_normals[nearest_indices])

    def forward(self, pred, label: torch.Tensor):
        logit_map = pred["logit_map"]
        final_meshes = pred["final_meshes"]
        final_vertices = pred["final_vertices"]
        final_faces = pred["final_faces"]
        relative_positions = pred["relative_positions"]

        if not final_vertices:
            zero = logit_map.sum() * 0.0
            return {
                "total": zero,
                "spatial": zero,
                "distance": zero,
                "normal": zero,
                "laplacian": zero,
            }

        label_meshes = self.labelmesh(label)
        spatial_loss = logit_map.new_zeros(())
        distance_loss = logit_map.new_zeros(())
        normal_loss = logit_map.new_zeros(())
        laplacian_loss = logit_map.new_zeros(())

        for cls, vertices in final_vertices.items():
            if vertices.numel() == 0:
                continue

            targets = self.sample_binary_targets(label, vertices, cls)
            spatial_loss = spatial_loss + self.spatial_consistency_loss(
                relative_positions[cls], targets
            )

            if cls in label_meshes:
                gt_mesh = label_meshes[cls]
                distance_loss = distance_loss + self.surface_distance_loss(
                    vertices, gt_mesh["vertices"]
                )
                normal_loss = normal_loss + self.normal_consistency_loss(
                    vertices,
                    final_faces[cls],
                    gt_mesh["vertices"],
                    gt_mesh["normals"],
                )

            if cls in final_meshes:
                laplacian_loss = laplacian_loss + mesh_laplacian_smoothing(
                    final_meshes[cls], method="uniform"
                )

        total_loss = (
            spatial_loss
            + self.lambda_distance * distance_loss
            + self.lambda_normal * normal_loss
            + self.lambda_laplacian * laplacian_loss
        )
        return {
            "total": total_loss,
            "spatial": spatial_loss.detach(),
            "distance": distance_loss.detach(),
            "normal": normal_loss.detach(),
            "laplacian": laplacian_loss.detach(),
        }
