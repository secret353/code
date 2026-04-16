import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet_util import (
    PointNetFeaturePropagation,
    PointNetSetAbstraction,
    farthest_point_sample,
    square_distance,
)


class PointCloudEncoder(nn.Module):
    def __init__(self, num_classes, inchannels=64):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = inchannels
        self.sa1 = PointNetSetAbstraction(
            0.06, 32, 6 + 3, [inchannels, inchannels, inchannels * 2], False
        )
        self.sa2 = PointNetSetAbstraction(
            0.1,
            32,
            inchannels * 2 + 3,
            [inchannels * 2, inchannels * 2, inchannels * 4],
            False,
        )
        self.sa3 = PointNetSetAbstraction(
            0.14,
            32,
            inchannels * 4 + 3,
            [inchannels * 4, inchannels * 4, inchannels * 8],
            False,
        )
        self.sa4 = PointNetSetAbstraction(
            0.18,
            32,
            inchannels * 8 + 3,
            [inchannels * 8, inchannels * 8, inchannels * 16],
            False,
        )
        self.fp4 = PointNetFeaturePropagation(
            inchannels * 24, [inchannels * 12, inchannels * 8]
        )
        self.fp3 = PointNetFeaturePropagation(
            inchannels * 12, [inchannels * 8, inchannels * 4]
        )
        self.fp2 = PointNetFeaturePropagation(
            inchannels * 6, [inchannels * 4, inchannels * 2]
        )
        self.fp1 = PointNetFeaturePropagation(
            inchannels * 2, [inchannels * 2, inchannels * 2, inchannels]
        )
        self.conv1 = nn.Conv1d(inchannels, inchannels, 1)
        self.bn1 = nn.InstanceNorm1d(inchannels)
        self.drop1 = nn.Dropout(0.5)

    def split_normals(self, vertices, normals):
        if isinstance(normals, (list, tuple)):
            return [item for item in normals]

        normal_list = []
        start_idx = 0
        for cls_vertices in vertices:
            cls_num_points = cls_vertices.shape[0]
            normal_list.append(normals[start_idx : start_idx + cls_num_points])
            start_idx += cls_num_points
        return normal_list

    def get_target_points(self, vertices):
        counts = [item.shape[0] for item in vertices if item.shape[0] > 0]
        if not counts:
            return 0
        return max(1, int(round(sum(counts) / len(counts))))

    def repeat_indices(self, indices, target_num):
        if indices.shape[0] >= target_num:
            return indices[:target_num]
        repeat_times = (target_num + indices.shape[0] - 1) // indices.shape[0]
        return indices.repeat(repeat_times)[:target_num]

    def standardize_single_organ(self, vertices, normals, target_num):
        num_points = vertices.shape[0]
        if num_points == 0:
            empty_idx = torch.empty(0, dtype=torch.long, device=vertices.device)
            return vertices, normals, empty_idx

        if num_points == target_num:
            sample_idx = torch.arange(num_points, device=vertices.device)
        elif num_points > target_num:
            sample_idx = farthest_point_sample(vertices[None], target_num)[0]
        else:
            base_idx = farthest_point_sample(vertices[None], num_points)[0]
            sample_idx = self.repeat_indices(base_idx, target_num)

        return vertices[sample_idx], normals[sample_idx], sample_idx

    def standardize_vertices(self, vertices, normals):
        target_num = self.get_target_points(vertices)
        sampled_vertices = []
        sampled_normals = []
        sample_indices = []
        for cls_vertices, cls_normals in zip(vertices, normals):
            (
                cls_sampled_vertices,
                cls_sampled_normals,
                cls_sample_idx,
            ) = self.standardize_single_organ(cls_vertices, cls_normals, target_num)
            sampled_vertices.append(cls_sampled_vertices)
            sampled_normals.append(cls_sampled_normals)
            sample_indices.append(cls_sample_idx)
        return sampled_vertices, sampled_normals, sample_indices

    def get_dist(self, vertices, total_points):
        if total_points == 0:
            device = vertices[0].device if vertices else torch.device("cpu")
            return torch.empty((0, 0), device=device)

        init_idx = 0
        dists = torch.ones((total_points, total_points), device=vertices[0].device)
        for cls_vertices in vertices:
            cls_num_points = cls_vertices.shape[0]
            sqrdists = square_distance(cls_vertices[None], cls_vertices[None])[0]
            dists[
                init_idx : init_idx + cls_num_points,
                init_idx : init_idx + cls_num_points,
            ] = sqrdists
            init_idx += cls_num_points
        return dists

    def interpolate_features(self, source_vertices, target_vertices, source_features):
        if source_vertices.shape[0] == target_vertices.shape[0] and torch.equal(
            source_vertices, target_vertices
        ):
            return source_features

        if source_vertices.shape[0] == 1:
            return source_features.repeat(target_vertices.shape[0], 1)

        pairwise_dist = square_distance(target_vertices[None], source_vertices[None])[0]
        k = min(3, source_vertices.shape[0])
        dists, idx = torch.topk(pairwise_dist, k, largest=False, sorted=False)
        weights = torch.softmax(-dists, dim=-1)
        gathered_features = source_features[idx]
        return torch.sum(gathered_features * weights.unsqueeze(-1), dim=1)

    def restore_full_resolution_features(
        self, original_vertices, sampled_vertices, sampled_features
    ):
        restored_features = []
        start_idx = 0
        for cls_original_vertices, cls_sampled_vertices in zip(
            original_vertices, sampled_vertices
        ):
            cls_num_sampled = cls_sampled_vertices.shape[0]
            cls_sampled_features = sampled_features[
                start_idx : start_idx + cls_num_sampled
            ]
            start_idx += cls_num_sampled
            restored_features.append(
                self.interpolate_features(
                    cls_sampled_vertices, cls_original_vertices, cls_sampled_features
                )
            )
        return torch.cat(restored_features, dim=0)[None]

    def encode_point_cloud(self, sampled_vertices, sampled_normals):
        total_sampled_points = sum(item.shape[0] for item in sampled_vertices)
        if total_sampled_points == 0:
            device = sampled_vertices[0].device if sampled_vertices else torch.device("cpu")
            return torch.empty((1, 0, self.feature_dim), device=device)

        dists = self.get_dist(sampled_vertices, total_sampled_points)
        xyz = torch.cat(
            [torch.cat(sampled_vertices, dim=0), torch.cat(sampled_normals, dim=0)],
            dim=-1,
        )[None]
        xyz = xyz.permute(0, 2, 1)
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points, fps_idx = self.sa1(l0_xyz, l0_points, dists)
        dists2 = dists[fps_idx[0]][:, fps_idx[0]]
        l2_xyz, l2_points, fps_idx2 = self.sa2(l1_xyz, l1_points, dists2)
        dists3 = dists2[fps_idx2[0]][:, fps_idx2[0]]
        l3_xyz, l3_points, fps_idx3 = self.sa3(l2_xyz, l2_points, dists3)
        dists4 = dists3[fps_idx3[0]][:, fps_idx3[0]]
        l4_xyz, l4_points, fps_idx4 = self.sa4(l3_xyz, l3_points, dists4)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points, fps_idx4, dists4)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points, fps_idx3, dists3)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points, fps_idx2, dists2)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points, fps_idx, dists)

        point_features = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        return point_features.permute(0, 2, 1)

    def forward(self, vertices, normals, istrain=False):
        if not vertices:
            device = normals.device if torch.is_tensor(normals) else torch.device("cpu")
            return torch.empty((1, 0, self.feature_dim), device=device)

        original_vertices = [item for item in vertices]
        normal_list = self.split_normals(original_vertices, normals)
        sampled_vertices, sampled_normals, _ = self.standardize_vertices(
            original_vertices, normal_list
        )
        sampled_features = self.encode_point_cloud(sampled_vertices, sampled_normals)[0]
        return self.restore_full_resolution_features(
            original_vertices, sampled_vertices, sampled_features
        )


class get_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, weight=None):
        return F.nll_loss(pred, target, weight=weight)


def get_model(num_classes, inchannels=64):
    return PointCloudEncoder(num_classes=num_classes, inchannels=inchannels)


if __name__ == "__main__":
    model = get_model(13)
    vertices = [torch.rand(128, 3), torch.rand(96, 3), torch.rand(160, 3)]
    normals = F.normalize(torch.rand(sum(item.shape[0] for item in vertices), 3), dim=-1)
    point_features = model(vertices, normals)
    print(point_features.shape)
