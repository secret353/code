import torch.nn as nn
from model import get_model
from pointnet import get_model as get_point_model
from rasterize import rasterize
from utils import load_pretrained
import copy
import torch
import numpy as np
from utils import get_each_mesh
import torch.nn.functional as F
import trimesh
from functools import reduce
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from paper_hyperparams import get_dataset_hparams, get_shared_hparams, normalize_dataset_name
from pytorch3d.structures import Meshes
from pointnet_util import square_distance, index_points
from rasterize.rasterize import Rasterize

try:
    import pymesh
except ModuleNotFoundError:
    pymesh = None


def fix_mesh(mesh, target_len=(1.0, 1.0, 1.0), detail="normal"):
    # bbox_min, bbox_max = mesh.bbox
    # diag_len = norm(bbox_max - bbox_min)
    # if detail == "normal":
    #     target_len = diag_len * 5e-3
    # elif detail == "high":
    #     target_len = diag_len * 2.5e-3
    # elif detail == "low":
    #     target_len = diag_len * 1e-2
    # target_len = 1
    # print("Target resolution: {} mm".format(target_len))

    count = 0
    mesh, _ = pymesh.remove_degenerated_triangles(mesh, 5)
    # start_time = time.time()
    mesh, _ = pymesh.split_long_edges(mesh, max(target_len) * 2.3)
    # print(f"split_long_edges time: {time.time() - start_time}")
    # mesh, __ = pymesh.split_long_edges(mesh, 4.0)
    num_vertices = mesh.num_vertices
    while count < 3:
        # mesh, _ = pymesh.collapse_short_edges(mesh, 1e-6)
        mesh, __ = pymesh.collapse_short_edges(
            mesh, min(target_len) * 0.5, preserve_feature=True
        )
        # mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 5)
        if mesh.num_vertices == num_vertices:
            break

        num_vertices = mesh.num_vertices
        print("#v: {}".format(num_vertices))
        count += 1

    # mesh = pymesh.resolve_self_intersection(mesh)
    # mesh, __ = pymesh.remove_duplicated_faces(mesh)
    # mesh = pymesh.compute_outer_hull(mesh)
    # mesh, __ = pymesh.remove_duplicated_faces(mesh)
    # mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5)
    # mesh, __ = pymesh.remove_isolated_vertices(mesh)

    return mesh


class GraphConv(nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]

    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.fc = nn.Linear(in_features, out_features)
        self.neighbours_fc = nn.Linear(in_features * 6, out_features)
        # self.neighbours_fc = nn.Conv1d(
        #     in_features*3, out_features, kernel_size=(1), padding=0)

        # self.bc = nn.BatchNorm1d(out_features) if batch_norm else Non()

    def forward(self, inputs, neighbors_index):
        neighbors_feat = torch.index_select(
            inputs.flatten(start_dim=0, end_dim=1), 0, neighbors_index.view(-1)
        )
        neighbors_feat = neighbors_feat.view(
            inputs.shape[0], -1, neighbors_index.shape[-1] * inputs.shape[-1]
        )  # (b, n, k*c)
        neighbors_feat = torch.cat([inputs, neighbors_feat], dim=-1)
        # neighbors_feat = neighbors_feat * neighbors_attn[..., None]
        # neighbors_feat = neighbors_feat.sum(-2)  # (b, n, c)
        # neighbors_feat = neighbors_feat * mask.unsqueeze(-1)
        # neighbors_feat = neighbors_feat.sum(-2)  # (b, n, c)
        # neighbors_feat = neighbors_feat / \
        #     mask.sum(-1, keepdim=True)  # (b, n, c)

        neighbors_feat = self.neighbours_fc(neighbors_feat)

        return neighbors_feat

    def extra_repr(self):
        return "in_features={}, out_features={}".format(
            self.in_features, self.out_features is not None
        )


class Feature2DeltaLayer(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = GraphConv(in_channels, in_channels // 2)
        self.lrelu = get_act_layer(
            name=("leakyrelu", {"inplace": True, "negative_slope": 0.01})
        )
        self.norm1 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv2 = GraphConv(in_channels // 2, in_channels // 2)
        self.norm2 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv3 = nn.Linear(in_channels, in_channels // 2)
        self.norm3 = get_norm_layer(
            name="instance", spatial_dims=1, channels=in_channels // 2
        )
        self.conv4 = nn.Linear(in_channels // 2, 1, bias=False)
        self.conv4.weight.data.zero_()
        # self.conv4.bias.data.zero_()

    def forward(self, features, neighbors_index):
        residual = features
        x = self.conv1(features, neighbors_index)
        x = self.norm1(x.transpose(1, 2)).transpose(1, 2)
        x = self.lrelu(x)
        x = self.conv2(x, neighbors_index)
        x = self.norm2(x.transpose(1, 2)).transpose(1, 2)
        residual = self.conv3(residual)
        residual = self.norm3(residual.transpose(1, 2)).transpose(1, 2)
        x += residual
        x = self.lrelu(x)
        out = self.conv4(x)
        return out


"""
class Paper3_model(nn.Module):
    def __init__(
        self,
        model_name,
        in_channels,
        base_channels,
        num_classes,
        roi_size,
        # pretrained_dir,
        logger,
    ):
        super().__init__()

        self.roi_size = roi_size
        self.num_classes = num_classes
        # 图像分支
        self.img_module = get_model(
            model_name,
            in_channels=in_channels,
            out_channels=num_classes,
            roi_size=roi_size,
        )
        # load_pretrained(self.model, pretrained_dir, logger, strict=False)
        self.pretrained_img_module = copy.deepcopy(self.img_module)
        # load_pretrained(self.pretrained_model, pretrained_dir, logger, strict=False)
        for param in self.pretrained_img_module.parameters():
            param.requires_grad = False
        # 点云分支
        self.point_module = get_point_model(
            num_classes=num_classes, inchannels=base_channels
        )
        # 图像和点云融合
        self.img_point_fuse = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels), nn.ReLU()
        )
        # 偏置距离
        self.f2v = Feature2DeltaLayer(base_channels)

    # 获取每个顶点的邻域顶点索引
    def get_neighbors(self, mesh):
        # 步骤 1: 计算平均长度
        # lengths = [len(x) for x in mesh.vertex_neighbors]
        # average_length = int(sum(lengths) / len(lengths))
        average_length = 5
        # 步骤 2: 调整每个列表的长度
        adjusted_data = []
        for neighbor in mesh.vertex_neighbors:
            if len(neighbor) >= average_length:
                # 如果列表长度大于平均长度，则截取前n个元素
                adjusted_data.append(neighbor[:average_length])
            else:
                # 如果列表长度小于平均长度，则通过重复采样扩展到n个元素
                repeats = average_length // len(neighbor) + 1
                repeated_data = (neighbor * repeats)[:average_length]
                adjusted_data.append(repeated_data)
        # 步骤 3: 转换为张量
        adjusted_neighbors = torch.tensor(adjusted_data)
        return adjusted_neighbors

    # 获取每个顶点的坐标和法向量
    def get_vertices_normals(self, mesh):
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        vertices[..., 0] = vertices[..., 0] / (self.roi_size[-3] - 1)
        vertices[..., 1] = vertices[..., 1] / (self.roi_size[-2] - 1)
        vertices[..., 2] = vertices[..., 2] / (self.roi_size[-1] - 1)

        normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)  # 法向量
        mesh.vertices = vertices
        # vertices = torch.cat([vertices, normals], dim=-1)
        return mesh, vertices, normals

    # 预测的体积转网格
    def volume2mesh(self, img):
        pred = self.pretrained_img_module(img, need_feat=False)  # (b, c, h, w, d)
        # to (b, 1, h, w, d)
        pred = torch.argmax(pred, dim=1, keepdim=False)[0]
        all_cls = np.unique(pred)
        meshes = {}
        neighbors = []
        vertices = []
        normals = []
        init_index = 0
        for cls in all_cls[1:]:  # 不考虑背景0
            mesh = get_each_mesh(pred.cpu(), cls)
            mesh, ver, nor = self.get_vertices_normals(
                mesh
            )  # mesh的顶点坐标归一化到[0, 1]
            meshes[cls] = mesh
            vertices.append(ver.cuda())  # [0, 1]
            normals.append(nor)
            neighbors.append(self.get_neighbors(mesh) + init_index)  # 注意更新邻居索引

            init_index += len(mesh.vertices)

        neighbors = torch.cat(neighbors, dim=0).cuda()
        # vertices = torch.cat(vertices, dim=0).cuda()
        normals = torch.cat(normals, dim=0).cuda()

        return meshes, neighbors, vertices, normals

    # 计算两个点之间的距离
    def get_dist(self, vertices, new_vertices):
        # sorted_dists_list = []
        # sorted_idx_list = []
        init_num = 0
        init_num_new = 0
        length = sum(len(sublist) for sublist in vertices)
        new_length = sum(len(sublist) for sublist in new_vertices)
        # max_nums = max([vertices[i].shape[0] for i in range(vertices.shape[0])])
        dists = torch.ones((new_length, length), device=vertices[0].device)
        # idxs = -1 * torch.ones((len_nums, len_nums))
        for i in range(len(vertices)):
            sqrdists = square_distance(
                new_vertices[i][None], vertices[i][None]
            )  # (b, m, n)
            dists[
                init_num_new : init_num_new + new_vertices[i].shape[0],
                init_num : init_num + vertices[i].shape[0],
            ] = sqrdists[0]
            # 排序
            # _, idx = sqrdists.sort(dim=-1)
            # idx += init_num
            # idxs[
            #     init_num : init_num + vertices[i].shape[0],
            #     init_num : init_num + vertices[i].shape[0],
            # ] = idx
            # # 创建一个全1的张量，大小为 (n, k-m)
            # ones = -1 * torch.ones(dists.shape[0], max_nums - dists.shape[1])
            # dists = torch.cat([dists, ones], dim=-1)
            # idx = torch.cat([idx, ones], dim=-1)
            # sorted_dists_list.append(dists)
            # sorted_idx_list.append(idx + init_num)
            init_num += vertices[i].shape[0]
            init_num_new += new_vertices[i].shape[0]

        # sorted_dists = torch.cat(sorted_dists_list, dim=0)
        # sorted_idx = torch.cat(sorted_idx_list, dim=0)
        return dists

    # 每次迭代后更新点云特征
    def update_points_feat(self, points_feat, vertices, new_vertices):
        # dists = square_distance(new_vertices[None], vertices[None])
        dists = self.get_dist(new_vertices, vertices)
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

        dists, idx = torch.topk(dists, 3, largest=False, sorted=False)

        # dist_recip = 1.0 / (dists + 1e-8)
        # norm = torch.sum(dist_recip, dim=2, keepdim=True)
        # weight0 = dist_recip / norm
        weight = torch.softmax(-dists, dim=-1)
        interpolated_points_feat = torch.sum(
            index_points(points_feat, idx[None]) * weight.view(1, dists.shape[0], 3, 1),
            dim=2,
        )

        return interpolated_points_feat

    # 每次迭代后更新顶点坐标
    def iter_move_module(
        self, img_feat, points_feat, vertices, new_vertices, neighbors, i
    ):
        # 更新points_feat
        if i != 0:
            points_feat = self.update_points_feat(points_feat, vertices, new_vertices)
            vertices = new_vertices

        vertices = torch.cat(vertices, dim=0).cuda()
        # vertices = 2 * vertices - 1
        vertices = vertices[None][:, :, None, None]
        img_feat = F.grid_sample(
            img_feat,
            2 * vertices - 1,  # 修正为[-1, 1]范围
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[
            :, :, :, 0, 0
        ]  # (b, c, n)
        img_feat = img_feat.permute(0, 2, 1)  # (b, n, c)

        feat = torch.cat([img_feat, points_feat], dim=-1)  # (b, n, 2c)
        feat = self.img_point_fuse(feat)
        # 偏置距离
        d_delta = self.f2v(feat, neighbors)
        return d_delta

    # 自适应拓扑优化模块
    def ATMO(self, vertices, faces):
        new_mesh = pymesh.form_mesh(vertices.cpu().detach().numpy(), faces)
        new_mesh = fix_mesh(
            new_mesh,
            target_len=(
                1.0 / self.roi_size[0],
                1.0 / self.roi_size[1],
                1.0 / self.roi_size[2],
            ),
        )
        new_mesh = trimesh.Trimesh(vertices=new_mesh.vertices, faces=new_mesh.faces)
        # if new_mesh.vertices.shape[0] < 10:
        #     return None
        new_mesh = trimesh.smoothing.filter_taubin(
            new_mesh, nu=0.5, lamb=0.5, iterations=5
        )
        return new_mesh

    # 更新网格
    def update_meshes(self, logit_map, d_delta, meshes, need_amto=True):
        init_num = 0
        init_num_new = 0
        train_meshes = {}
        new_vertices = []
        new_neighbors = []
        for cls, mesh in meshes.items():
            vertices = torch.tensor(mesh.vertices, dtype=torch.float32).cuda()
            num_vertices = len(vertices)
            normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32).cuda()
            logit_vertices = F.grid_sample(
                logit_map,
                2 * vertices[None, :, None, None] - 1,  # [-1, 1]
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )[:, :, :, 0, 0]
            logit_vertices = logit_vertices.permute(0, 2, 1)  # (b, n, cls)
            # logit_vertices = torch.sigmoid(logit_vertices) - 0.5
            logit_vertices = (logit_vertices[..., cls] - logit_vertices[..., 0]) / (
                torch.abs(logit_vertices[..., cls] - logit_vertices[..., 0]) + 1e-8
            )  # 1, -1 判断是前景还是背景
            update_vertices = (
                vertices
                + d_delta[0, init_num : init_num + num_vertices, :]
                * logit_vertices[0][:, None]
                * normals
            )
            init_num += num_vertices
            # 限制new_vertices在0-1之间
            update_vertices = torch.clip(update_vertices, 0, 1)
            train_mesh = Meshes(
                verts=list(update_vertices[None]),
                faces=list(torch.tensor(mesh.faces, dtype=torch.float32).cuda()[None]),
            )
            train_meshes[cls] = train_mesh

            # 自适应拓扑优化
            if need_amto:
                new_mesh = self.ATMO(update_vertices, mesh.faces)
                # new_meshes.append(new_mesh)
                new_vertices.append(
                    torch.tensor(new_mesh.vertices, dtype=torch.float32).cuda()
                )
                new_neighbors.append(self.get_neighbors(new_mesh) + init_num_new)

                num_new_vertices = len(new_vertices)
                init_num_new += num_new_vertices
            else:
                new_vertices.append(update_vertices)
                new_neighbors.append(self.get_neighbors(mesh) + init_num_new)
                num_new_vertices = len(new_vertices)
                init_num_new += num_new_vertices

        # new_vertices = torch.cat(new_vertices, dim=0).cuda()
        new_neighbors = torch.cat(new_neighbors, dim=0).cuda()

        return train_meshes, new_vertices, new_neighbors

    def forward(self, x, istrain=False):
        meshes, neighbors, vertices, normals = self.volume2mesh(x)
        img_feat, logit_map = self.img_module(x)
        points_feat = self.point_module(vertices, normals)  # (b, n, c)

        iter_meshes = []
        iter_vertices = []
        new_vertices = vertices
        # 迭代三次
        for i in range(3):
            # 偏置距离
            d_delta = self.iter_move_module(
                img_feat, points_feat, vertices, new_vertices, neighbors, i
            )
            vertices = new_vertices
            # 移动顶点
            train_meshes, new_vertices, neighbors = self.update_meshes(
                logit_map, d_delta, meshes, need_amto=False
            )
            iter_meshes.append(train_meshes)
            iter_vertices.append(torch.cat(new_vertices, dim=0))

        vertices = torch.cat(iter_vertices, dim=0)

        if istrain:
            return logit_map, iter_meshes, vertices
        else:
            # 测试推理
            # pred_voxels = torch.zeros_like(x)[0].long()
            # 直接根据顶点位置进行光栅化，获取体素输出
            pred_voxels = torch.zeros(
                (
                    1,
                    self.num_classes,
                    self.roi_size[0],
                    self.roi_size[1],
                    self.roi_size[2],
                ),
                dtype=torch.long,
            )
            for i, (cls, mesh) in enumerate(train_meshes.items()):
                cls_vertices = new_vertices[i]
                cls_vertices[..., 0] *= self.roi_size[-3] - 1
                cls_vertices[..., 1] *= self.roi_size[-2] - 1
                cls_vertices[..., 2] *= self.roi_size[-1] - 1
                cls_faces = mesh._faces_list[0]

                rasterizer = Rasterize(self.roi_size)
                pred_cls_voxels = rasterizer(
                    cls_vertices[None][..., [2, 1, 0]], cls_faces[None]
                ).long()  # (b, h, w, d)
                pred_voxels[0, cls][pred_cls_voxels[0] == 1] = 1

            pred_voxels[:, 0, ...] = (pred_voxels.sum(dim=1) == 0).int()

            return pred_voxels

    def __init__(
        self,
        model_name,
        in_channels,
        base_channels,
        num_classes,
        roi_size,
        logger,
        dataset_name="BTCV",
        evolution_iters=60,
        topology_interval=10,
        enable_topology=True,
    ):
        super().__init__()
        self.dataset_name = normalize_dataset_name(dataset_name)
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.logger = logger
        self.evolution_iters = evolution_iters
        self.topology_interval = topology_interval
        self.enable_topology = enable_topology
        self.topology_threshold = DATASET_MODEL_SETTINGS[self.dataset_name][
            "topology_threshold"
        ]
        self.use_pretrained_init = False

        self.img_module = get_model(
            model_name,
            in_channels=in_channels,
            out_channels=num_classes,
            roi_size=roi_size,
        )
        self.pretrained_img_module = copy.deepcopy(self.img_module)
        for param in self.pretrained_img_module.parameters():
            param.requires_grad = False

        self.point_module = get_point_model(
            num_classes=num_classes, inchannels=base_channels
        )
        self.img_point_fuse = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels), nn.ReLU()
        )
        self.f2v = Feature2DeltaLayer(base_channels)

    def set_pretrained_init(self, enabled: bool = True):
        self.use_pretrained_init = enabled

    def compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor):
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
        return F.normalize(vertex_normals, dim=-1, eps=1e-6)

    def get_neighbors(self, mesh):
        average_length = 5
        adjusted_data = []
        for vertex_idx, neighbor in enumerate(mesh.vertex_neighbors):
            neighbor = list(neighbor)
            if not neighbor:
                neighbor = [vertex_idx]
            if len(neighbor) >= average_length:
                adjusted_data.append(neighbor[:average_length])
            else:
                repeats = average_length // len(neighbor) + 1
                adjusted_data.append((neighbor * repeats)[:average_length])
        return torch.as_tensor(adjusted_data, dtype=torch.long)

    def normalize_vertices(self, vertices: torch.Tensor):
        vertices = vertices.clone()
        vertices[..., 0] = vertices[..., 0] / (self.roi_size[-3] - 1)
        vertices[..., 1] = vertices[..., 1] / (self.roi_size[-2] - 1)
        vertices[..., 2] = vertices[..., 2] / (self.roi_size[-1] - 1)
        return vertices

    def build_surface_mesh(self, vertices: torch.Tensor, faces: torch.Tensor):
        return trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        )

    def extract_mesh(self, pred_volume: torch.Tensor, cls: int):
        try:
            mesh = get_each_mesh(pred_volume, cls)
        except Exception:
            return None
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        return mesh

    def volume2mesh(self, coarse_logits: torch.Tensor):
        device = coarse_logits.device
        pred = torch.argmax(coarse_logits, dim=1)[0].detach().cpu()
        meshes = {}
        vertices_by_cls = {}
        faces_by_cls = {}
        neighbors = []
        normals = []
        init_index = 0

        for cls in torch.unique(pred):
            cls = int(cls.item())
            if cls == 0:
                continue
            mesh = self.extract_mesh(pred, cls)
            if mesh is None:
                continue

            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
            vertices = self.normalize_vertices(vertices)
            normals_cls = self.compute_vertex_normals(vertices, faces)
            surface_mesh = self.build_surface_mesh(vertices, faces)

            meshes[cls] = surface_mesh
            vertices_by_cls[cls] = vertices
            faces_by_cls[cls] = faces
            normals.append(normals_cls)
            neighbors.append(self.get_neighbors(surface_mesh).to(device) + init_index)
            init_index += vertices.shape[0]

        if neighbors:
            neighbors = torch.cat(neighbors, dim=0)
            normals = torch.cat(normals, dim=0)
        else:
            neighbors = torch.empty((0, 5), dtype=torch.long, device=device)
            normals = torch.empty((0, 3), dtype=torch.float32, device=device)

        return meshes, vertices_by_cls, faces_by_cls, neighbors, normals

    def get_dist(self, vertices, new_vertices):
        if not vertices or not new_vertices:
            device = vertices[0].device if vertices else new_vertices[0].device
            return torch.empty((0, 0), device=device)

        init_num = 0
        init_num_new = 0
        length = sum(item.shape[0] for item in vertices)
        new_length = sum(item.shape[0] for item in new_vertices)
        dists = torch.full((new_length, length), float("inf"), device=vertices[0].device)

        for old_vertices, updated_vertices in zip(vertices, new_vertices):
            sqrdists = square_distance(updated_vertices[None], old_vertices[None])
            dists[
                init_num_new : init_num_new + updated_vertices.shape[0],
                init_num : init_num + old_vertices.shape[0],
            ] = sqrdists[0]
            init_num += old_vertices.shape[0]
            init_num_new += updated_vertices.shape[0]
        return dists

    def update_points_feat(self, points_feat, vertices, new_vertices):
        dists = self.get_dist(new_vertices, vertices)
        if dists.numel() == 0:
            return points_feat

        k = min(3, dists.shape[-1])
        dists, idx = torch.topk(dists, k, largest=False, sorted=False)
        weight = torch.softmax(-dists, dim=-1)
        return torch.sum(
            index_points(points_feat, idx[None]) * weight.view(1, dists.shape[0], k, 1),
            dim=2,
        )

    def sample_image_features(self, img_feat: torch.Tensor, vertices_by_cls):
        vertices = torch.cat(list(vertices_by_cls.values()), dim=0)
        grid = 2 * vertices[None, :, None, None] - 1
        sampled = F.grid_sample(
            img_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, :, 0, 0]
        return sampled.permute(0, 2, 1)

    def predict_vertex_offsets(
        self,
        img_feat,
        points_feat,
        previous_vertices,
        current_vertices,
        current_neighbors,
        step_idx,
    ):
        current_list = list(current_vertices.values())
        if step_idx != 0:
            previous_list = list(previous_vertices.values())
            points_feat = self.update_points_feat(points_feat, previous_list, current_list)

        img_vertex_feat = self.sample_image_features(img_feat, current_vertices)
        feat = torch.cat([img_vertex_feat, points_feat], dim=-1)
        feat = self.img_point_fuse(feat)
        return self.f2v(feat, current_neighbors), points_feat

    def sample_relative_position(
        self, logit_map: torch.Tensor, vertices: torch.Tensor, cls: int
    ):
        grid = 2 * vertices[None, :, None, None] - 1
        sampled_logits = F.grid_sample(
            logit_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )[:, :, :, 0, 0]
        sampled_logits = sampled_logits.permute(0, 2, 1)[0]
        sampled_probs = torch.softmax(sampled_logits, dim=-1)
        class_prob = sampled_probs[:, cls]
        background_prob = sampled_probs[:, 0]
        return (class_prob / (class_prob + background_prob + 1e-6)).clamp(1e-4, 1 - 1e-4)

    def mesh_complexity(self, vertices: torch.Tensor, mesh) -> float:
        laplacian_terms = []
        for vertex_idx, neighbor in enumerate(mesh.vertex_neighbors):
            neighbor = list(neighbor)
            if not neighbor:
                continue
            neighbor_tensor = vertices[torch.as_tensor(neighbor, device=vertices.device)]
            laplacian_terms.append(
                torch.norm(vertices[vertex_idx] - neighbor_tensor.mean(dim=0), p=2)
            )
        if not laplacian_terms:
            return 0.0
        return torch.stack(laplacian_terms).mean().item()

    def should_apply_topology(self, iteration_idx: int, meshes, vertices_by_cls):
        if not self.enable_topology or (iteration_idx + 1) % self.topology_interval != 0:
            return False
        complexities = [
            self.mesh_complexity(vertices_by_cls[cls], mesh)
            for cls, mesh in meshes.items()
            if cls in vertices_by_cls
        ]
        if not complexities:
            return False
        return float(np.mean(complexities)) > self.topology_threshold

    def ATMO(self, vertices: torch.Tensor, faces: torch.Tensor):
        try:
            new_mesh = pymesh.form_mesh(
                vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
            )
            new_mesh = fix_mesh(
                new_mesh,
                target_len=(
                    1.0 / self.roi_size[0],
                    1.0 / self.roi_size[1],
                    1.0 / self.roi_size[2],
                ),
            )
            refined_mesh = trimesh.Trimesh(
                vertices=new_mesh.vertices, faces=new_mesh.faces, process=False
            )
            smoothed_mesh = trimesh.smoothing.filter_taubin(
                refined_mesh, nu=0.5, lamb=0.5, iterations=5
            )
            return smoothed_mesh if smoothed_mesh is not None else refined_mesh
        except Exception:
            return None

    def update_meshes(self, logit_map, d_delta, meshes, need_amto):
        device = logit_map.device
        init_num = 0
        init_num_new = 0
        new_meshes = {}
        new_vertices_by_cls = {}
        new_faces_by_cls = {}
        new_neighbors = []

        for cls, mesh in meshes.items():
            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
            normals = self.compute_vertex_normals(vertices, faces)
            relative_position = self.sample_relative_position(logit_map, vertices, cls)
            num_vertices = vertices.shape[0]
            direction = torch.where(
                relative_position[:, None] >= 0.5,
                torch.ones_like(vertices),
                -torch.ones_like(vertices),
            )
            updated_vertices = torch.clamp(
                vertices
                + d_delta[0, init_num : init_num + num_vertices] * direction * normals,
                0.0,
                1.0,
            )
            init_num += num_vertices

            refined_mesh = None
            if need_amto:
                refined_mesh = self.ATMO(updated_vertices, faces)
            if refined_mesh is None:
                refined_mesh = self.build_surface_mesh(updated_vertices, faces)

            refined_vertices = torch.as_tensor(
                refined_mesh.vertices, dtype=torch.float32, device=device
            )
            refined_faces = torch.as_tensor(
                refined_mesh.faces, dtype=torch.long, device=device
            )
            surface_mesh = self.build_surface_mesh(refined_vertices, refined_faces)

            new_meshes[cls] = surface_mesh
            new_vertices_by_cls[cls] = refined_vertices
            new_faces_by_cls[cls] = refined_faces
            new_neighbors.append(self.get_neighbors(surface_mesh).to(device) + init_num_new)
            init_num_new += refined_vertices.shape[0]

        if new_neighbors:
            new_neighbors = torch.cat(new_neighbors, dim=0)
        else:
            new_neighbors = torch.empty((0, 5), dtype=torch.long, device=device)

        return new_meshes, new_vertices_by_cls, new_faces_by_cls, new_neighbors

    def build_pytorch3d_meshes(self, vertices_by_cls, faces_by_cls, device):
        train_meshes = {}
        for cls, vertices in vertices_by_cls.items():
            train_meshes[cls] = Meshes(
                verts=[vertices],
                faces=[faces_by_cls[cls].to(device=device, dtype=torch.long)],
            )
        return train_meshes

    def rasterize_prediction(self, vertices_by_cls, faces_by_cls, device):
        pred_voxels = torch.zeros(
            (1, self.num_classes, self.roi_size[0], self.roi_size[1], self.roi_size[2]),
            dtype=torch.float32,
            device=device,
        )
        rasterizer = Rasterize(self.roi_size)
        for cls, vertices in vertices_by_cls.items():
            scaled_vertices = vertices.clone()
            scaled_vertices[..., 0] *= self.roi_size[-3] - 1
            scaled_vertices[..., 1] *= self.roi_size[-2] - 1
            scaled_vertices[..., 2] *= self.roi_size[-1] - 1
            cls_faces = faces_by_cls[cls]
            pred_cls_voxels = rasterizer(
                scaled_vertices[None][..., [2, 1, 0]], cls_faces[None]
            ).float()
            pred_voxels[0, cls][pred_cls_voxels[0] > 0] = 1.0

        pred_voxels[:, 0, ...] = (pred_voxels[:, 1:, ...].sum(dim=1) == 0).float()
        return pred_voxels

    def forward(self, x, istrain=False):
        device = x.device
        img_feat, logit_map = self.img_module(x)
        if self.use_pretrained_init:
            with torch.no_grad():
                coarse_logits = self.pretrained_img_module(x, need_feat=False)
        else:
            coarse_logits = logit_map.detach()

        meshes, vertices_by_cls, faces_by_cls, neighbors, normals = self.volume2mesh(
            coarse_logits
        )
        if not meshes:
            if istrain:
                return {
                    "logit_map": logit_map,
                    "final_meshes": {},
                    "final_vertices": {},
                    "final_faces": {},
                    "relative_positions": {},
                }
            return logit_map

        points_feat = self.point_module(list(vertices_by_cls.values()), normals)
        previous_vertices = {
            cls: vertices.clone() for cls, vertices in vertices_by_cls.items()
        }
        current_meshes = meshes
        current_vertices = vertices_by_cls
        current_faces = faces_by_cls
        current_neighbors = neighbors

        for iteration_idx in range(self.evolution_iters):
            current_vertices_before_update = {
                cls: vertices.clone() for cls, vertices in current_vertices.items()
            }
            d_delta, points_feat = self.predict_vertex_offsets(
                img_feat,
                points_feat,
                previous_vertices,
                current_vertices,
                current_neighbors,
                iteration_idx,
            )
            need_amto = self.should_apply_topology(
                iteration_idx, current_meshes, current_vertices
            )
            (
                current_meshes,
                current_vertices,
                current_faces,
                current_neighbors,
            ) = self.update_meshes(logit_map, d_delta, current_meshes, need_amto)
            previous_vertices = current_vertices_before_update

        relative_positions = {
            cls: self.sample_relative_position(logit_map, vertices, cls)
            for cls, vertices in current_vertices.items()
        }
        train_meshes = self.build_pytorch3d_meshes(current_vertices, current_faces, device)

        if istrain:
            return {
                "logit_map": logit_map,
                "final_meshes": train_meshes,
                "final_vertices": current_vertices,
                "final_faces": current_faces,
                "relative_positions": relative_positions,
            }
        return self.rasterize_prediction(current_vertices, current_faces, device)
"""


class Paper3_model(nn.Module):
    def __init__(
        self,
        model_name,
        in_channels,
        base_channels,
        num_classes,
        roi_size,
        logger,
        dataset_name="BTCV",
        evolution_iters=None,
        topology_interval=None,
        enable_topology=True,
    ):
        super().__init__()
        self.dataset_name = normalize_dataset_name(dataset_name)
        dataset_hparams = get_dataset_hparams(self.dataset_name)
        shared_hparams = get_shared_hparams()
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.logger = logger
        self.evolution_iters = (
            shared_hparams["evolution_iters"]
            if evolution_iters is None
            else evolution_iters
        )
        self.topology_interval = (
            shared_hparams["topology_interval"]
            if topology_interval is None
            else topology_interval
        )
        self.enable_topology = enable_topology
        self.topology_threshold = dataset_hparams["topology_threshold"]
        self.use_pretrained_init = False

        self.img_module = get_model(
            model_name=model_name,
            in_channels=in_channels,
            out_channels=num_classes,
            roi_size=roi_size,
            feature_channels=base_channels,
        )
        self.pretrained_img_module = copy.deepcopy(self.img_module)
        for param in self.pretrained_img_module.parameters():
            param.requires_grad = False

        self.point_module = get_point_model(
            num_classes=num_classes, inchannels=base_channels
        )
        self.relative_position_head = nn.Sequential(
            nn.Linear(base_channels, base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_channels // 2, 1),
            nn.Sigmoid(),
        )
        self.img_point_fuse = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
        )
        self.f2v = Feature2DeltaLayer(base_channels)

    def set_pretrained_init(self, enabled: bool = True):
        self.use_pretrained_init = enabled

    def normalize_vertices(self, vertices: torch.Tensor):
        vertices = vertices.clone()
        vertices[..., 0] = vertices[..., 0] / max(self.roi_size[-3] - 1, 1)
        vertices[..., 1] = vertices[..., 1] / max(self.roi_size[-2] - 1, 1)
        vertices[..., 2] = vertices[..., 2] / max(self.roi_size[-1] - 1, 1)
        return vertices

    def compute_vertex_normals(self, vertices: torch.Tensor, faces: torch.Tensor):
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
        return F.normalize(vertex_normals, dim=-1, eps=1e-6)

    def build_surface_mesh(self, vertices: torch.Tensor, faces: torch.Tensor):
        return trimesh.Trimesh(
            vertices=vertices.detach().cpu().numpy(),
            faces=faces.detach().cpu().numpy(),
            process=False,
        )

    def get_neighbors(self, mesh):
        average_length = 5
        adjusted_neighbors = []
        for vertex_idx, neighbor in enumerate(mesh.vertex_neighbors):
            neighbor = list(neighbor)
            if not neighbor:
                neighbor = [vertex_idx]
            if len(neighbor) >= average_length:
                adjusted_neighbors.append(neighbor[:average_length])
            else:
                repeats = average_length // len(neighbor) + 1
                adjusted_neighbors.append((neighbor * repeats)[:average_length])
        return torch.as_tensor(adjusted_neighbors, dtype=torch.long)

    def extract_mesh(self, pred_volume: torch.Tensor, cls: int):
        try:
            mesh = get_each_mesh(pred_volume, cls)
        except Exception:
            return None
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            return None
        return mesh

    def volume2mesh(self, coarse_logits: torch.Tensor):
        device = coarse_logits.device
        pred_volume = torch.argmax(coarse_logits, dim=1)[0].detach().cpu()

        meshes = {}
        vertices_by_cls = {}
        faces_by_cls = {}
        neighbors = []
        normals = []
        init_index = 0

        for cls in torch.unique(pred_volume):
            cls = int(cls.item())
            if cls == 0:
                continue

            mesh = self.extract_mesh(pred_volume, cls)
            if mesh is None:
                continue

            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
            vertices = self.normalize_vertices(vertices)
            normals_cls = self.compute_vertex_normals(vertices, faces)
            surface_mesh = self.build_surface_mesh(vertices, faces)

            meshes[cls] = surface_mesh
            vertices_by_cls[cls] = vertices
            faces_by_cls[cls] = faces
            neighbors.append(self.get_neighbors(surface_mesh).to(device) + init_index)
            normals.append(normals_cls)
            init_index += vertices.shape[0]

        if neighbors:
            neighbors = torch.cat(neighbors, dim=0)
            normals = torch.cat(normals, dim=0)
        else:
            neighbors = torch.empty((0, 5), dtype=torch.long, device=device)
            normals = torch.empty((0, 3), dtype=torch.float32, device=device)

        return meshes, vertices_by_cls, faces_by_cls, neighbors, normals

    def split_by_class(self, tensor: torch.Tensor, vertices_by_cls):
        outputs = {}
        start_idx = 0
        for cls, cls_vertices in vertices_by_cls.items():
            cls_num_vertices = cls_vertices.shape[0]
            outputs[cls] = tensor[:, start_idx : start_idx + cls_num_vertices]
            start_idx += cls_num_vertices
        return outputs

    def sample_image_features(self, img_feat: torch.Tensor, vertices_by_cls):
        if not vertices_by_cls:
            return torch.empty((img_feat.shape[0], 0, img_feat.shape[1]), device=img_feat.device)

        vertices = torch.cat(list(vertices_by_cls.values()), dim=0)
        grid = 2 * vertices[None, :, None, None] - 1
        sampled = F.grid_sample(
            img_feat,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )[:, :, :, 0, 0]
        return sampled.permute(0, 2, 1)

    def predict_relative_positions(self, img_vertex_feat: torch.Tensor, vertices_by_cls):
        relative_positions = self.relative_position_head(img_vertex_feat).squeeze(-1)
        relative_positions = relative_positions.clamp(1e-4, 1.0 - 1e-4)
        return {
            cls: values[0, :, 0]
            for cls, values in self.split_by_class(relative_positions[:, :, None], vertices_by_cls).items()
        }

    def get_dist(self, source_vertices, target_vertices):
        if not source_vertices or not target_vertices:
            device = source_vertices[0].device if source_vertices else target_vertices[0].device
            return torch.empty((0, 0), device=device)

        init_source = 0
        init_target = 0
        source_length = sum(item.shape[0] for item in source_vertices)
        target_length = sum(item.shape[0] for item in target_vertices)
        dists = torch.full(
            (target_length, source_length),
            float("inf"),
            device=source_vertices[0].device,
        )
        for cls_source_vertices, cls_target_vertices in zip(source_vertices, target_vertices):
            sqrdists = square_distance(cls_target_vertices[None], cls_source_vertices[None])[0]
            dists[
                init_target : init_target + cls_target_vertices.shape[0],
                init_source : init_source + cls_source_vertices.shape[0],
            ] = sqrdists
            init_source += cls_source_vertices.shape[0]
            init_target += cls_target_vertices.shape[0]
        return dists

    def update_points_feat(self, points_feat, source_vertices, target_vertices):
        dists = self.get_dist(source_vertices, target_vertices)
        if dists.numel() == 0:
            return points_feat

        k = min(3, dists.shape[-1])
        dists, idx = torch.topk(dists, k, largest=False, sorted=False)
        weights = torch.softmax(-dists, dim=-1)
        return torch.sum(
            index_points(points_feat, idx[None]) * weights.view(1, dists.shape[0], k, 1),
            dim=2,
        )

    def predict_vertex_offsets(
        self,
        img_feat,
        points_feat,
        previous_vertices,
        current_vertices,
        current_neighbors,
        step_idx,
    ):
        current_list = list(current_vertices.values())
        if step_idx != 0:
            previous_list = list(previous_vertices.values())
            points_feat = self.update_points_feat(points_feat, previous_list, current_list)

        img_vertex_feat = self.sample_image_features(img_feat, current_vertices)
        relative_positions = self.predict_relative_positions(img_vertex_feat, current_vertices)
        fused_feat = torch.cat([img_vertex_feat, points_feat], dim=-1)
        fused_feat = self.img_point_fuse(fused_feat)
        d_delta = self.f2v(fused_feat, current_neighbors)
        return d_delta, points_feat, relative_positions

    def mesh_complexity(self, vertices: torch.Tensor, mesh) -> float:
        laplacian_terms = []
        for vertex_idx, neighbor in enumerate(mesh.vertex_neighbors):
            neighbor = list(neighbor)
            if not neighbor:
                continue
            neighbor_tensor = vertices[torch.as_tensor(neighbor, device=vertices.device)]
            laplacian_terms.append(
                torch.norm(vertices[vertex_idx] - neighbor_tensor.mean(dim=0), p=2)
            )
        if not laplacian_terms:
            return 0.0
        return torch.stack(laplacian_terms).mean().item()

    def should_apply_topology(self, iteration_idx: int, meshes, vertices_by_cls):
        if not self.enable_topology or pymesh is None:
            return False
        if (iteration_idx + 1) % self.topology_interval != 0:
            return False
        complexities = [
            self.mesh_complexity(vertices_by_cls[cls], mesh)
            for cls, mesh in meshes.items()
            if cls in vertices_by_cls
        ]
        if not complexities:
            return False
        return float(np.mean(complexities)) > self.topology_threshold

    def ATMO(self, vertices: torch.Tensor, faces: torch.Tensor):
        if pymesh is None:
            return None
        try:
            new_mesh = pymesh.form_mesh(
                vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()
            )
            new_mesh = fix_mesh(
                new_mesh,
                target_len=(
                    1.0 / self.roi_size[0],
                    1.0 / self.roi_size[1],
                    1.0 / self.roi_size[2],
                ),
            )
            refined_mesh = trimesh.Trimesh(
                vertices=new_mesh.vertices, faces=new_mesh.faces, process=False
            )
            smoothed_mesh = trimesh.smoothing.filter_taubin(
                refined_mesh, nu=0.5, lamb=0.5, iterations=5
            )
            return smoothed_mesh if smoothed_mesh is not None else refined_mesh
        except Exception:
            return None

    def update_meshes(self, d_delta, meshes, relative_positions_by_cls, need_amto):
        device = d_delta.device
        init_num = 0
        init_num_new = 0
        new_meshes = {}
        new_vertices_by_cls = {}
        new_faces_by_cls = {}
        new_neighbors = []

        for cls, mesh in meshes.items():
            vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32, device=device)
            faces = torch.as_tensor(mesh.faces, dtype=torch.long, device=device)
            normals = self.compute_vertex_normals(vertices, faces)
            num_vertices = vertices.shape[0]
            relative_position = relative_positions_by_cls[cls]
            direction = torch.where(
                relative_position[:, None] >= 0.5,
                torch.ones_like(vertices),
                -torch.ones_like(vertices),
            )
            updated_vertices = torch.clamp(
                vertices
                + d_delta[0, init_num : init_num + num_vertices] * direction * normals,
                0.0,
                1.0,
            )
            init_num += num_vertices

            refined_mesh = self.ATMO(updated_vertices, faces) if need_amto else None
            if (
                refined_mesh is None
                or len(refined_mesh.vertices) == 0
                or len(refined_mesh.faces) == 0
            ):
                refined_mesh = self.build_surface_mesh(updated_vertices, faces)

            refined_vertices = torch.as_tensor(
                refined_mesh.vertices, dtype=torch.float32, device=device
            )
            refined_faces = torch.as_tensor(
                refined_mesh.faces, dtype=torch.long, device=device
            )
            surface_mesh = self.build_surface_mesh(refined_vertices, refined_faces)

            new_meshes[cls] = surface_mesh
            new_vertices_by_cls[cls] = refined_vertices
            new_faces_by_cls[cls] = refined_faces
            new_neighbors.append(self.get_neighbors(surface_mesh).to(device) + init_num_new)
            init_num_new += refined_vertices.shape[0]

        if new_neighbors:
            new_neighbors = torch.cat(new_neighbors, dim=0)
        else:
            new_neighbors = torch.empty((0, 5), dtype=torch.long, device=device)

        return new_meshes, new_vertices_by_cls, new_faces_by_cls, new_neighbors

    def build_pytorch3d_meshes(self, vertices_by_cls, faces_by_cls, device):
        train_meshes = {}
        for cls, vertices in vertices_by_cls.items():
            train_meshes[cls] = Meshes(
                verts=[vertices],
                faces=[faces_by_cls[cls].to(device=device, dtype=torch.long)],
            )
        return train_meshes

    def rasterize_prediction(self, vertices_by_cls, faces_by_cls, device):
        pred_voxels = torch.zeros(
            (1, self.num_classes, self.roi_size[0], self.roi_size[1], self.roi_size[2]),
            dtype=torch.float32,
            device=device,
        )
        rasterizer = Rasterize(self.roi_size)
        for cls, vertices in vertices_by_cls.items():
            scaled_vertices = vertices.clone()
            scaled_vertices[..., 0] *= self.roi_size[-3] - 1
            scaled_vertices[..., 1] *= self.roi_size[-2] - 1
            scaled_vertices[..., 2] *= self.roi_size[-1] - 1
            pred_cls_voxels = rasterizer(
                scaled_vertices[None][..., [2, 1, 0]],
                faces_by_cls[cls][None],
            ).float()
            pred_voxels[0, cls][pred_cls_voxels[0] > 0] = 1.0

        pred_voxels[:, 0, ...] = (pred_voxels[:, 1:, ...].sum(dim=1) == 0).float()
        return pred_voxels

    def forward(self, x, istrain=False):
        device = x.device
        img_feat, logit_map = self.img_module(x)
        if self.use_pretrained_init:
            with torch.no_grad():
                coarse_logits = self.pretrained_img_module(x, need_feat=False)
        else:
            coarse_logits = logit_map.detach()

        meshes, vertices_by_cls, faces_by_cls, neighbors, normals = self.volume2mesh(
            coarse_logits
        )
        if not meshes:
            if istrain:
                return {
                    "logit_map": logit_map,
                    "final_meshes": {},
                    "final_vertices": {},
                    "final_faces": {},
                    "relative_positions": {},
                }
            return logit_map

        points_feat = self.point_module(list(vertices_by_cls.values()), normals)
        previous_vertices = {
            cls: vertices.clone() for cls, vertices in vertices_by_cls.items()
        }
        current_meshes = meshes
        current_vertices = vertices_by_cls
        current_faces = faces_by_cls
        current_neighbors = neighbors

        for iteration_idx in range(self.evolution_iters):
            current_vertices_before_update = {
                cls: vertices.clone() for cls, vertices in current_vertices.items()
            }
            d_delta, points_feat, relative_positions = self.predict_vertex_offsets(
                img_feat,
                points_feat,
                previous_vertices,
                current_vertices,
                current_neighbors,
                iteration_idx,
            )
            need_amto = self.should_apply_topology(
                iteration_idx, current_meshes, current_vertices
            )
            (
                current_meshes,
                current_vertices,
                current_faces,
                current_neighbors,
            ) = self.update_meshes(
                d_delta, current_meshes, relative_positions, need_amto
            )
            previous_vertices = current_vertices_before_update

        final_img_vertex_feat = self.sample_image_features(img_feat, current_vertices)
        final_relative_positions = self.predict_relative_positions(
            final_img_vertex_feat, current_vertices
        )
        train_meshes = self.build_pytorch3d_meshes(
            current_vertices, current_faces, device
        )

        if istrain:
            return {
                "logit_map": logit_map,
                "final_meshes": train_meshes,
                "final_vertices": current_vertices,
                "final_faces": current_faces,
                "relative_positions": final_relative_positions,
            }
        return self.rasterize_prediction(current_vertices, current_faces, device)
