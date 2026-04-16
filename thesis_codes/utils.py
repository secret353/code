import numpy as np
import nibabel as nib
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import grey_erosion, grey_dilation

import torch

# from PIL import Image
# import io
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from monai.transforms import LoadImage


# def load_local_pretrained(model, pretrained_model, logger, strict=False):

#     logger.info("=> loading pretrain...")
#     ckpt = torch.load(pretrained_model)
#     if "model" in ckpt:
#         state_dict = ckpt["model"]
#     else:
#         state_dict = ckpt
#     if strict:
#         model_state_dict = model.state_dict()
#         model_state_dict.update(state_dict)
#         model.load_state_dict(state_dict, strict=True)
#     else:
#         for k in list(state_dict.keys()):
#             if (model.state_dict().get(k) is None) or model.state_dict()[
#                 k
#             ].shape != state_dict[k].shape:
#                 state_dict.pop(k)
#         model_state_dict = model.state_dict()
#         model_state_dict.update(state_dict)
#         model.load_state_dict(state_dict, strict=False)
#     logger.info(f"==>  loaded pretrained weights '{pretrained_model}'  successfully")


def _select_compatible_state_dict(model_state_dict, state_dict):
    candidates = [state_dict]
    known_prefixes = ("module.", "backbone.", "module.backbone.")

    for prefix in known_prefixes:
        stripped = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if stripped:
            candidates.append(stripped)

    def overlap_score(candidate):
        score = 0
        for key, value in candidate.items():
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                score += 1
        return score

    return max(candidates, key=overlap_score)


def load_pretrained(model, pretrained_model, logger, strict=False):

    logger.info("=> loading pretrain...")
    ckpt = torch.load(pretrained_model, map_location="cpu")
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    state_dict = _select_compatible_state_dict(model.state_dict(), state_dict)
    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        for k in list(state_dict.keys()):
            if (model.state_dict().get(k) is None) or model.state_dict()[
                k
            ].shape != state_dict[k].shape:
                state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    logger.info(f"==>  loaded pretrained weights '{pretrained_model}'  successfully")


def get_each_mesh(image, cls: int):
    """
    Returns a mesh for each class.
    image: npy image
    """
    cls_image = image == cls
    cls_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(cls_image.cpu(), pitch=1.0)
    # 获取最大的mesh，避免出现多个离散孤立mesh
    cls_meshes = cls_mesh.split(only_watertight=False)
    if not cls_meshes:
        return None
    cls_mesh = max(cls_meshes, key=lambda m: m.volume)

    # # 创建一个3D绘图
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # # 获取mesh的顶点和面
    # vertices = cls_mesh.vertices
    # faces = cls_mesh.faces
    # # 绘制mesh
    # mesh_collection = Poly3DCollection(vertices[faces], alpha=0.25)  # 可以设置透明度
    # ax.add_collection3d(mesh_collection)
    # # 设置视图的缩放比例是相等的
    # ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    # # 保存图像
    # plt.savefig("./mesh_visualization.png")
    # # 关闭绘图窗口
    # plt.close(fig)

    # show mesh
    # cls_mesh.show()
    return cls_mesh


def erosion_dilation(
    volume, kernel_size=3, erosion_iterations=5, dilation_iterations=5
):
    # 对所有前景标签进行腐蚀操作
    eroded = np.zeros_like(volume)
    for label in np.unique(volume):
        if label == 0:
            continue
        mask = volume == label
        for _ in range(erosion_iterations):
            mask = grey_erosion(mask, size=kernel_size)
        eroded[mask] = label
    # 对所有前景标签进行膨胀操作
    dilated = np.zeros_like(eroded)
    for label in np.unique(eroded):
        if label == 0:
            continue
        mask = eroded == label
        for _ in range(dilation_iterations):
            mask = grey_dilation(mask, size=kernel_size)
        dilated[mask] = label

    return dilated


def get_all_meshes(image_path, use_dilation=True):
    """
    Returns a list of meshes for all classes.
    image_path: npy image path
    """
    # image = np.load(image_path)
    # image = LoadImage()(image_path)[0].numpy()
    image = nib.load(image_path).get_fdata()
    if use_dilation:
        image = erosion_dilation(
            image, kernel_size=3, erosion_iterations=0, dilation_iterations=1
        )
    all_cls = np.unique(image)
    meshes = []
    for cls in all_cls[1:]:
        meshes.append(get_each_mesh(image, cls))
    return meshes, all_cls[1:]


def show_all_meshes(meshes, colors=None):
    o3d_meshes = []
    if colors is None:
        num_meshes = len(meshes)
        colors = np.random.uniform(0, 1, size=(num_meshes, 3))
    for i, mesh in enumerate(meshes):
        if mesh is None:
            continue
        cls_mesh = o3d.geometry.TriangleMesh()
        cls_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        cls_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
        cls_mesh.compute_vertex_normals()

        color = colors[i]
        cls_mesh.paint_uniform_color(color)  # Red

        o3d_meshes.append(cls_mesh)

    o3d.visualization.draw_geometries(
        o3d_meshes,
        mesh_show_back_face=True,
        point_show_normal=True,
        mesh_show_wireframe=True,
    )


def crop_each_mesh(pred_mesh, min_v: "(x, y, z)", max_v):
    """
    Crop the mesh to the cropped image range according to the min_v and max_v.
    """
    vertices = pred_mesh.vertices
    faces = pred_mesh.faces

    mask = np.all((min_v < vertices) & (vertices < max_v), axis=1)
    # 获取所所有Ture的索引
    v_indices = np.nonzero(mask)[0]
    if v_indices.shape[0] == 0:
        return None, vertices

    # 获取所有包含v_indices的面
    faces_mask = np.isin(faces, v_indices)
    faces_mask = np.all(faces_mask, axis=1)
    crop_faces = faces[faces_mask]
    # 更新顶点索引
    new_v_indices = np.unique(crop_faces)
    # crop_vertices = vertices[new_v_indices] - min_v
    crop_vertices = vertices[new_v_indices]
    # get the rest of the vertices
    mask_vertices = np.ones(len(vertices), dtype=bool)
    mask_vertices[new_v_indices] = False
    remaining_vertices = vertices[mask_vertices]

    assert crop_vertices.min() >= 0, "mask_vertices min is smaller than 0"
    # 更新mask
    new_mask = np.zeros_like(mask)
    new_mask[new_v_indices] = True
    # 由于vertices的删除，整体的索引会发生变化，需要重新计算
    indi = np.cumsum(new_mask, axis=0) - 1
    crop_faces = indi[crop_faces]

    pred_mesh = trimesh.Trimesh(vertices=crop_vertices, faces=crop_faces)
    return pred_mesh, remaining_vertices


def crop_image_mesh(img_path, label_path):
    image = np.load(img_path)
    # show_image(image)
    # part 选择ROI区域
    roi_center = np.array(
        [image.shape[0] // 2 - 5, image.shape[1] // 2 - 5, image.shape[2] // 2]
    )
    min_v = roi_center - np.array([48, 48, 48])
    max_v = roi_center + np.array([48, 48, 48])
    # part 裁剪图像
    cropped_image = image[min_v[0] : max_v[0], min_v[1] : max_v[1], min_v[2] : max_v[2]]
    # show_image(cropped_image)
    # part 裁剪mesh
    all_meshes, cls_list = get_all_meshes(label_path)
    cropped_meshes = []
    all_remaining_vertiices = []
    for mesh in all_meshes:
        cropped_mesh, remaining_vertices = crop_each_mesh(mesh, min_v, max_v)
        # if cropped_mesh is not None:
        cropped_meshes.append(cropped_mesh)
        all_remaining_vertiices.append(remaining_vertices)
    # show_all_meshes(cropped_meshes)
    return cropped_image, cropped_meshes, all_remaining_vertiices


def show_all_vertices(
    all_meshes, remaining_vertices=None, sample=False, use_color=True
):
    """
    Show all vertices of all meshes.
    :param all_meshes:
    :param sample: 是否对每个mesh顶点进行统一采样
    :param remaining_vertices: 未被采样的顶点
    :param use_color: 是否显示彩色
    :return:
    """
    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111, projection="3d")
    colors = [
        np.array([128, 174, 128]) / 255,
        np.array([216, 101, 79]) / 255,
        np.array([241, 214, 145]) / 255,
        np.array([255, 220, 70]) / 255,
        np.array([183, 156, 220]) / 255,
        np.array([111, 184, 210]) / 255,
        np.array([183, 214, 211]) / 255,
        np.array([68, 172, 100]) / 255,
        np.array([111, 197, 131]) / 255,
    ]

    # new_colors = []
    for i, mesh in enumerate(all_meshes):
        if mesh is None:
            if remaining_vertices is not None and len(remaining_vertices[i]) > 0:
                vertices = remaining_vertices[i]
            else:
                continue
        else:

            vertices = mesh.vertices.reshape(-1, 3)
            if remaining_vertices is not None and remaining_vertices[i] is not None:
                vertices = np.concatenate([vertices, remaining_vertices[i]])

        if use_color:
            color = colors[i]
            s = 2
        else:
            # color = np.random.uniform(0, 1, size=3)
            value = np.random.uniform(0, 1)  # 在0和1之间随机选取一个值
            color = np.full(3, value)
            s = 4
        if sample:
            # 采样为相同的顶点数
            points_index = np.arange(0, vertices.shape[0])
            if len(points_index) >= 2048:
                selected_point_idxs = np.random.choice(
                    points_index, 2048, replace=False
                )
            else:
                selected_point_idxs = np.random.choice(points_index, 2048, replace=True)
            # vertices = vertices[torch.from_numpy(selected_point_idxs)]  # (2048, 3)
            vertices = vertices[selected_point_idxs]

        ax1.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], c=color, marker="*", s=s
        )
    # 添加颜色条
    fig.colorbar(ax1.collections[0], ax=ax1, fraction=0.01)
    # 隐藏坐标轴TICK标签
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.axis("off")
    plt.show(block=True)


def show_image(img, imgname):
    # 平均从第三维度上选择6个切片，并使用plt进行可视化
    for i in range(6):
        # plt.subplot(2, 3, i + 1)
        plt.imshow(img[:, :, i * 5 + 30], cmap="gray")
        plt.axis("off")
        plt.savefig(
            f"../thesis/c2/{imgname}{i}.png", bbox_inches="tight", pad_inches=0, dpi=300
        )
        plt.show()


# for HaN2015
colors = [
    np.array([128, 174, 128]) / 255,
    np.array([216, 101, 79]) / 255,
    np.array([241, 214, 145]) / 255,
    np.array([255, 220, 70]) / 255,
    np.array([183, 156, 220]) / 255,
    np.array([111, 184, 210]) / 255,
    np.array([183, 214, 211]) / 255,
    np.array([68, 172, 100]) / 255,
    np.array([111, 197, 131]) / 255,
]

if __name__ == "__main__":
    image_path = (
        r"C:\Users\chang\Documents\paper4\experiments\HaN\segresnet\ratio1\image4.npy"
    )
    pred_path = (
        r"C:\Users\chang\Documents\paper4\experiments\HaN\segresnet\ratio1\val4.npy"
    )
    # part 完整输入图像
    # image = np.load(image_path)
    # show_image(image, imgname="image")
    # part 裁剪图像
    cropped_image, cropped_meshes, remaining_vertices = crop_image_mesh(
        image_path, pred_path
    )
    # show_image(cropped_image, imgname="cropped_image")
    # part 完整mesh
    meshes = get_all_meshes(pred_path)
    # show_all_meshes(meshes, colors)
    # part 裁剪mesh
    # show_all_meshes(cropped_meshes, colors)
    # part 完整点云
    # show_all_vertices(meshes, sample=False)
    # show_all_vertices(meshes, sample=True)
    # show_all_vertices(meshes, sample=True, use_color=False)
    # show_all_vertices(meshes, sample=False, use_color=False)
    # show_all_vertices(all_meshes=[None] * len(meshes), remaining_vertices=remaining_vertices, sample=False)

    get_each_mesh(
        r"C:\Users\chang\Documents\paper4\experiments\HaN\segresnet\ratio1\val4.npy", 1
    )
