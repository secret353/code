import torch
import os

# 导入F
import torch.nn.functional as F

from utils import get_each_mesh
from pytorch3d.structures import Meshes

# import SimpleITK as sitk

# import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import SaveImage
from paper_hyperparams import PAPER_SHARED_HPARAMS

# import trimesh

"""
class Trainer:
    def __init__(
        self,
        # pretrained_model,
        model,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        saler,
        logger,
        eval_num,
        max_epoches,
        dataset_name,
        model_name,
        post_label,
        post_pred,
        dice_metric,
        filename,
        roi_size,
        save_pred=False,
        save_dice_csv=False,
        num_classes=31,
    ):
        # self.pretrained_model = pretrained_model
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = saler
        self.logger = logger
        self.eval_num = eval_num
        self.max_epoches = max_epoches
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_dice_csv = save_dice_csv
        self.post_label = post_label
        self.post_pred = post_pred
        self.dice_metric = dice_metric
        self.save_pred = save_pred
        self.filename = filename
        self.num_classes = num_classes
        self.roi_size = roi_size

        self.epoch_loss_values = []
        self.metric_values = []

    # label to mesh
    def labelmesh(self, label):
        all_cls = torch.unique(label)
        meshes = {}
        for cls in all_cls[1:]:
            mesh = get_each_mesh(label.cpu(), cls)
            # mesh, _, _ = self.get_vertices_normals(mesh)
            meshes[cls] = Meshes(verts=[mesh.vertices], faces=[mesh.faces])

        return meshes

    # section epoch train
    def train(self, global_epoch, train_loader, dice_val_best, global_step_best):
        self.model.train()
        epoch_loss = 0
        step = 0
        for step, batch in enumerate(train_loader):
            step += 1

            # label_meshes = self.labelmesh(batch["label"])
            # meshes = self.volume2mesh(batch["image"])
            x, y = (batch["image"].cuda(), batch["label"].cuda())
            self.optimizer.zero_grad()
            # with torch.cuda.amp.autocast(): # 会导致mesh损失难以计算
            logit_map, iter_meshes, vertices = self.model(x, istrain=True)
            loss = self.loss_function((logit_map, iter_meshes, vertices), y)

            epoch_loss += loss.item()
            self.logger.info(
                f"Training {global_epoch}: ({step} / {len(train_loader)} Steps) (loss={loss:.3f})"
            )

            self.scaler.scale(loss).backward()
            # max_grad_norm = 1.0  # 设置梯度裁剪的阈值
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 打印模型参数的梯度值
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad, torch.norm(param.grad.detach(), 2))

        epoch_loss /= step
        self.epoch_loss_values.append(epoch_loss)
        self.logger.info(f"Epoch {global_epoch} average loss: {epoch_loss:.4f}")

        if (
            global_epoch % self.eval_num == 0 and global_epoch != 0
        ) or global_epoch == self.max_epoches:
            dice_val = self.validation(self.val_loader)
            self.metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        f"./{self.model_name}/{self.dataset_name}/",
                        f"best_metric_model_{dice_val}.pth",
                    ),
                )
                self.logger.info(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best: .3f} Current Avg. Dice: {dice_val: .3f}"
                )
            else:
                self.logger.info(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best: .3f} Current Avg. Dice: {dice_val: .3f}"
                )

            if not self.save_dice_csv or not self.save_pred:
                plt.figure("train", (12, 6))
                plt.subplot(1, 2, 1)
                plt.title("Iteration Average Loss")
                x = [i + 1 for i in range(len(self.epoch_loss_values))]
                y = self.epoch_loss_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.subplot(1, 2, 2)
                plt.title("Val Mean Dice")
                x = [self.eval_num * (i + 1) for i in range(len(self.metric_values))]
                y = self.metric_values
                plt.xlabel("Iteration")
                plt.plot(x, y)
                plt.savefig(f"./{self.model_name}/{self.dataset_name}/loss.png")

        global_epoch += 1
        return global_epoch, dice_val_best, global_step_best

    def validation(self, epoch_iterator_val):
        self.model.eval()
        with torch.no_grad():
            # dice_list = []
            if self.save_dice_csv:
                cols = ["cls" + str(i) for i in range(self.num_classes)]
                cols += ["mean_dice"]
                df = pd.DataFrame(columns=cols)
                # df = pd.read_excel(filename, sheet_name=f'{model_name}')

            for i, batch in enumerate(epoch_iterator_val):
                # if i == 5:
                #     print("break")
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                # plot x, y
                # plt.figure("train", (12, 6))
                # plt.subplot(1, 2, 1)
                # plt.title("image")
                # plt.imshow(val_inputs[0, 0, :, :, 90].detach().cpu().numpy(), cmap="gray")
                # plt.subplot(1, 2, 2)
                # plt.title("label")
                # plt.imshow(val_labels[0, 0, :, :, 90].detach().cpu().numpy())
                # plt.show()
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(
                        val_inputs, self.roi_size, 2, self.model, overlap=0
                    )
                    batch["pred"] = val_outputs
                val_labels_list = decollate_batch(val_labels)
                val_labels_convert = [
                    self.post_label(val_label_tensor)
                    for val_label_tensor in val_labels_list
                ]
                val_outputs_list = decollate_batch(val_outputs)
                val_output_convert = [
                    self.post_pred(val_pred_tensor)
                    for val_pred_tensor in val_outputs_list
                ]
                current_dice = self.dice_metric(
                    y_pred=val_output_convert, y=val_labels_convert
                )
                self.logger.info(
                    f"Validate {i + 1} / {len(epoch_iterator_val)}: dice={current_dice.nanmean().item():.4f}"
                )
                if self.save_dice_csv:
                    # 保存current_dice到名为cls_dice的csv文件中的cls0, cls1, cls2, ..., cls13列中
                    # current_dice的shape为(14)
                    current_dice = current_dice.cpu().numpy()
                    # 将numpy数据插入到对应的列中
                    df.loc[i, "cls0":f"cls{self.num_classes - 1}"] = current_dice
                    df.loc[i, "mean_dice"] = np.nanmean(current_dice)

                if self.save_pred:
                    if type(self.model).__name__ == "UNet":
                        # save img
                        # val_inputs_list = decollate_batch(val_inputs)
                        # SaveImage(output_dir=self.filename, output_postfix=f'few_shot_img{i}', separate_folder=False, resample=False)(
                        #     val_inputs_list[0],
                        # )
                        # # save label
                        # SaveImage(output_dir=self.filename, output_postfix=f'few_shot_label{i}', separate_folder=False, resample=False)(
                        #     val_labels_list[0],
                        # )
                        pass
                    # save pred
                    val_pred = torch.argmax(
                        val_outputs_list[0], dim=0, keepdim=True
                    )  # (1, h, w, d)
                    SaveImage(
                        output_dir=self.filename,
                        output_postfix=f"few_shot_pred{i}",
                        separate_folder=False,
                        resample=False,
                    )(
                        val_pred,
                    )

                # dice_list.append(current_dice[0, 9].item())
                # logger.info(f"Validate {i + 1} / {len(epoch_iterator_val)}: dice={current_dice[0, 9].item():.4f}")
            if self.save_dice_csv:
                # 将数据写入文件
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with pd.ExcelWriter(self.filename, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=f"{self.model_name}", index=False)

            mean_dice_val = self.dice_metric.aggregate().item()
            # mean_dice_val = sum(dice_list) / len(dice_list)
            self.dice_metric.reset()
        return mean_dice_val
"""


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        saler,
        logger,
        eval_num,
        max_epoches,
        dataset_name,
        model_name,
        post_label,
        post_pred,
        dice_metric,
        filename,
        roi_size,
        save_pred=False,
        save_dice_csv=False,
        num_classes=31,
        scheduler=None,
        sliding_window_overlap=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = saler
        self.logger = logger
        self.eval_num = eval_num
        self.max_epoches = max_epoches
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.save_dice_csv = save_dice_csv
        self.post_label = post_label
        self.post_pred = post_pred
        self.dice_metric = dice_metric
        self.save_pred = save_pred
        self.filename = filename
        self.num_classes = num_classes
        self.roi_size = roi_size
        self.scheduler = scheduler
        self.sliding_window_overlap = (
            PAPER_SHARED_HPARAMS["sliding_window_overlap"]
            if sliding_window_overlap is None
            else sliding_window_overlap
        )
        self.output_dir = os.path.join(".", self.model_name, self.dataset_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.epoch_loss_values = []
        self.metric_values = []

    @property
    def device(self):
        return next(self.model.parameters()).device

    def log_step_loss(self, global_epoch, step, num_steps, loss_dict):
        self.logger.info(
            "Training %d: (%d / %d Steps) "
            "(total=%.4f, spatial=%.4f, distance=%.4f, normal=%.4f, laplacian=%.4f)"
            % (
                global_epoch,
                step,
                num_steps,
                float(loss_dict["total"].item()),
                float(loss_dict["spatial"].item()),
                float(loss_dict["distance"].item()),
                float(loss_dict["normal"].item()),
                float(loss_dict["laplacian"].item()),
            )
        )

    def plot_history(self):
        if self.save_dice_csv or self.save_pred:
            return

        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        x_axis = [idx + 1 for idx in range(len(self.epoch_loss_values))]
        plt.xlabel("Epoch")
        plt.plot(x_axis, self.epoch_loss_values)

        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        x_axis = [self.eval_num * (idx + 1) for idx in range(len(self.metric_values))]
        plt.xlabel("Epoch")
        plt.plot(x_axis, self.metric_values)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close()

    def train(self, global_epoch, train_loader, dice_val_best, global_step_best):
        self.model.train()
        epoch_loss = 0.0
        component_sums = {
            "spatial": 0.0,
            "distance": 0.0,
            "normal": 0.0,
            "laplacian": 0.0,
        }
        num_steps = len(train_loader)

        for step, batch in enumerate(train_loader, start=1):
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x, istrain=True)
            loss_dict = self.loss_function(pred, y)
            loss = loss_dict["total"]

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_loss += float(loss.item())
            for key in component_sums:
                component_sums[key] += float(loss_dict[key].item())
            self.log_step_loss(global_epoch, step, num_steps, loss_dict)

        if self.scheduler is not None:
            self.scheduler.step()

        epoch_loss /= max(num_steps, 1)
        self.epoch_loss_values.append(epoch_loss)
        self.logger.info(
            "Epoch %d average loss: %.4f (spatial=%.4f, distance=%.4f, normal=%.4f, laplacian=%.4f)"
            % (
                global_epoch,
                epoch_loss,
                component_sums["spatial"] / max(num_steps, 1),
                component_sums["distance"] / max(num_steps, 1),
                component_sums["normal"] / max(num_steps, 1),
                component_sums["laplacian"] / max(num_steps, 1),
            )
        )

        if (
            global_epoch % self.eval_num == 0 and global_epoch != 0
        ) or global_epoch == self.max_epoches:
            dice_val = self.validation(self.val_loader)
            self.metric_values.append(dice_val)
            if dice_val > dice_val_best:
                dice_val_best = dice_val
                global_step_best = global_epoch
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.output_dir, "best_metric_model.pth"),
                )
                self.logger.info(
                    "Model Was Saved ! Current Best Avg. Dice: %.3f Current Avg. Dice: %.3f"
                    % (dice_val_best, dice_val)
                )
            else:
                self.logger.info(
                    "Model Was Not Saved ! Current Best Avg. Dice: %.3f Current Avg. Dice: %.3f"
                    % (dice_val_best, dice_val)
                )
            self.plot_history()

        global_epoch += 1
        return global_epoch, dice_val_best, global_step_best

    def validation(self, epoch_iterator_val):
        self.model.eval()
        with torch.no_grad():
            if self.save_dice_csv:
                columns = [f"cls{i}" for i in range(self.num_classes)] + ["mean_dice"]
                df = pd.DataFrame(columns=columns)

            for i, batch in enumerate(epoch_iterator_val):
                val_inputs = batch["image"].to(self.device)
                val_labels = batch["label"].to(self.device)
                val_outputs = sliding_window_inference(
                    val_inputs,
                    self.roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=self.sliding_window_overlap,
                )

                val_labels_list = decollate_batch(val_labels)
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_convert = [
                    self.post_label(label_tensor) for label_tensor in val_labels_list
                ]
                val_output_convert = [
                    self.post_pred(pred_tensor) for pred_tensor in val_outputs_list
                ]
                current_dice = self.dice_metric(
                    y_pred=val_output_convert, y=val_labels_convert
                )
                current_mean = current_dice.nanmean().item()
                self.logger.info(
                    f"Validate {i + 1} / {len(epoch_iterator_val)}: dice={current_mean:.4f}"
                )

                if self.save_dice_csv:
                    current_dice_np = current_dice.detach().cpu().numpy().reshape(-1)
                    df.loc[i, [f"cls{j}" for j in range(self.num_classes)]] = (
                        current_dice_np[: self.num_classes]
                    )
                    df.loc[i, "mean_dice"] = np.nanmean(current_dice_np)

                if self.save_pred:
                    val_pred = torch.argmax(val_outputs_list[0], dim=0, keepdim=True)
                    SaveImage(
                        output_dir=self.filename,
                        output_postfix=f"few_shot_pred{i}",
                        separate_folder=False,
                        resample=False,
                    )(val_pred)

            if self.save_dice_csv:
                os.makedirs(os.path.dirname(self.filename), exist_ok=True)
                with pd.ExcelWriter(self.filename, engine="openpyxl") as writer:
                    df.to_excel(writer, sheet_name=self.model_name, index=False)

            aggregate = self.dice_metric.aggregate()
            if torch.is_tensor(aggregate):
                mean_dice_val = float(aggregate.nanmean().item())
            else:
                mean_dice_val = float(aggregate)
            self.dice_metric.reset()
        return mean_dice_val
