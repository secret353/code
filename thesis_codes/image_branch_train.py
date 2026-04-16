import os

import matplotlib.pyplot as plt
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference

from paper_hyperparams import PAPER_SHARED_HPARAMS


class ImageBranchTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_function,
        train_loader,
        val_loader,
        scaler,
        logger,
        eval_num,
        max_epochs,
        dataset_name,
        model_name,
        post_label,
        post_pred,
        dice_metric,
        roi_size,
        scheduler=None,
        sliding_window_overlap=None,
        output_subdir="image_branch_pretrain",
    ):
        self.model = model
        self.backbone = getattr(model, "backbone", model)
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = scaler
        self.logger = logger
        self.eval_num = eval_num
        self.max_epochs = max_epochs
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.post_label = post_label
        self.post_pred = post_pred
        self.dice_metric = dice_metric
        self.roi_size = roi_size
        self.scheduler = scheduler
        self.sliding_window_overlap = (
            PAPER_SHARED_HPARAMS["sliding_window_overlap"]
            if sliding_window_overlap is None
            else sliding_window_overlap
        )
        self.output_dir = os.path.join(".", self.model_name, self.dataset_name, output_subdir)
        os.makedirs(self.output_dir, exist_ok=True)

        self.epoch_loss_values = []
        self.metric_values = []

    @property
    def device(self):
        return next(self.backbone.parameters()).device

    def predict_logits(self, x):
        return self.model(x, need_feat=False)

    def save_checkpoint(self, filename, metric_value):
        checkpoint = {
            "model": self.backbone.state_dict(),
            "best_metric": metric_value,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
        }
        torch.save(checkpoint, os.path.join(self.output_dir, filename))

    def save_full_image_branch(self, filename, metric_value):
        checkpoint = {
            "model": self.model.state_dict(),
            "best_metric": metric_value,
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
        }
        torch.save(checkpoint, os.path.join(self.output_dir, filename))

    def plot_history(self):
        plt.figure("image_branch_pretrain", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        plt.xlabel("Epoch")
        plt.plot([idx + 1 for idx in range(len(self.epoch_loss_values))], self.epoch_loss_values)

        plt.subplot(1, 2, 2)
        plt.title("Val Mean Dice")
        plt.xlabel("Epoch")
        plt.plot([self.eval_num * (idx + 1) for idx in range(len(self.metric_values))], self.metric_values)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close()

    def train(self):
        best_metric = 0.0
        best_epoch = 0

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = self.train_epoch(epoch)
            self.epoch_loss_values.append(epoch_loss)

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % self.eval_num == 0 or epoch == self.max_epochs:
                dice_val = self.validation()
                self.metric_values.append(dice_val)

                if dice_val > best_metric:
                    best_metric = dice_val
                    best_epoch = epoch
                    self.save_checkpoint("best_metric_model_backbone.pth", best_metric)
                    self.save_full_image_branch(
                        "best_metric_model_image_branch.pth", best_metric
                    )
                    self.logger.info(
                        "Image branch checkpoint saved. "
                        f"best_metric={best_metric:.4f}, epoch={best_epoch}"
                    )
                else:
                    self.logger.info(
                        "Image branch checkpoint not updated. "
                        f"best_metric={best_metric:.4f}, current_metric={dice_val:.4f}"
                    )
                self.plot_history()

        self.save_checkpoint("last_metric_model_backbone.pth", best_metric)
        self.save_full_image_branch("last_metric_model_image_branch.pth", best_metric)
        return best_metric, best_epoch

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_steps = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.predict_logits(inputs)
                loss = self.loss_function(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += float(loss.item())
            self.logger.info(
                "Image Pretrain %d: (%d / %d Steps) loss=%.4f"
                % (epoch, step, num_steps, float(loss.item()))
            )

        epoch_loss = running_loss / max(num_steps, 1)
        self.logger.info("Epoch %d average loss: %.4f" % (epoch, epoch_loss))
        return epoch_loss

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                val_inputs = batch["image"].to(self.device)
                val_labels = batch["label"].to(self.device)
                val_outputs = sliding_window_inference(
                    val_inputs,
                    self.roi_size,
                    sw_batch_size=1,
                    predictor=self.predict_logits,
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
                    f"Image Pretrain Validate {i + 1} / {len(self.val_loader)}: dice={current_mean:.4f}"
                )

            aggregate = self.dice_metric.aggregate()
            mean_dice_val = (
                float(aggregate.nanmean().item())
                if torch.is_tensor(aggregate)
                else float(aggregate)
            )
            self.dice_metric.reset()
        return mean_dice_val
