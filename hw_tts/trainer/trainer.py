import torch
from torch.nn.utils import clip_grad_norm_

from tqdm import tqdm
import numpy as np
import PIL
from torchvision.transforms import ToTensor

from .base_trainer import BaseTrainer
from hw_tts.utils import inf_loop, MetricTracker
from hw_tts.logger.utils import plot_spectrogram_to_buf


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            lr_scheduler,
            config,
            device,
            dataloader,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler, config, device)
        self.config = config
        self.train_dataloader = dataloader
        self.skip_oom = skip_oom
        self.batches_in_dataloader = len(self.train_dataloader)
        self.minibatches_per_batch = self.config["trainer"]["minibatches_per_batch"]
        self.minibatches_in_dataloader = self.batches_in_dataloader * self.minibatches_per_batch
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = self.minibatches_in_dataloader
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.log_step = self.len_epoch // 5

        self.train_metrics = MetricTracker(
            "total_loss", "mel_loss", "duration_loss", 
            "pitch_loss", "pitch_mean_loss", "pitch_std_loss", 
            "energy_loss" , "grad norm"
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["text", "duration", "pitch", "pitch_mean", "pitch_std", "energy", "mel", "mel_pos", "src_pos"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.len_epoch)
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.batches_in_dataloader)
        ):
            for mini_batch_idx, mini_batch in enumerate(batch):
                try:
                    mini_batch = self.process_batch(
                        mini_batch,
                        is_train=True,
                        metrics=self.train_metrics
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics.update("grad norm", self.get_grad_norm())
                cur_batch_id = self.minibatches_per_batch * batch_idx + mini_batch_idx
                if cur_batch_id % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + cur_batch_id)
                    self.logger.debug(
                        "Train Epoch: {} {} total_loss: {:.6f}, mel_loss: {:.6f}, duration_loss: {:.6f},"
                        " pitch_loss: {:.6f}, pitch_mean_loss: {:.6f}, pitch_std_loss: {:.6f},"
                        " energy_loss: {:.6f}".format(
                            epoch, 
                            self._progress(cur_batch_id), 
                            mini_batch["total_loss"].item(), 
                            mini_batch["mel_loss"].item(), 
                            mini_batch["duration_loss"].item(),
                            mini_batch["pitch_loss"].item(), 
                            mini_batch["pitch_mean_loss"].item(), 
                            mini_batch["pitch_std_loss"].item(),
                            mini_batch["energy_loss"].item()
                        )
                    )
                    if self.lr_scheduler is not None:
                        self.writer.add_scalar(
                            "learning rate", self.lr_scheduler.get_last_lr()[0]
                        )
                    self._log_sample(**mini_batch)
                    self._log_scalars(self.train_metrics)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        if is_train:
            mel_predicted, duration_predicted, pitch_predicted, pitch_mean_predicted, pitch_std_predicted, energy_predicted = self.model(
                batch["text"], 
                batch["src_pos"], 
                mel_pos=batch["mel_pos"], 
                mel_max_length=batch["mel_max_len"], 
                target_duration=batch["duration"], 
                target_pitch=batch["pitch"],
                target_pitch_mean=batch["pitch_mean"],
                target_pitch_std=batch["pitch_std"],
                target_energy=batch["energy"])
        else:
            mel_predicted, duration_predicted, pitch_predicted, pitch_mean_predicted, pitch_std_predicted, energy_predicted = self.model(
                batch["text"], 
                batch["src_pos"])

        batch["mel_predicted"] = mel_predicted
        batch["duration_predicted"] = duration_predicted
        batch["pitch_predicted"] = pitch_predicted
        batch["pitch_mean_predicted"] = pitch_mean_predicted
        batch["pitch_std_predicted"] = pitch_std_predicted
        batch["energy_predicted"] = energy_predicted

        total_loss, mel_loss, duration_loss, pitch_loss, pitch_mean_loss, pitch_std_loss, energy_loss = self.criterion(**batch)
        batch["total_loss"] = total_loss
        batch['mel_loss'] = mel_loss
        batch['duration_loss'] = duration_loss
        batch['pitch_loss'] = pitch_loss
        batch['pitch_mean_loss'] = pitch_mean_loss
        batch['pitch_std_loss'] = pitch_std_loss
        batch['energy_loss'] = energy_loss

        if is_train:
            batch["total_loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("total_loss", total_loss.item())
        metrics.update("mel_loss", mel_loss.item())
        metrics.update("duration_loss", duration_loss.item())
        metrics.update("pitch_loss", pitch_loss.item())
        metrics.update("pitch_mean_loss", pitch_mean_loss.item())
        metrics.update("pitch_std_loss", pitch_std_loss.item())
        metrics.update("energy_loss", energy_loss.item())

        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        current = batch_idx
        total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_sample(self, mel, mel_predicted, **kwargs):
        ind = np.random.choice(mel.shape[0])
        spectrogram_target = mel[ind].cpu().detach().numpy().T
        spectrogram_predicted = mel_predicted[ind].cpu().detach().numpy().T
        image_sp_target = PIL.Image.open(plot_spectrogram_to_buf(spectrogram_target))
        image_sp_predicted = PIL.Image.open(plot_spectrogram_to_buf(spectrogram_predicted))
        self.writer.add_image("target spectrogram", ToTensor()(image_sp_target))
        self.writer.add_image("predicted spectrogram", ToTensor()(image_sp_predicted))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))

