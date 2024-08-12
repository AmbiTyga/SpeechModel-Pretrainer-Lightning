from typing import Any
from torch.optim.optimizer import Optimizer
import lightning as L
from transformers import (
    AdamW,
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    Wav2Vec2FeatureExtractor,
    get_scheduler
)
import torch
from .data import DataCollatorForWav2Vec2Pretraining, AudioDataset
from .utils import multiply_grads, gather_for_metrics
class Wav2Vec2PretrainingModule(L.LightningModule):
    def __init__(
            self, 
            model_name: str,
            datasets_path: List[str],
            learning_rate: float = 1e-3, 
            batch_size_per_device: int = 2,
            lr_warmup_steps: int = 10, 
            max_gumbel_temperature: float = 2.0,
            min_gumbel_temperature: float = 0.5,
            gumbel_temperature_decay: float = 0.999995
    ):
        super(Wav2Vec2PretrainingModule, self).__init__()

        ## Setup parameters required for training
        self.learning_rate = learning_rate
        self.batch_size = batch_size_per_device
        self.lr_warmup_steps = lr_warmup_steps
        self.model_name = model_name

        ## Setup dataset pretraining paramters
        config = Wav2Vec2Config.from_pretrained(model_name)
        self.mask_time_prob = config.mask_time_prob if config.mask_time_prob>0 else 0.65
        self.mask_time_length = config.mask_time_length if config.mask_time_length>0 else 10  

        ## After stepping update parameters
        self.max_gumbel_temperature = max_gumbel_temperature
        self.min_gumbel_temperature = min_gumbel_temperature
        self.gumbel_temperature_decay = gumbel_temperature_decay
        self.save_hyperparameters()

        ## Setup model for pretraining
        self.model = Wav2Vec2ForPreTraining(config)
        self.model.gradient_checkpointing_enable()
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        ## Initialize data collator for data loaders
        self.data_collator = DataCollatorForWav2Vec2Pretraining(
            model=self.model,
            feature_extractor=feature_extractor,
            mask_time_prob=self.mask_time_prob,
            mask_time_length=self.mask_time_length
        )

        ## Initialize datasets
        train_path, val_path = datasets_path
        self.trainset = AudioDataset(pd.read_csv(train_path, header=None)[0].tolist())
        self.valset = AudioDataset(pd.read_csv(val_path, header=None)[0].tolist())
    
    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=(self.batch_size%(os.cpu_count()+1))
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valset,
            batch_size=self.batch_size*2,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=(self.batch_size%(os.cpu_count()+1))//2
        )
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        """
        num_losses = batch['mask_time_indices'].sum()
        sub_attention_mask = batch.pop("sub_attention_mask", None)
        sub_attention_mask = (
            sub_attention_mask if sub_attention_mask is not None else torch.ones_like(batch["masked_time_indices"])
        )

        outputs = self.model(**batch)
    
        if self.trainer.num_nodes>1:
            num_losses = gather_for_metrics(num_losses).sum()
            gradient_multiplier = self.trainer.num_nodes / num_losses
            multiply_grads(self.model.module.parameters(), gradient_multiplier)
        else:
            multiply_grads(self.model.parameters(), 1.0 / num_losses)
        return outputs
    
    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        # For logging
        loss_log = outputs['loss'].detach()
        contrastive_loss_log = outputs['contrastive_loss'].detach()
        diversity_loss_log = outputs['diversity_loss'].detach()

        if self.trainer.world_size > 1:
            loss_log = gather_for_metrics(loss_log)
            contrastive_loss_log = gather_for_metrics(contrastive_loss_log)
            diversity_loss_log = gather_for_metrics(diversity_loss_log)
        self.log('loss', loss_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('contrastive_loss', contrastive_loss_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('diversity_loss', diversity_loss_log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    
    def validation_step(self, batch, batch_idx):
        batch.pop("sub_attention_mask", None)
        outputs = self.model(**batch)
        loss = outputs.loss
        contrastive_loss = outputs.contrastive_loss
        diversity_loss = outputs.diversity_loss
        if self.trainer.world_size > 1:
            loss = gather_for_metrics(loss)
            contrastive_loss = gather_for_metrics(contrastive_loss)
            diversity_loss = gather_for_metrics(diversity_loss)

        num_losses = batch['mask_time_indices'].sum()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_contrastive_loss", contrastive_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_diversity_loss", diversity_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_num_losses", num_losses, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = AdamW(
            list(self.model.parameters()),
            lr = self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01
        )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer= optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=self.lr_warmup_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        gumbel_temperature = max(
                self.max_gumbel_temperature * self.gumbel_temperature_decay**(self.global_step%(self.batch_size*self.trainer.num_nodes)),
                self.min_gumbel_temperature,
            )
        if hasattr(self.model, "module"):
            self.model.module.set_gumbel_temperature(gumbel_temperature)
        else:
            self.model.set_gumbel_temperature(gumbel_temperature)