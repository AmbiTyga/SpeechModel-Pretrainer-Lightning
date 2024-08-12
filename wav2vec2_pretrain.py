import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from pretrainer.module import Wav2Vec2PretrainingModule
from pretrainer.utils import parse_args

if __name__ == "__main__":
    args = parse_args()
    
    model = Wav2Vec2PretrainingModule(
        model_name=args.model_name,
        datasets_path=[args.train_path, args.val_path],
        learning_rate=args.learning_rate,
        batch_size_per_device=args.batch_size,
        lr_warmup_steps=args.lr_warmup_steps,
        max_gumbel_temperature=args.max_gumbel_temperature,
        min_gumbel_temperature=args.min_gumbel_temperature,
        gumbel_temperature_decay=args.gumbel_temperature_decay
    )

    ckpting = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="{step:06}.ckpt",
        save_weights_only=args.save_weights_only,
        every_n_train_steps=args.save_every_n_steps,
        save_last=True
    )

    tensorboardlogger = TensorBoardLogger(
        save_dir=args.output_dir,
        name=f"{args.output_dir}-tb",
        version=len(os.listdir(args.output_dir))
    )

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_steps=args.training_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        logger=tensorboardlogger,
        callbacks=[ckpting]
    )
    trainer.fit(
        model,
        ckpt_path=args.resume_ckpt_path
    )

