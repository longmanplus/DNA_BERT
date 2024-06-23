# %%
# ------ summary -----
# This script is to test the BERT model on a synthetic dataset.
# The synthetic dataset is generated by GenerateDataset class.
# The BERT model is defined in pl_model class.
# The training is done by Lightning Trainer.
# The results are plotted in the end.
# ---------------------

# ----- python module requirement -------
#torch
#lightning
#wandb
#matplotlib
# ---------------------------------------

# ----------------------------------------------------------------
#  load our functions for preparing data and model
# ----------------------------------------------------------------
# add path
import sys
sys.path.append('func')
from func.data_random import GeneratedDataModule
from func.model_bert import pl_model
from func.plots import plot_example_batch, plot_predictions, plot_attention_maps

def main(config):

    # -------Data--------
    dm = GeneratedDataModule(config)
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    # plot example batch: example_batch.png
    plot_example_batch(train_loader)

    # -------Model--------
    model = pl_model(config)

    # -------Train--------
    import torch
    torch.set_float32_matmul_precision('high') # 'medium' or 'high'
    from lightning.pytorch.loggers import WandbLogger
    logger = WandbLogger(project=config.wandb['project'], name=config.wandb['run'], config=config, tags=config.wandb['tags'], save_code=True, settings=wandb.Settings(code_dir=".")) 
    import os
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor 
    lr_monitor = LearningRateMonitor(logging_interval='step')
    import lightning as L

    trainer = L.Trainer(
        # strategy="ddp_find_unused_parameters_true",
        strategy = 'ddp',
        accelerator='gpu',
        devices='auto', 
        # devices=[0,1], 
        num_nodes=config.nnodes,
        max_epochs=config.MAX_EPOCHS, 
        logger=logger, 
        # log_every_n_steps=10,
        callbacks=[
            ModelCheckpoint(monitor='val/loss', mode='min', save_top_k=2, dirpath=config.ckpt_dir, filename=config.ckpt_filename, enable_version_counter=False, verbose=True),
            lr_monitor],
    )

    # if ckpt file exists, resume training
    ckpt_path = config.ckpt_dir + '/' + config.ckpt_filename + '.ckpt'
    if config.ckpt_resume and os.path.exists(ckpt_path):
        print(f'Resuming training from {ckpt_path}')
        trainer.fit(model, dm, ckpt_path=ckpt_path)
    else:
        trainer.fit(model, dm)

    # -------Plot results--------
    one_batch = next(iter(train_loader))
    id, input, target = one_batch
    pred, attn = model.forward_w_attn(id, input)

    input = input.detach().cpu().numpy()
    pred = pred.argmax(dim=2).detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    # heatmap of predictions: predictions.png
    plot_predictions(input, target, pred)

    # plot attention maps: attn_maps.png, avg_attn_map.png
    avg_attn = plot_attention_maps(id, attn, idx=0)
    print(f'avg_attn.shape: {avg_attn.shape}')


if __name__ == "__main__":

    # ------CONFIG------
    from config import config
    #-------------------

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--nnodes", default=1)
    cmd_args = parser.parse_args()
    config.nnodes = int(cmd_args.nnodes)

    import wandb
    wandb.login(key=config.wandb['key'])

    main(config)
