from argparse import Namespace
config = Namespace(
    # data
    sample_size = 100_000,
    seq_len = 100, 
    n_ids = 100, # equal to seq_len, eg: 200 SNP sites
    n_classes = 13, # masked, AA, AT, AC, AG, CA, CT, GA, GC, TA, TC, TG, TT
    
    # model
    MAX_EPOCHS = 35,
    batch_size = 100,
    n_layers=8,
    n_heads=8,
    d_model=64,
    d_ffn=64*4,
    dropout=0.05,
    
    # training
    mask_frac=0.15,
    fix_lr = True,
    lr=1e-4,
    lr_warmup = 100, # steps
    lr_max_iters = 400, # steps
    ckpt_dir = './ckpt',
    ckpt_filename = 'min_val_loss',
    ckpt_resume = True,

    # wandb
    wandb = {
        'key': '2f76f9b7eb342042c286445e6a110e63aed1e352',
        'project': 'project_name',
        'run': 'run_name',
        'tags': ['tag1', 'tag2'],
    }
)
