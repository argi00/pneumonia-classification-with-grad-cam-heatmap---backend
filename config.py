from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Optional, Sequence
import torch
# Load environment variables from .env file if it exists
load_dotenv()

@dataclass
class Args():

    # wandb configuration
    project_name = "mini_capston-xray"
    log_weight: bool = False



    # Data Augmentation and Transformation
    size:int = 224
    rotation:int = 10
    mean:Sequence[float] = (0.485, 0.456, 0.406)
    std:Sequence[float] = (0.229, 0.224, 0.225)

    # Augmentation based on the mode
    train_augmentation:Sequence[str] = ("Resize","Horizontal","Rotation","Tensor","Normalize")
    val_augmentation:Sequence[str] = ("Resize","Tensor")


    # Dataloader config
    batch:int = 32


    # Encoder name
    encoder_name:str = "densenet121" # efficientnetb2


    #loss configuration
    pos_weight=1.0


    # Train configuration
    seed:int = 23
    metric_to_monitor_early_stop = "val_f1"
    earlystop_mode:str = "max"
    save_weights_only: bool = False

    max_epochs:int = 100
    max_steps: int = -1
    precision: int = 16 
    pos_weight = 3.5
    patience: int = 5


    # --Device configuration
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    debug_mode: bool = False
    tag: Sequence[str] = ('V0')
    run_name: str = 'debug'
    num_workers =4
    lr = 0.001

    checkpoint_path: str = (
   "/teamspace/studios/this_studio/Xray Pneumonia classifier/reports/best/densenet121 - best_model.ckpt"
)