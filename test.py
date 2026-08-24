from pathlib import Path

import fire
from omegaconf import OmegaConf, DictConfig
import lightning as L

from lightning_data import TBPSDataModule
from lightning_models import LitTBPS
from lightning.pytorch import seed_everything


def resolve_tuple(*args):
    return tuple(args)


OmegaConf.register_new_resolver("tuple", resolve_tuple)
OmegaConf.register_new_resolver("eval", eval)


def load_test_loader(dataset_name: str, config: DictConfig):
    """
    Load the dataset from the configuration.

    Args:
        dataset_name (str): The name of the dataset.
        config (DictConfig): The configuration.
    Returns:
        test_loader (DataLoader): The test loader.
    """
    config.dataset.dataset_name = dataset_name
    dm = TBPSDataModule(config)
    dm.setup()
    test_loader = dm.test_dataloader()

    return test_loader


def run_test(ckpt_path: str | Path, dataset_name: str):
    model = LitTBPS.load_from_checkpoint(ckpt_path)
    config = model.hparams.config
    seed_everything(config.seed)
    test_loader = load_test_loader(dataset_name, config)

    trainer = L.Trainer(**config.trainer)
    trainer.test(model, test_loader)

def run_test_late(ckpt_path: str | Path, dataset_name: str):
    # this function was added because i had issues with the launching of lora using the first one
    # if this version doesn't work, please try run_test() instead.
    print("\n🚩 STEP 1 : script launching, LoRA preparation...")
    import torch
    from lightning.pytorch import seed_everything
    import lightning as L
    
    # 1. Open checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vrai_config = ckpt["hyper_parameters"]["config"]
    
    # 2. light up the model with the correct config
    model = LitTBPS(config=vrai_config, num_iters_per_epoch=239)
    
    if "lora" in vrai_config:
        print("🛠️ LoRA activation...")
        model.setup_lora(vrai_config["lora"])
    else:
        print("⚠️ no lora config in the checkpoint.")

    # 3. Load the state dict into the model
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print("🚩 STEP 2 : Model loaded successfully and LoRA is in place !")


    seed_everything(vrai_config["seed"])
    test_loader = load_test_loader(dataset_name, vrai_config)
    trainer = L.Trainer(**vrai_config["trainer"])
    
    print("\n🚩 STEP 3 : Beginning evaluation...")
    trainer.test(model, test_loader)

if __name__ == "__main__":
    fire.Fire(run_test_late)
