import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from aizynthmodels.chemformer import Chemformer

# flake8: noqa: F401
from aizynthmodels.utils.configs.chemformer import pretrain
from aizynthmodels.utils.hydra import custom_config


@hydra.main(version_base=None, config_name="pretrain")
@custom_config
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    chemformer = Chemformer(config)
    chemformer.fit()


if __name__ == "__main__":
    main()
