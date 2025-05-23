import torch

# Default model hyperparams
DEFAULT_MAX_SEQ_LEN = 512
DEFAULT_CHEM_TOKEN_START = 272
REGEX = r"\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9]"

USE_GPU = True
use_gpu = USE_GPU and torch.cuda.is_available()
