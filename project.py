from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    BASE_PATH: Path = Path(__file__).parents[0]
    DATA_PATH: Path = BASE_PATH / 'data'
    CHECKPOINT_PATH: Path = BASE_PATH / 'checkpoints'

    def __post_init__(self):
        self.DATA_PATH.mkdir(exist_ok=True)
        self.CHECKPOINT_PATH.mkdir(exist_ok=True)



