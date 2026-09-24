"""Small deterministic demo CSVs (synthetic, Titanic-like) shipped with the engine.

Deliberate train/test inconsistencies: `Survived` (target) only in train,
`Embarked` = "Q" only in test, `Age` numeric in train but text in test
("unknown"), one duplicated row in train.
"""

from pathlib import Path

DEMO_DIR = Path(__file__).parent
TRAIN_CSV = str(DEMO_DIR / "train.csv")
TEST_CSV = str(DEMO_DIR / "test.csv")
