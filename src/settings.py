# all the constants used across the project, so nothing is hardcoded in the pipeline files
from pathlib import Path

# the law this whole chatbot is scoped to
LAW_NAME = "The Contract Act, 1872"
LAW_YEAR = 1872

# where things get saved/loaded from
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "contract_act_1872.pdf"
