import os
import sys
import importlib

# ── Paramètres à modifier ────────────────────────────────────────────────────
INPUT_FILE  = "input/overlapping_sample_tevLouis_short.wav"
OUTPUT_DIR  = "output"
MODEL_NAME  = "SepReformer_Base_WSJ0"   
# "SepReformer_Large_DM_WSJ0" ou "SepReformer_Base_WSJ0"
# ─────────────────────────────────────────────────────────────────────────────

# S'assurer que les imports relatifs (utils, models) fonctionnent depuis la racine du projet
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

module_path = f"models.{MODEL_NAME}.main"
main_module = importlib.import_module(module_path)
MainProcessor = main_module.MainProcessor

MainProcessor.run(MODEL_NAME, "infer_sample", INPUT_FILE, OUTPUT_DIR)
