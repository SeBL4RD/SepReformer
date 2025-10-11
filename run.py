import os
from models.SepReformer_Base_WSJ0.main import MainProcessor

class SepForm:
    @staticmethod
    def separ_tracks(output_path, input_file):
        # Sauvegarder le répertoire actuel
        original_dir = os.getcwd()
        
        # Changer temporairement pour SepReformer
        os.chdir('/app/SepReformer')
        
        try:
            model_name = "SepReformer_Base_WSJ0"
            engine_mode = "infer_sample"
            MainProcessor.run(model_name, engine_mode, input_file, output_path)
        finally:
            # Revenir au répertoire original
            os.chdir(original_dir)
