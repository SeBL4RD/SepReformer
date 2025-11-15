import os
import importlib

class SepForm:
    @staticmethod
    def separ_tracks(output_path, input_file, model_name="SepReformer_Base_WSJ0"):
        # Sauvegarder le répertoire actuel
        original_dir = os.getcwd()
        
        # Changer temporairement pour SepReformer
        os.chdir('/app/SepReformer')
        
        try:
            # Import dynamique du bon MainProcessor
            module_path = f"models.{model_name}.main"
            main_module = importlib.import_module(module_path)
            MainProcessor = main_module.MainProcessor
            
            engine_mode = "infer_sample"
            MainProcessor.run(model_name, engine_mode, input_file, output_path)
        finally:
            # Revenir au répertoire original
            os.chdir(original_dir)