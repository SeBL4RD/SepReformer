from SepReformer.models.SepReformer_Base_WSJ0.main import MainProcessor

class SepForm:
    @staticmethod
    def separ_tracks(output_path, input_file):
        model_name="SepReformer_Base_WSJ0"
        MainProcessor.run(model_name, "infer_sample", input_file, output_path)