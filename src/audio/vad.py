import torch

class SileroVAD:
    def __init__(self, sample_rate=16000, threshold=0.5, min_silence_duration_ms=1500):
        self.sample_rate = sample_rate
        
        # Carrega o modelo leve do Silero via TorchHub
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
        
        # O VADIterator gerencia o estado da fala (início/fim) em tempo real
        self.vad_iterator = VADIterator(
            self.model, 
            threshold=threshold, 
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=min_silence_duration_ms
        )

    def processar_chunk(self, chunk_numpy):
        """
        Recebe o chunk do sounddevice (numpy array) e converte para Tensor.
        Retorna um dicionário se detectar 'start' ou 'end' da fala, senão None.
        """
        # Converte o array do sounddevice (float32) para Tensor do PyTorch
        chunk_tensor = torch.from_numpy(chunk_numpy).squeeze()
        
        # O VADIterator retorna dicts como {'start': 1234} ou {'end': 5678}
        resultado = self.vad_iterator(chunk_tensor, return_seconds=True)
        return resultado

    def resetar(self):
        """Reseta o estado interno do iterator após cada frase finalizada."""
        self.vad_iterator.reset_states()