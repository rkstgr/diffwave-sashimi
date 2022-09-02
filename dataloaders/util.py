from pathlib import Path
import torchaudio

class SingleSample:
    def __init__(self, file_path: str) -> None:
        self.file_path = Path(file_path)
        assert self.file_path.exists(), f"Give filepath does not exist: {self.file_path}"

        self.waveform, self.sample_rate = torchaudio.load(self.file_path)

    def __getitem__(self, _: int):
        return self.waveform, self.sample_rate, ""

    def __len__(self) -> int:
        return 100