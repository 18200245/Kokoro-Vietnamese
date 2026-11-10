from .model import KModel
from dataclasses import dataclass
from loguru import logger
from vig2p import VI_FIXUPS as VI_PHONEME_FIXUPS
from typing import Callable, Generator, List, Optional, Union
import re
import torch
import os

MAX_PHONEMES = 510
CHUNK_SIZE = 400


class KPipeline:
    '''
    Vietnamese KPipeline with 2 main responsibilities:
    1. Perform Vietnamese G2P (vig2p), mapping (and chunking) text -> phonemes
    2. Manage and store local voicepacks

    A KPipeline can reuse an existing KModel:
    1. On init: pipeline = KPipeline(model=model)
    2. On call: pipeline(text, voice, model=model)

    By default, KPipeline initializes its own KModel, which requires local
    weights. To suppress this, construct a "quiet" KPipeline with model=False.

    A "quiet" KPipeline yields (graphemes, phonemes, None) without generating
    any audio. A "loud" KPipeline with a model yields (graphemes, phonemes, audio).
    '''
    def __init__(
        self,
        lang_code: str = 'v',
        repo_id: Optional[str] = None,
        model: Union[KModel, bool] = True,
        device: Optional[str] = None
    ):
        """Initialize a KPipeline.

        Args:
            lang_code: Must be 'v' (or 'vi'); kept for backward compatibility
            repo_id: Optional repo identifier, kept for backward compatibility
            model: KModel instance, True to create new model, False for no model
            device: Override default device selection ('cuda', 'mps' or 'cpu', or None for auto)
        """
        if lang_code.lower() not in ('v', 'vi'):
            raise ValueError(f"Only Vietnamese is supported (lang_code='v'), got {lang_code!r}")
        self.lang_code = 'v'
        self.repo_id = repo_id
        self.model = None
        if isinstance(model, KModel):
            self.model = model
        elif model:
            if device == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if device == 'mps' and not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            if device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1':
                raise RuntimeError("MPS requested but fallback not enabled")
            if device is None:
                if torch.cuda.is_available():
                    device = 'cuda'
                elif os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' and torch.backends.mps.is_available():
                    device = 'mps'
                else:
                    device = 'cpu'
            self.model = KModel(repo_id=repo_id).to(device).eval()
        self.voices = {}
        from vig2p import VietnameseG2P
        self.g2p = VietnameseG2P()

    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if not voice.endswith('.pt'):
            raise ValueError(f"Voice must be a local .pt voicepack path, got {voice!r}")
        pack = torch.load(voice, weights_only=True)
        self.voices[voice] = pack
        return pack

    def load_voice(self, voice: Union[str, torch.FloatTensor], delimiter: str = ",") -> torch.FloatTensor:
        """Load a voicepack tensor or local .pt path; multiple paths (delimited) are averaged."""
        if isinstance(voice, torch.FloatTensor):
            return voice
        if voice in self.voices:
            return self.voices[voice]
        logger.debug(f"Loading voice: {voice}")
        packs = [self.load_single_voice(v) for v in voice.split(delimiter)]
        if len(packs) == 1:
            return packs[0]
        self.voices[voice] = torch.mean(torch.stack(packs), dim=0)
        return self.voices[voice]

    @staticmethod
    def infer(
        model: KModel,
        ps: str,
        pack: torch.FloatTensor,
        speed: Union[float, Callable[[int], float]] = 1
    ) -> KModel.Output:
        if callable(speed):
            speed = speed(len(ps))
        return model(ps, pack[len(ps)-1], speed, return_output=True)

    def generate_from_tokens(
        self,
        tokens: str,
        voice: str,
        speed: float = 1,
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        """Generate audio from a raw phoneme string.

        Raises:
            ValueError: If no voice is provided or the phoneme string exceeds model limits
        """
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline.generate_from_tokens(..., voice="path/to/voice.pt")')
        pack = self.load_voice(voice).to(model.device) if model else None
        if len(tokens) > MAX_PHONEMES:
            raise ValueError(f'Phoneme string too long: {len(tokens)} > {MAX_PHONEMES}')
        output = KPipeline.infer(model, tokens, pack, speed) if model else None
        yield self.Result(graphemes='', phonemes=tokens, output=output)

    @staticmethod
    def _chunk_text(graphemes: str) -> List[str]:
        """Split text into chunks of roughly CHUNK_SIZE characters on sentence boundaries."""
        chunks = []
        sentences = re.split(r'([.!?]+)', graphemes)
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        if not chunks:
            chunks = [graphemes[i:i+CHUNK_SIZE] for i in range(0, len(graphemes), CHUNK_SIZE)]
        return chunks

    @dataclass
    class Result:
        graphemes: str
        phonemes: str
        output: Optional[KModel.Output] = None
        text_index: Optional[int] = None

        @property
        def audio(self) -> Optional[torch.FloatTensor]:
            return None if self.output is None else self.output.audio

        @property
        def pred_dur(self) -> Optional[torch.LongTensor]:
            return None if self.output is None else self.output.pred_dur

        ### MARK: BEGIN BACKWARD COMPAT ###
        def __iter__(self):
            yield self.graphemes
            yield self.phonemes
            yield self.audio

        def __getitem__(self, index):
            return [self.graphemes, self.phonemes, self.audio][index]

        def __len__(self):
            return 3
        #### MARK: END BACKWARD COMPAT ####

    def __call__(
        self,
        text: Union[str, List[str]],
        voice: Optional[str] = None,
        speed: Union[float, Callable[[int], float]] = 1,
        split_pattern: Optional[str] = r'\n+',
        model: Optional[KModel] = None
    ) -> Generator['KPipeline.Result', None, None]:
        model = model or self.model
        if model and voice is None:
            raise ValueError('Specify a voice: pipeline(text="Xin chào", voice="path/to/voice.pt")')
        pack = self.load_voice(voice).to(model.device) if model else None

        if isinstance(text, str):
            text = re.split(split_pattern, text.strip()) if split_pattern else [text]

        for graphemes_index, graphemes in enumerate(text):
            if not graphemes.strip():
                continue
            for chunk in self._chunk_text(graphemes):
                if not chunk.strip():
                    continue
                ps = self.g2p.many([chunk])[0]
                if not ps:
                    continue
                if len(ps) > MAX_PHONEMES:
                    logger.warning(f'Truncating len(ps) == {len(ps)} > {MAX_PHONEMES}')
                    ps = ps[:MAX_PHONEMES]
                output = KPipeline.infer(model, ps, pack, speed) if model else None
                yield self.Result(graphemes=chunk, phonemes=ps, output=output, text_index=graphemes_index)
