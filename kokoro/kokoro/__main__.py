"""Kokoro Vietnamese TTS CLI
Example usage:
python3 -m kokoro --config config.json --model model.pth --voice voice.pt -t "Xin chào Việt Nam" -o file.wav
"""

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
from loguru import logger


def generate_and_save_audio(
    output_file: Path, text: str, config: Path, model: Path, voice: Path, speed=1
) -> None:
    from kokoro import KModel, KPipeline

    kmodel = KModel(config=str(config), model=str(model)).eval()
    pipeline = KPipeline(lang_code="v", model=kmodel)
    with wave.open(str(output_file.resolve()), "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit audio)
        wav_file.setframerate(24000)  # Sample rate

        for result in pipeline(text, voice=str(voice), speed=speed, split_pattern=r"\n+"):
            logger.debug(result.phonemes)
            if result.audio is None:
                continue
            audio_bytes = (result.audio.numpy() * 32767).astype(np.int16).tobytes()
            wav_file.writeframes(audio_bytes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("--model", type=Path, required=True, help="Path to model weights (.pth)")
    parser.add_argument("-m", "--voice", type=Path, required=True, help="Path to voicepack (.pt)")
    parser.add_argument("-o", "--output-file", "--output_file", type=Path, required=True, help="Path to output WAV file")
    parser.add_argument("-i", "--input-file", "--input_file", type=Path, help="Path to input text file (default: stdin)")
    parser.add_argument("-t", "--text", help="Text to use instead of reading from stdin")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--debug", action="store_true", help="Print DEBUG messages to console")
    args = parser.parse_args()
    if args.debug:
        logger.level("DEBUG")
    logger.debug(args)

    if args.text is not None and args.input_file is not None:
        raise Exception("You cannot specify both 'text' and 'input_file'")
    elif args.text:
        text = args.text
    elif args.input_file:
        text = args.input_file.read_text(encoding="utf-8")
    else:
        print("Press Ctrl+D to stop reading input and start generating", flush=True)
        text = "\n".join(sys.stdin)

    logger.debug(f"Input text: {text!r}")

    if args.output_file.suffix != ".wav":
        logger.warning("The output file name should end with .wav")
    generate_and_save_audio(
        output_file=args.output_file,
        text=text,
        config=args.config,
        model=args.model,
        voice=args.voice,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
