#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio

from enum import Enum
from typing import AsyncGenerator

import numpy as np

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
# from pipecat.services.ai_services import SegmentedSTTService
from ai_services import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from loguru import logger




try:
    from openai import OpenAI,AsyncOpenAI
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use OpenAIWhisper, you need to `pip install OpenAI`.")
    raise Exception(f"Missing module: {e}")


class Model(Enum):
    """Class of basic Whisper model selection options"""

    whisper = "whisper-1"


class WhisperOpenaiSTTService(SegmentedSTTService):
    """Class to transcribe audio with a openai maintain Whisper model"""

    def __init__(
        self,
        *,
        model: str | Model = Model.whisper,
        # device: str = "auto",
        # compute_type: str = "default",
        # no_speech_prob: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self._device: str = device
        # self._compute_type = compute_type
        self.set_model_name(model if isinstance(model, str) else model.value)
        # self._no_speech_prob = no_speech_prob
        self._model: AsyncOpenAI | None = None
        self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        logger.debug("Connection Openai Whisper model...")
        self._model = AsyncOpenAI()
        logger.debug("Connection successfully Whisper model")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using Whisper"""
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Divide by 32768 because we have signed 16-bit data.
        # audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        transcript = await self._model.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav",audio)
        )

        text = transcript.text

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()
        
        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601())
