#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from livekit import api

from whisperopenai import WhisperOpenaiSTTService
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.services.livekit import LiveKitParams, LiveKitTransport
from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from openai.types.chat import ChatCompletionToolParam


load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")


async def start_fetch_weather(function_name, llm, context):
    # note: we can't push a frame to the LLM here. the bot
    # can interrupt itself and/or cause audio overlapping glitches.
    # possible question for Aleix and Chad about what the right way
    # to trigger speech is, now, with the new queues/async/sync refactors.
    # await llm.push_frame(TextFrame("Let me check on that."))
    logger.debug(f"Starting fetch_weather_from_api with function_name: {function_name}")

async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    await result_callback({"conditions": "nice", "temperature": "75"})


def generate_token(room_name: str, participant_name: str, api_key: str, api_secret: str) -> str:
    token = api.AccessToken(api_key, api_secret)
    token.with_identity(participant_name).with_name(participant_name).with_grants(
        api.VideoGrants(
            room_join=True,
            room=room_name,
        )
    )

    return token.to_jwt()


async def configure_livekit():
    parser = argparse.ArgumentParser(description="LiveKit AI SDK Bot Sample")
    parser.add_argument(
        "-r", "--room", type=str, required=False, help="Name of the LiveKit room to join"
    )
    parser.add_argument("-u", "--url", type=str, required=False, help="URL of the LiveKit server")

    args, unknown = parser.parse_known_args()

    room_name = args.room or os.getenv("LIVEKIT_ROOM_NAME")
    url = args.url or os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not room_name:
        raise Exception(
            "No LiveKit room specified. Use the -r/--room option from the command line, or set LIVEKIT_ROOM_NAME in your environment."
        )

    if not url:
        raise Exception(
            "No LiveKit server URL specified. Use the -u/--url option from the command line, or set LIVEKIT_URL in your environment."
        )

    if not api_key or not api_secret:
        raise Exception(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables."
        )

    token = generate_token(room_name, "Say One Thing", api_key, api_secret)

    user_token = generate_token(room_name, "User", api_key, api_secret)
    logger.info(f"User token: {user_token}")

    return (url, token, room_name)



async def fetch_weather_from_api(function_name, tool_call_id, args, llm, context, result_callback):
    temperature = 75 if args["format"] == "fahrenheit" else 24
    await result_callback(
        {
            "conditions": "nice",
            "temperature": temperature,
            "format": args["format"],
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
    )


tools = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    }
]


async def main():
    async with aiohttp.ClientSession() as session:
        (url, token, room_name) = await configure_livekit()

        transport = LiveKitTransport(
            url=url,
            token=token,
            room_name=room_name,
            params=LiveKitParams(
                audio_in_enabled=True,
                # audio_in_sample_rate=24000,
                audio_out_enabled=True,
                # audio_out_sample_rate=24000,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)),
            ),
        )

        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
        # Register a function_name of None to get all functions
        # sent to the same callback with an additional function_name parameter.
        llm.register_function(None, fetch_weather_from_api, start_callback=start_fetch_weather)

        tools = [
            ChatCompletionToolParam(
                type="function",
                function={
                    "name": "get_current_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use. Infer this from the users location.",
                            },
                        },
                        "required": ["location", "format"],
                    },
                },
            )
        ]

        messages = [
            {
                "role": "system",
                "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way.",
            },
        ]

        context = OpenAILLMContext(messages, tools)
        context_aggregator = llm.create_context_aggregator(context)


        stt = WhisperOpenaiSTTService()

        tts = OpenAITTSService(api_key=os.getenv("OPENAI_API_KEY"), voice="alloy")

        tl = TranscriptionLogger()


        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
                tl,
            ]
        )

        task = PipelineTask(
            pipeline,
            PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            await task.queue_frames([context_aggregator.user().get_context_frame()])
        
        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())