# coding=utf-8
"""
Hold a hands-free, bidirectional audio conversation with an existing Microsoft
Foundry voice agent. The agent, model, instructions, voice, and tools are already
configured in Foundry and are applied automatically when the session connects.
Microphone audio is streamed as PCM16 and reply audio is played through the
speakers.

Based on: https://github.com/microsoft/voice-agent/blob/main/samples/basic_voice_agent.py

Requires:
    pip install "azure-ai-projects[voice]>=2.7.0" "azure-identity>=1.17.0,<2.0.0" "pyaudio>=0.2.14"
  az login

Use a headset to reduce echo. Press Ctrl-C to stop.
"""

import asyncio
import base64
import queue
from typing import Any, Final, Optional

import aiohttp
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.aio.operations import AsyncRealtimeConnection 
from azure.ai.projects.models import RealtimeServerEventType    
from azure.core.pipeline.transport import AioHttpTransport
from azure.identity.aio import DefaultAzureCredential

SAMPLE_RATE: Final = 24000
CHUNK_SAMPLES: Final = 1200
PROJECT_ENDPOINT = ""
AGENT_NAME = "speech-to-text-agent"


def event_value(event: Any, name: str) -> Any:
    """Read a field from a typed Voice Agent event or an open mapping."""
    value = getattr(event, name, None)
    if value is None and hasattr(event, "get"):
        value = event.get(name)
    return value


def audio_bytes(delta: Any) -> bytes:
    """Normalize typed bytes and base64 audio deltas."""
    if isinstance(delta, bytes):
        return delta
    if isinstance(delta, str):
        return base64.b64decode(delta)
    raise RuntimeError("The service returned an unsupported audio delta.")


try:
    import pyaudio
except ImportError:  # pragma: no cover - required runtime dependency
    pyaudio = None  # type: ignore[assignment]


class AudioProcessor:
    """Capture microphone PCM and play response PCM with barge-in support."""

    def __init__(self, connection: AsyncBetaRealtimeConnection) -> None:
        self.connection = connection
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.audio = pyaudio.PyAudio()
        self.playback_queue: "queue.Queue[tuple[int, Optional[bytes]]]" = (
            queue.Queue()
        )
        self.playback_base = 0
        self.next_sequence = 0
        self.input_stream = None
        self.output_stream = None

    def next_sequence_number(self) -> int:
        sequence = self.next_sequence
        self.next_sequence += 1
        return sequence

    def start_capture(self) -> None:
        """Stream PCM frames through the Voice Agent SDK input buffer."""
        self.loop = asyncio.get_running_loop()

        def callback(in_data, _frame_count, _time_info, _status):
            assert self.loop is not None
            asyncio.run_coroutine_threadsafe(
                self.connection.input_audio_buffer.append(audio=in_data),
                self.loop,
            )
            return (None, pyaudio.paContinue)

        self.input_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SAMPLES,
            stream_callback=callback,
        )

    def start_playback(self) -> None:
        """Play queued agent audio without blocking the asyncio event loop."""
        remaining = b""

        def callback(_in_data, frame_count, _time_info, _status):
            nonlocal remaining
            wanted = frame_count * pyaudio.get_sample_size(pyaudio.paInt16)
            output = remaining[:wanted]
            remaining = remaining[wanted:]

            while len(output) < wanted:
                try:
                    sequence, data = self.playback_queue.get_nowait()
                except queue.Empty:
                    output += bytes(wanted - len(output))
                    continue
                if data is None:
                    break
                if sequence < self.playback_base:
                    remaining = b""
                    continue
                take = wanted - len(output)
                output += data[:take]
                remaining = data[take:]

            return (output, pyaudio.paContinue)

        self.output_stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK_SAMPLES,
            stream_callback=callback,
        )

    def queue_audio(self, pcm: bytes) -> None:
        self.playback_queue.put((self.next_sequence_number(), pcm))

    def skip_pending_audio(self) -> None:
        """Drop queued output immediately when the user barges in."""
        self.playback_base = self.next_sequence_number()

    def shutdown(self) -> None:
        if self.input_stream is not None:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream is not None:
            self.skip_pending_audio()
            self.playback_queue.put((self.next_sequence_number(), None))
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.audio.terminate()


async def run_microphone_session(
    client: AIProjectClient,
) -> Optional[str]:
    """Run a live conversation with the configured voice agent."""
    if pyaudio is None:
        raise RuntimeError("Install pyaudio to use the microphone sample.")

    conversation_id: Optional[str] = None
    async with client.beta.voice_agents.realtime.connect(
        agent_name=AGENT_NAME
    ) as connection:
        audio = AudioProcessor(connection)
        try:
            audio.start_playback()
            audio.start_capture()
            print(f"Connected to existing voice agent: {AGENT_NAME}")
            print("Speak now. Talk over the response to interrupt it. Press Ctrl-C to stop.")

            async for event in connection:
                event_type = event_value(event, "type")
                if event_type == RealtimeServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                    audio.skip_pending_audio()
                    print("(listening...)")
                elif (
                    event_type
                    == RealtimeServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
                ):
                    print(f"You:   {event_value(event, 'transcript')}")
                elif event_type == RealtimeServerEventType.RESPONSE_OUTPUT_AUDIO_DELTA:
                    audio.queue_audio(audio_bytes(event_value(event, "delta")))
                elif (
                    event_type
                    == RealtimeServerEventType.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE
                ):
                    transcript = (
                        event_value(event, "transcript")
                        or event_value(event, "text")
                        or ""
                    )
                    print(f"Agent: {transcript}")
                elif event_type == RealtimeServerEventType.SESSION_CREATED:
                    conversation_id = event_value(event, "conversation_id")
                elif event_type == RealtimeServerEventType.ERROR:
                    error = event_value(event, "error")
                    print(
                        "Session error: "
                        f"{event_value(error, 'message') or 'unknown error'}"
                    )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nEnding session...")
        finally:
            audio.shutdown()

    return conversation_id


async def lifecycle() -> None:
    credential = DefaultAzureCredential()
    transport = AioHttpTransport(
        session=aiohttp.ClientSession(
            auto_decompress=False,
            headers={"Accept-Encoding": "gzip, deflate"},
        )
    )

    async with credential, AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=credential,
        allow_preview=True,
        transport=transport,
    ) as client:
        conversation_id = await run_microphone_session(client)
        if conversation_id:
            print(f"Conversation id: {conversation_id}")
        else:
            print("No persisted conversation id was returned.")


if __name__ == "__main__":
    try:
        asyncio.run(lifecycle())
    except KeyboardInterrupt:
        print("\nInterrupted.")
