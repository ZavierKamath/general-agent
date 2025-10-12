import asyncio
import json
import os

import websockets
from dotenv import load_dotenv
import pyaudio

from tools.function_mapper import FUNCTION_MAP

load_dotenv()


def sts_connect():
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        raise Exception("DEEPGRAM_API_KEY not found")
    return websockets.connect(
        "wss://agent.deepgram.com/v1/agent/converse",
        subprotocols=["token", api_key]
    )


def load_config():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.json")
    prompt_path = os.path.join(base_dir, "prompts", "system_prompt.txt")
    greeting_path = os.path.join(base_dir, "prompts", "greeting.txt")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    with open(prompt_path, "r") as f:
        config["agent"]["think"]["prompt"] = f.read().strip()
    
    with open(greeting_path, "r") as f:
        config["agent"]["greeting"] = f.read().strip()
    
    return config


CHUNK_MS = 20
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = int(SAMPLE_RATE * CHUNK_MS / 1000)


async def audio_sender(ws, in_stream):
    loop = asyncio.get_event_loop()
    while True:
        data = await loop.run_in_executor(None, in_stream.read, FRAMES_PER_BUFFER)
        await ws.send(data)


def _create_function_call_response(func_id: str, func_name: str, result: dict):
    return {
        "type": "FunctionCallResponse",
        "id": func_id,
        "name": func_name,
        "content": json.dumps(result),
    }


async def _handle_function_call_request(decoded: dict, ws):
    try:
        for function_call in decoded.get("functions", []):
            func_name = function_call.get("name")
            func_id = function_call.get("id")
            try:
                arguments = json.loads(function_call.get("arguments", "{}"))
            except Exception:
                arguments = {}

            if func_name in FUNCTION_MAP:
                try:
                    print(f"Calling function: {func_name} with arguments: {arguments}")
                    result = FUNCTION_MAP[func_name](**arguments)
                except TypeError:
                    expr = arguments.get("expression") or arguments.get("query") or arguments.get("input")
                    result = FUNCTION_MAP[func_name](expr)
            else:
                result = {"error": f"Unknown function: {func_name}"}

            response = _create_function_call_response(func_id, func_name, result)
            await ws.send(json.dumps(response))
    except Exception as e:
        err = _create_function_call_response(
            func_id if "func_id" in locals() else "unknown",
            func_name if "func_name" in locals() else "unknown",
            {"error": f"Function call failed: {str(e)}"},
        )
        await ws.send(json.dumps(err))


async def audio_receiver(ws, out_stream):
    async for message in ws:
        if isinstance(message, (bytes, bytearray)):
            out_stream.write(message)
        else:
            try:
                decoded = json.loads(message)
                if decoded.get("type") == "FunctionCallRequest":
                    await _handle_function_call_request(decoded, ws)
            except Exception:
                pass


async def run_agent():
    pa = pyaudio.PyAudio()
    in_stream = None
    out_stream = None

    try:
        in_stream = pa.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )
        out_stream = pa.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=FRAMES_PER_BUFFER,
        )

        async with sts_connect() as ws:
            config_message = load_config()
            await ws.send(json.dumps(config_message))
            print("Connected to Deepgram agent. Speak into your microphone. Press Ctrl+C to exit.")

            sender_task = asyncio.create_task(audio_sender(ws, in_stream))
            receiver_task = asyncio.create_task(audio_receiver(ws, out_stream))

            await asyncio.gather(sender_task, receiver_task)

    finally:
        try:
            if in_stream is not None:
                in_stream.stop_stream()
                in_stream.close()
        except Exception:
            pass
        try:
            if out_stream is not None:
                out_stream.stop_stream()
                out_stream.close()
        except Exception:
            pass
        pa.terminate()


def main():
    try:
        asyncio.run(run_agent())
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()