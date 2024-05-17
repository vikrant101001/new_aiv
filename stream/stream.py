import asyncio
import websockets
from deepgream_text_to_speech import synthesize_audio, chunk_text_by_sentence
from open_ai_llm_response import ChatResponse
import time



async def echo(websocket, path):
    messages_received = []
    async for message in websocket:
        messages_received.append(message)
        if len(messages_received) == 3:
            # Record the start time
            start_time = time.time()
            compiled_text = " ".join(messages_received)
            chat_response=ChatResponse()
            _chat_response = chat_response.chat_response(compiled_text)

            chunks = chunk_text_by_sentence(_chat_response)
            # Synthesize each chunk into audio and play the audio
            for chunk_text in chunks:
                print("sent")
                audio = synthesize_audio(chunk_text)
                # Record the end time
                end_time = time.time()
                # Calculate the elapsed time
                elapsed_time = end_time - start_time

                print(f"Elapsed time: {elapsed_time} seconds")
                await websocket.send(audio)
            messages_received = []  # Reset for next set of messages


async def main():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
