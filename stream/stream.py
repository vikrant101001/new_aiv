import asyncio
import websockets
from deepgream_text_to_speech import synthesize_audio, chunk_text_by_sentence
from open_ai_llm_response import ChatResponse
import time


async def echo(websocket, path):
    messages_received = []
    async for message in websocket:
        messages_received.append(message)
        if len(messages_received) == 1:
            # Record the start time
            start_time = time.time()
            compiled_text = " ".join(messages_received)

            chat_response=ChatResponse()
            _chat_response = chat_response.chat_response(compiled_text)

            # chunks = chunk_text_by_sentence(_chat_response)

            gen = _chat_response
            while True:
                try: 
                    sentences = next(gen)
                    print(sentences)
                    
                    print("sent")
                    audio = synthesize_audio(sentences)
                            # Record the end time
                    end_time = time.time()
                            # Calculate the elapsed time
                    elapsed_time = end_time - start_time

                    print(f"Elapsed time: {elapsed_time} seconds")
                    await websocket.send(audio)
                    messages_received = []  # Reset for next set of messages
                except StopIteration:
                    break


async def main():
    async with websockets.serve(echo, "localhost", 8765):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
