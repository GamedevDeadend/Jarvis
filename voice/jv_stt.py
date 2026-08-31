"""Transcribes live audio from the default microphone for Jarvis AI Assitant using Moonshine"""""

import time
from moonshine_voice import MicTranscriber, TranscriptEventListener
from browser.jv_browser_agent_graph import run_browser_agent


class MicListener(TranscriptEventListener):

    def on_line_completed(self, event):
        print(f"Line completed: {event.line.text}")
        run_browser_agent(event.line.text)



# MicTranscriber handles connecting to the microphone, capturing the audio
# data, detecting voice activity, breaking the speech up into segments,
# transcribing it, and calling you back as the results firm up over time.

mic = MicTranscriber()
mic.add_listener(MicListener())
mic.load()
mic.start()

print("Listening to the microphone, press Ctrl+C to stop...")
while True:
    time.sleep(0.1)
