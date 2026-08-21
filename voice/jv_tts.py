"""Text-to-speech test for Jarvis AI Assistant using Moonshine"""

from moonshine_voice import TextToSpeech, list_tts_voices

tts = TextToSpeech().language("en_us")

# List available preset voices before picking one
# print("Available voices:", list_tts_voices("en_us"))

for voice in list_tts_voices("en_us")["downloadable"]:
    print(f"Voice: {voice}")

tts = tts.voice("kokoro_am_echo")  # placeholder - replace with a real voice from the list above
tts.load()

tts.say("Hello, I am Jarvis, your AI assistant. How can I help you today?")
tts.wait()