## Voice (STT/TTS)

Speech input and output for Jarvis, built on Moonshine. Currently two separate
pieces, not yet joined into a single voice loop.

### Status: 🟡 Built, not wired to orchestrator

- Speech-to-text works standalone and successfully triggers the Browser Agent directly
- Text-to-speech works standalone as a test script
- STT and TTS are not connected to each other — the assistant currently has
  no way to speak a response back after hearing a command
- Neither is wired into the planned Orchestrator yet

### How it works today

**Speech-to-text (`jv_stt.py`)**
Listens to the default microphone via `MicTranscriber`, transcribes speech in
real time, and on each completed line, calls the Browser Agent directly with
the transcribed text as its goal. Verified working end to end — voice input
successfully drives real browser actions. This bypasses the orchestrator
entirely, which is expected at this stage — it's a direct voice-to-agent path
for testing, not the final architecture.

**Text-to-speech (`jv_tts.py`)**
Standalone script that loads a Moonshine voice and speaks a fixed test
sentence. Confirmed working, but not yet connected to any live pipeline —
currently used to test voice quality and confirm the TTS setup, not to speak
real responses.

### Known Limitations

- TTS voice is still a placeholder (`kokoro_am_echo`) — not yet evaluated
  against the full voice list and confirmed as the final choice
- No feedback loop: the assistant can hear you, and separately can speak, but
  can't yet speak *in response to* what it heard
- Not wired to the Orchestrator — STT currently drives the Browser Agent
  directly, which will need to change once the Orchestrator exists

### Stack
Moonshine (`moonshine_voice`) — `MicTranscriber`, `TranscriptEventListener`,
`TextToSpeech`