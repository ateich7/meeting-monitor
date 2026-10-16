# Meeting Stress Monitor

A heart-rate sensor worn during meetings, lined up with the meeting transcript, then analyzed by an LLM to find which moments spiked my stress and why.

- **Hardware:** an ESP32 wired to a GY-MAX3010X pulse sensor, streaming heart rate over serial.
- **Record:** `./record` captures heart rate and meeting audio (mic and system output) on the same clock.
- **Transcribe:** locally with Whisper, so the audio never leaves the laptop. Speakers are split with pyannote when `HF_TOKEN` is set.
- **Analyze:** `analyze.py` marks every transcript line with my heart rate (spikes, elevations and drops against the meeting's baseline) and sends it to Claude for a structured analysis: ranked triggers, speaker dynamics, topic patterns and recommendations.

```
pip install -r requirements.txt
./record                                   # during the meeting
export ANTHROPIC_API_KEY=...
python analyze.py recordings/meeting_<ts>.wav recordings/hr_<ts>.csv
```

Recordings, transcripts and results stay local; they're in `.gitignore`.

Write-up: https://ariteicher.com/projects/meeting-stress-monitor
