#!/usr/bin/env python3
"""
analyze.py — Post-session analysis pipeline.

Transcribes meeting audio with speaker diarization, aligns with HR data,
then calls Claude for stress pattern analysis and psychoanalytic insights.

Usage:
    python analyze.py meeting_20240101_120000.wav hr_20240101_120000.csv [options]

Required env vars:
    ANTHROPIC_API_KEY    — https://console.anthropic.com

Optional env vars:
    HF_TOKEN             — HuggingFace token for pyannote speaker diarization.
                           Without this, all speech is labeled "Speaker A".
                           To enable: accept pyannote/speaker-diarization-3.1 at
                           huggingface.co/pyannote/speaker-diarization-3.1, then
                           set HF_TOKEN to your HuggingFace access token.

Transcription runs fully locally via OpenAI Whisper (no API key needed).
"""

import os
import sys
import csv
import argparse
from pathlib import Path

import whisper


# ── Data loading ──────────────────────────────────────────────────────────────

def load_hr_data(csv_path: str) -> list[dict]:
    data = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({'timestamp': float(row['timestamp']), 'bpm': int(row['bpm'])})
    return sorted(data, key=lambda x: x['timestamp'])


# ── Transcription ─────────────────────────────────────────────────────────────

def _assign_speaker(start: float, end: float, speaker_segments: list) -> str | None:
    """Return the speaker with the most overlap in the [start, end] window (seconds)."""
    overlap: dict[str, float] = {}
    for seg_start, seg_end, speaker in speaker_segments:
        o = min(end, seg_end) - max(start, seg_start)
        if o > 0:
            overlap[speaker] = overlap.get(speaker, 0) + o
    return max(overlap, key=overlap.get) if overlap else None


def transcribe(audio_path: str, num_speakers: int | None = None,
               model_size: str = "base") -> list[dict]:
    """Transcribe audio locally with Whisper; optionally diarize with pyannote."""
    print(f"Loading Whisper model '{model_size}'...")
    model = whisper.load_model(model_size)

    print(f"Transcribing {audio_path}...")
    result = model.transcribe(audio_path, verbose=False)
    segments = result["segments"]  # [{start, end, text}, ...]

    # Attempt speaker diarization via pyannote if HF_TOKEN is available
    speaker_segments: list | None = None
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        try:
            from pyannote.audio import Pipeline
            print("Running speaker diarization with pyannote...")
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            kwargs = {"num_speakers": num_speakers} if num_speakers else {}
            diarization = pipeline(audio_path, **kwargs)
            speaker_segments = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]
            print(f"Diarization complete: {len(set(s[2] for s in speaker_segments))} speakers found")
        except ImportError:
            print("pyannote.audio not installed — skipping diarization (pip install pyannote.audio)")
        except Exception as e:
            print(f"Diarization failed: {e} — falling back to single speaker")
    else:
        print("HF_TOKEN not set — skipping speaker diarization (all speech labeled 'Speaker A')")

    utterances = []
    for seg in segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)

        if speaker_segments:
            raw = _assign_speaker(seg["start"], seg["end"], speaker_segments)
            speaker = f"Speaker {raw}" if raw else "Speaker A"
        else:
            speaker = "Speaker A"

        utterances.append({
            "speaker": speaker,
            "text": seg["text"].strip(),
            "start_ms": start_ms,
            "end_ms": end_ms,
        })

    speaker_count = len(set(u["speaker"] for u in utterances))
    print(f"Got {len(utterances)} segments across {speaker_count} speaker(s)")
    return utterances


# ── HR alignment ──────────────────────────────────────────────────────────────

def compute_baseline(hr_data: list[dict]) -> float:
    """Session-wide average BPM, used as the stress baseline."""
    bpms = [r['bpm'] for r in hr_data]
    return sum(bpms) / len(bpms) if bpms else 70.0


def get_hr_stats(hr_data: list[dict], start_unix: float, end_unix: float) -> dict | None:
    """HR stats for readings that fall within [start_unix, end_unix]."""
    readings = [r['bpm'] for r in hr_data if start_unix <= r['timestamp'] <= end_unix]
    if not readings:
        return None
    return {
        'avg_bpm': round(sum(readings) / len(readings), 1),
        'max_bpm': max(readings),
        'min_bpm': min(readings),
        'n_readings': len(readings),
    }


def align_hr_with_utterances(
    utterances: list[dict],
    hr_data: list[dict],
    session_start: float,
) -> list[dict]:
    """Attach HR stats to each utterance based on overlapping timestamps."""
    annotated = []
    for u in utterances:
        start_unix = session_start + u['start_ms'] / 1000.0
        end_unix   = session_start + u['end_ms']   / 1000.0
        hr = get_hr_stats(hr_data, start_unix, end_unix)
        annotated.append({**u, 'hr': hr})
    return annotated


# ── Formatting ────────────────────────────────────────────────────────────────

def format_timeline(annotated: list[dict], baseline: float) -> str:
    """Human-readable annotated transcript for Claude's context window."""
    lines = []
    for u in annotated:
        start_s = u['start_ms'] // 1000
        m, s = divmod(start_s, 60)
        time_str = f"{m:02d}:{s:02d}"

        hr_str = ""
        if u['hr']:
            avg = u['hr']['avg_bpm']
            delta = avg - baseline
            if delta > 10:
                tag = " ⚠ [HR SPIKE]"
            elif delta > 5:
                tag = " △ [HR ELEVATED]"
            elif delta < -5:
                tag = " ↓ [HR DROP]"
            else:
                tag = ""
            hr_str = f"  |  HR {avg} BPM ({delta:+.0f}){tag}"

        lines.append(f"[{time_str}] {u['speaker']}: {u['text']}{hr_str}")

    return "\n".join(lines)


def format_speaker_summary(annotated: list[dict], baseline: float) -> str:
    """Per-speaker HR statistics."""
    speakers: dict[str, list[float]] = {}
    for u in annotated:
        if u['hr']:
            speakers.setdefault(u['speaker'], []).append(u['hr']['avg_bpm'])

    lines = []
    for speaker, bpms in sorted(speakers.items()):
        avg = sum(bpms) / len(bpms)
        delta = avg - baseline
        lines.append(f"  {speaker}: avg HR {avg:.1f} BPM ({delta:+.1f} vs baseline), "
                     f"across {len(bpms)} utterances")
    return "\n".join(lines) if lines else "  No per-speaker HR data available."


# ── Claude analysis ───────────────────────────────────────────────────────────

def build_prompt(
    timeline: str,
    speaker_summary: str,
    baseline: float,
    hr_data: list[dict],
) -> str:
    bpms = [r['bpm'] for r in hr_data]
    overall = {
        'baseline': round(baseline, 1),
        'max': max(bpms),
        'min': min(bpms),
        'n': len(bpms),
    }

    return f"""I've been wearing a continuous heart rate monitor during a work meeting. \
I want an honest, deep analysis of my stress patterns and what the data reveals about me. \
Don't be gentle — be direct and insightful.

## Session HR Overview
- Baseline HR (session avg): {overall['baseline']} BPM
- Range: {overall['min']}–{overall['max']} BPM
- Total HR readings: {overall['n']}

## Per-Speaker HR Stats (my HR while each person was talking)
{speaker_summary}

## Annotated Meeting Transcript
Legend: ⚠ [HR SPIKE] = >10 BPM above baseline | △ [HR ELEVATED] = 5–10 above | ↓ [HR DROP] = >5 below

{timeline}

---

Please provide a thorough analysis covering:

### 1. Stress Trigger Map
Which specific moments, topics, or speakers caused the strongest physiological stress \
responses? Quote the actual utterances where HR spiked. Rank the top triggers.

### 2. Speaker Dynamics
Did my HR respond differently to different speakers? What does this suggest about the \
nature of those relationships — power dynamics, trust, conflict, anxiety around approval?

### 3. Topic Analysis
What themes or subjects elevated my HR? Are there patterns — e.g., always stressed by \
accountability questions, or specific project areas?

### 4. Emotional Arc
Describe my emotional journey through the meeting. Did I start anxious and calm down, \
or escalate? Were there moments of genuine engagement vs. withdrawal?

### 5. Psychoanalytic Interpretation
Based on *all* of the above, what patterns do you see in my psychology? Think about: \
anxiety style (anticipatory vs reactive), conflict avoidance, need for approval, \
self-assertion patterns, impostor syndrome markers, or anything else the data suggests.

### 6. Blind Spots to Watch
What might I be unaware of that this data reveals? Things I might rationalize away \
but the physiology doesn't lie about.

### 7. Concrete Recommendations
Specific, actionable strategies for the identified triggers — both in-the-moment \
techniques and longer-term changes. Prioritize the highest-impact ones.

Be specific. Reference timestamps and quotes. Be psychologically honest even if uncomfortable."""


def analyze_with_claude(
    timeline: str,
    speaker_summary: str,
    baseline: float,
    hr_data: list[dict],
) -> str:
    import anthropic
    prompt = build_prompt(timeline, speaker_summary, baseline, hr_data)
    print("Sending data to Claude for analysis...")
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze meeting audio + HR data with Claude psychoanalysis"
    )
    parser.add_argument("audio",   help="Path to meeting WAV file")
    parser.add_argument("hr_csv",  help="Path to HR CSV file")
    parser.add_argument(
        "--speakers", type=int, default=None,
        help="Expected number of speakers (helps diarization accuracy)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output report path (default: analysis_<audio_stem>.md)"
    )
    parser.add_argument(
        "--session-start", type=float, default=None,
        help="Session start as Unix timestamp. Default: earliest timestamp in HR CSV."
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Larger = more accurate but slower."
    )
    parser.add_argument(
        "--prompt-only", action="store_true",
        help="Skip Claude API call. Save the analysis prompt to a .txt file instead."
    )
    args = parser.parse_args()

    # Validate env (only needed when actually calling the API)
    if not args.prompt_only and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: missing env var: ANTHROPIC_API_KEY (or use --prompt-only)", file=sys.stderr)
        sys.exit(1)

    # Load HR
    print("Loading HR data...")
    hr_data = load_hr_data(args.hr_csv)
    if not hr_data:
        print("Error: no HR data in CSV", file=sys.stderr)
        sys.exit(1)

    session_start = args.session_start or hr_data[0]['timestamp']
    baseline = compute_baseline(hr_data)
    print(f"Session start (unix): {session_start}")
    print(f"Baseline HR: {baseline:.1f} BPM  |  "
          f"Range: {min(r['bpm'] for r in hr_data)}–{max(r['bpm'] for r in hr_data)} BPM")

    # Transcribe
    utterances = transcribe(args.audio, args.speakers, args.model)

    # Align HR with transcript
    annotated = align_hr_with_utterances(utterances, hr_data, session_start)

    # Format for Claude
    timeline       = format_timeline(annotated, baseline)
    speaker_summary = format_speaker_summary(annotated, baseline)

    # Analyze (or save prompt for manual submission)
    if args.prompt_only:
        prompt = build_prompt(timeline, speaker_summary, baseline, hr_data)
        os.makedirs("prompts", exist_ok=True)
        prompt_path = f"prompts/prompt_{Path(args.audio).stem}.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        print(f"\nPrompt saved to: {prompt_path}")
        print("Paste the contents of that file into Claude to get your analysis.")
        return

    analysis = analyze_with_claude(timeline, speaker_summary, baseline, hr_data)

    # Save report
    os.makedirs("analysis", exist_ok=True)
    output_path = args.output or f"analysis/analysis_{Path(args.audio).stem}.md"
    with open(output_path, 'w') as f:
        f.write(f"# Meeting HR Analysis\n\n")
        f.write(f"**Audio:** `{args.audio}`  \n")
        f.write(f"**HR data:** `{args.hr_csv}`  \n")
        f.write(f"**Baseline HR:** {baseline:.1f} BPM  \n\n")
        f.write("---\n\n")
        f.write("## Annotated Timeline\n\n```\n")
        f.write(timeline)
        f.write("\n```\n\n---\n\n")
        f.write("## Claude's Analysis\n\n")
        f.write(analysis)
        f.write("\n")

    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 70 + "\n")
    print(analysis)


if __name__ == '__main__':
    main()
