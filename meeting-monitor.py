import os
import serial
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading
import time
import csv
from datetime import datetime

# Settings
SERIAL_PORT = '/dev/cu.usbserial-0001'
BAUD_RATE = 115200
SAMPLE_RATE = 44100
OUTPUT_DIR = 'recordings'

# Shared data
hr_data = []
audio_chunks = []
recording = True

def record_hr():
    """Read heart rate from ESP32 over serial"""
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Connected to ESP32")
        window = []
        window_start = time.time()
        WINDOW_SECS = 5
        while recording:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if 'No signal' in line or 'Signal too HIGH' in line:
                print(f"HR sensor: {line}")
            elif 'Avg BPM=' in line:
                try:
                    avg_bpm = int(line.split('Avg BPM=')[1].split()[0])
                    if avg_bpm > 0:
                        window.append(avg_bpm)
                    now = time.time()
                    if now - window_start >= WINDOW_SECS and window:
                        smoothed = round(sum(window) / len(window))
                        hr_data.append({'timestamp': now, 'bpm': smoothed})
                        print(f"HR: {smoothed} BPM")
                        window = []
                        window_start = now
                except:
                    pass
        ser.close()
    except Exception as e:
        print(f"Serial error: {e}")

#--------------------------- BOOKMARK -------------------------------

def record_audio(filename, use_mic=False):
    """Record audio from both mic and system (BlackHole)"""
    print("Recording audio...")
    def callback(indata, frames, time_info, status):
        audio_chunks.append(indata.copy())

    # List available devices
    devices = sd.query_devices()
    mic_idx = None
    blackhole_idx = None

    for i, d in enumerate(devices):
        if 'blackhole' in d['name'].lower() and d['max_input_channels'] > 0:
            blackhole_idx = i
        if 'macbook' in d['name'].lower() and d['max_input_channels'] > 0:
            mic_idx = i

    print(f"Mic device: {mic_idx} - {devices[mic_idx]['name'] if mic_idx is not None else 'not found'}")
    print(f"BlackHole device: {blackhole_idx} - {devices[blackhole_idx]['name'] if blackhole_idx is not None else 'not found'}")

    # Use BlackHole for system audio (routes meeting audio), fallback to MacBook mic
    # For real meetings: set System Audio Output to BlackHole 2ch (or a Multi-Output Device)
    if use_mic:
        input_device = mic_idx
    else:
        input_device = blackhole_idx if blackhole_idx is not None else mic_idx

    channels = min(2, devices[input_device]['max_input_channels'])
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=channels,
                        device=input_device, callback=callback):
        while recording:
            time.sleep(0.1)

    # Save audio
    audio_data = np.concatenate(audio_chunks, axis=0)
    wav.write(filename, SAMPLE_RATE, audio_data.astype(np.float32))
    print(f"Audio saved to {filename}")

def save_hr_csv(filename):
    """Save heart rate data to CSV"""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'bpm'])
        writer.writeheader()
        writer.writerows(hr_data)
    print(f"HR data saved to {filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mic', action='store_true', help='Use MacBook mic instead of BlackHole')
    args = parser.parse_args()

    # Create filenames with timestamp
    session_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    audio_file = f"{OUTPUT_DIR}/meeting_{session_time}.wav"
    hr_file = f"{OUTPUT_DIR}/hr_{session_time}.csv"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Meeting HR Monitor ===")
    if args.mic:
        print("Audio source: MacBook mic")
    else:
        print("Audio source: BlackHole (system audio)")
    print("Press Ctrl+C to stop recording\n")

    # Start HR and audio recording in parallel threads
    hr_thread = threading.Thread(target=record_hr)
    audio_thread = threading.Thread(target=record_audio, args=(audio_file, args.mic))

    hr_thread.start()
    audio_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        recording = False

    hr_thread.join()
    audio_thread.join()

    save_hr_csv(hr_file)
    print(f"\nSession saved:")
    print(f"  Audio: {audio_file}")
    print(f"  HR data: {hr_file}")

    import subprocess
    print("\nRunning analysis...")
    subprocess.run([
        './venv/bin/python', 'analyze.py', audio_file, hr_file, '--prompt-only'
    ])