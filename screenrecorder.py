import datetime
import pyaudio
import wave
import numpy as np
import cv2
from PIL import ImageGrab
from win32api import GetSystemMetrics
from moviepy.editor import *

# Video settings
width = GetSystemMetrics(0)
height = GetSystemMetrics(1)
time_stamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
video_file_name = f'{time_stamp}.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
captured_video = cv2.VideoWriter(video_file_name, fourcc, 4.0, (width, height))

# Audio settings
audio_file_name = f'{time_stamp}.wav'
audio_format = pyaudio.paInt16
audio_channels = 2
audio_rate = 44100
audio_chunk_size = 1024
audio_speed =0.1 # Adjust the speed factor (0.5 = 50% slower, 2.0 = 2x faster, etc.)

# Initialize audio stream
audio = pyaudio.PyAudio()
audio_stream = audio.open(
    format=audio_format,
    channels=audio_channels,
    rate=int(audio_rate * audio_speed),  # Adjust the sample rate based on speed
    input=True,
    frames_per_buffer=audio_chunk_size
)

# Webcam
webcam = cv2.VideoCapture(0)  # Use the appropriate index for your webcam

# Initialize audio file
audio_frames = []

while True:
    # Capture screen
    img = ImageGrab.grab(bbox=(0, 0, width, height))
    img_np = np.array(img)
    img_final = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # Capture webcam frame
    _, frame = webcam.read()
    fr_height, fr_width, _ = frame.shape
    img_final[0:fr_height, 0:fr_width, :] = frame[0:fr_height, 0:fr_width, :]

    # Display screen capture
    cv2.imshow('Secret Capture', img_final)

    # Write screen capture to video file
    captured_video.write(img_final)

    # Read audio from microphone
    audio_data = audio_stream.read(audio_chunk_size)
    audio_frames.append(audio_data)

    # Check for exit command
    if cv2.waitKey(10) == ord('q'):
        break

# Release resources
captured_video.release()
cv2.destroyAllWindows()

# Save audio frames to a WAV file
audio_stream.stop_stream()
audio_stream.close()
audio.terminate()
audio_wav = wave.open(audio_file_name, 'wb')
audio_wav.setnchannels(audio_channels)
audio_wav.setsampwidth(audio.get_sample_size(audio_format))
audio_wav.setframerate(int(audio_rate * audio_speed))  # Adjust the sample rate based on speed
audio_wav.writeframes(b''.join(audio_frames))
audio_wav.close()

# Combine video and audio using moviepy
video = VideoFileClip(video_file_name)
audio = AudioFileClip(audio_file_name)
final_output = video.set_audio(audio)
final_output.write_videofile(f'{time_stamp}_combined.mp4', codec='libx264')