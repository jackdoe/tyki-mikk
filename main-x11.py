from Xlib import X, display, XK
from Xlib.ext import record
from Xlib.protocol import rq
import numpy as np
import pyaudio
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import pyautogui

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.generation_config.language = "<|en|>"
model.generation_config.task = "transcribe"
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

pressed = threading.Event()
local_dpy = display.Display()
record_dpy = display.Display()
ROOT = local_dpy.screen().root
F_KEYCODE = local_dpy.keysym_to_keycode(XK.string_to_keysym("F8"))
ROOT.grab_key(F_KEYCODE, 0, True, X.GrabModeAsync, X.GrabModeAsync)

def record_callback(reply):
    if reply.category != record.FromServer or reply.client_swapped or not len(reply.data):
        return

    data = reply.data
    while len(data):
        event, data = rq.EventField(None).parse_binary_value(data, record_dpy.display, None, None)

        if event.detail == F_KEYCODE:
            if event.type == X.KeyPress and not pressed.is_set():
                pressed.set()
            elif event.type == X.KeyRelease and pressed.is_set():
                pressed.clear()

def processor():
    pa = pyaudio.PyAudio()
    beeper = pa.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=44_100,
                 output=True)

    t = np.linspace(0, 0.10, int(44_100 * 0.10), False)
    beep_1 = (0.2 * np.sin(2 * np.pi * 550 * t)).astype(np.float32).tobytes()
    beep_2 = (0.2 * np.sin(2 * np.pi * 660 * t)).astype(np.float32).tobytes()
    while True:
        if pressed.wait():
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16_000,
                input=True,
                frames_per_buffer=1_024,
            )

            beeper.write(beep_1)
            
            frames = []
            while pressed.is_set():
                data = stream.read(1_024, exception_on_overflow=False)
                frames.append(data)

            stream.stop_stream()
            stream.close()

            beeper.write(beep_2)

            audio_bytes = b"".join(frames)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32_768.0
            result = pipe(audio_float, batch_size=1)

            pyautogui.write(result["text"].strip(), interval=0.001)

threading.Thread(target=processor,daemon=True).start()
ctx = record_dpy.record_create_context(
        0,
        [record.AllClients],
        [{
            'core_requests'   : (0, 0),
            'core_replies'    : (0, 0),
            'ext_requests'    : (0, 0, 0, 0),
            'ext_replies'     : (0, 0, 0, 0),
            'delivered_events': (0, 0),
            'device_events'   : (X.KeyPress, X.KeyRelease),
            'errors'          : (0, 0),
            'client_started'  : False,
            'client_died'     : False,
        }]
)

record_dpy.record_enable_context(ctx, record_callback)
record_dpy.record_free_context(ctx)
