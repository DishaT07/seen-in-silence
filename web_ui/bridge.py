import numpy as np
import sounddevice as sd
import threading
import time
import webview

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024
IDLE_TIMEOUT = 5.0

active_notes = []
lock = threading.Lock()
drone_wave = None
drone_pos = 0
drone_gain = 0.0
drone_target_gain = 1.0
last_tap_time = time.time()
stream = None


def audio_callback(outdata, frames, time_info, status):
    global drone_pos, drone_gain
    buffer = np.zeros(frames, dtype=np.float32)
    with lock:
        if drone_wave is not None:
            step = 0.02
            if drone_gain < drone_target_gain:
                drone_gain = min(drone_gain + step, drone_target_gain)
            elif drone_gain > drone_target_gain:
                drone_gain = max(drone_gain - step, drone_target_gain)
            end = drone_pos + frames
            if end <= len(drone_wave):
                buffer += drone_wave[drone_pos:end] * drone_gain
                drone_pos = end % len(drone_wave)
            else:
                part1 = drone_wave[drone_pos:]
                remaining = frames - len(part1)
                part2 = drone_wave[:remaining]
                buffer[:len(part1)] += part1 * drone_gain
                buffer[len(part1):len(part1) + len(part2)] += part2 * drone_gain
                drone_pos = remaining
        still_active = []
        for note in active_notes:
            wave = note["wave"]
            pos = note["pos"]
            end = pos + frames
            chunk = wave[pos:end]
            buffer[:len(chunk)] += chunk
            note["pos"] += len(chunk)
            if note["pos"] < len(wave):
                still_active.append(note)
        active_notes[:] = still_active
    np.clip(buffer, -1.0, 1.0, out=buffer)
    outdata[:, 0] = buffer


def idle_watcher():
    global drone_target_gain
    while True:
        time.sleep(0.5)
        idle_for = time.time() - last_tap_time
        with lock:
            drone_target_gain = 0.0 if idle_for > IDLE_TIMEOUT else 1.0


def generate_tone(frequency, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = (
            1.00 * np.sin(2 * np.pi * frequency * t) +
            0.25 * np.sin(2 * np.pi * frequency * 2 * t) +
            0.08 * np.sin(2 * np.pi * frequency * 3 * t) +
            0.15 * np.sin(2 * np.pi * frequency * 0.5 * t)
    )
    wave = wave / np.max(np.abs(wave))
    kernel_size = 35
    kernel = np.ones(kernel_size) / kernel_size
    wave = np.convolve(wave, kernel, mode="same")
    wave = wave / np.max(np.abs(wave))
    attack_samples = int(sample_rate * 0.4)
    envelope = np.ones_like(wave)
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
    decay = np.exp(-1.0 * np.linspace(0, 1, len(wave)))
    envelope = np.minimum(envelope, 1.0) * decay
    wave *= envelope
    return (wave * 0.4).astype(np.float32)


def trigger_note(wave):
    global last_tap_time
    with lock:
        active_notes.append({"wave": wave, "pos": 0})
    last_tap_time = time.time()


def build_drone(frequency):
    loop_duration = 4.0
    t = np.linspace(0, loop_duration, int(SAMPLE_RATE * loop_duration), False)
    wave = (
            1.00 * np.sin(2 * np.pi * frequency * t) +
            0.3 * np.sin(2 * np.pi * frequency * 1.5 * t)
    )
    wave = wave / np.max(np.abs(wave))
    wave *= 0.10
    return wave.astype(np.float32)


def start_audio_stream():
    global stream
    stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE,
                             blocksize=BLOCK_SIZE, callback=audio_callback)
    stream.start()
    threading.Thread(target=idle_watcher, daemon=True).start()


SCALE_RATIOS = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]

def generate_scale(root_freq, ratios):
    return [root_freq * r for r in ratios]

RANGE_SHIFTS = {"low": 0.5, "mid": 1.0, "high": 2.0}

RASA_PROFILES = {
    "shanta":     {"root": 174.61, "duration": 4.0},
    "shringara":  {"root": 261.63, "duration": 3.5},
    "hasya":      {"root": 349.23, "duration": 2.0},
    "veera":      {"root": 293.66, "duration": 2.5},
    "karuna":     {"root": 196.00, "duration": 4.5},
    "raudra":     {"root": 220.00, "duration": 1.8},
    "bhayanaka":  {"root": 207.65, "duration": 2.8},
    "bibhatsa":   {"root": 155.56, "duration": 3.0},
    "adbhuta":    {"root": 392.00, "duration": 2.5},
}

MOOD_TO_RASA = {
    "calm": "shanta", "love": "shringara", "joyful": "hasya",
    "confident": "veera", "sad": "karuna", "angry": "raudra",
    "anxious": "bhayanaka", "uneasy": "bibhatsa", "surprised": "adbhuta",
}

MOOD_COLORS = {
    "calm": "#b09cd1", "love": "#e68ca0", "joyful": "#ffc759",
    "confident": "#d6593e", "sad": "#5e7294", "angry": "#b23a3a",
    "anxious": "#6d8169", "uneasy": "#6b5b6b", "surprised": "#63bdba",
}

MOOD_SYMBOLS = {
    "calm": "~", "love": "\u2665", "joyful": "\u263a", "confident": "\u25b2",
    "sad": "\u25e1", "angry": "\u2726", "anxious": "?", "uneasy": "\u2248", "surprised": "!",
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def freq_to_note_name(freq):
    semitones_from_a4 = round(12 * np.log2(freq / 440.0))
    note_index = (9 + semitones_from_a4) % 12
    octave = 4 + (9 + semitones_from_a4) // 12
    return f"{NOTE_NAMES[note_index]}{octave}"


def build_tiles(rasa_name, range_name):
    profile = RASA_PROFILES[rasa_name]
    shifted_root = profile["root"] * RANGE_SHIFTS[range_name]
    scale = generate_scale(shifted_root, SCALE_RATIOS)
    tile_map = {str(i + 1): note for i, note in enumerate(scale)}
    return tile_map, profile["duration"], shifted_root


class Api:
    def __init__(self):
        self.wave_map = {}
        self.audio_started = False

    def get_moods(self):
        return [
            {"mood": m, "color": MOOD_COLORS[m], "symbol": MOOD_SYMBOLS[m]}
            for m in MOOD_TO_RASA.keys()
        ]
    def get_manual(self):
        manual = {}
        for mood, rasa in MOOD_TO_RASA.items():
            manual[mood] = {}
            for range_name in RANGE_SHIFTS.keys():
                tile_map, _, _ = build_tiles(rasa, range_name)
                manual[mood][range_name] = [
                    {"tile": tile, "note": freq_to_note_name(freq)}
                    for tile, freq in tile_map.items()
                ]
        return manual

    def select_mood_and_range(self, mood, range_name):
        global drone_wave
        rasa = MOOD_TO_RASA[mood]
        tile_map, duration, root_freq = build_tiles(rasa, range_name)

        self.wave_map = {
            tile: generate_tone(freq, duration) for tile, freq in tile_map.items()
        }
        drone_wave = build_drone(root_freq * 0.5)

        if not self.audio_started:
            start_audio_stream()
            self.audio_started = True

        tiles = []
        for tile_num, freq in tile_map.items():
            tiles.append({
                "id": tile_num,
                "note": freq_to_note_name(freq),
            })
        return {
            "tiles": tiles,
            "color": MOOD_COLORS[mood],
            "symbol": MOOD_SYMBOLS[mood],
        }

    def tap_tile(self, tile_id):
        if tile_id in self.wave_map:
            trigger_note(self.wave_map[tile_id])
        return True


if __name__ == "__main__":
    api = Api()
    window = webview.create_window(
        "SeenInSilence", "index.html", js_api=api,
        width=900, height=650, resizable=True
    )
    webview.start()