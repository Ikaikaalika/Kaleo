python
import librosa
from pydub import AudioSegment
from google.cloud import speech_v1p1beta1 as speech

def detect_language_segments(file_path):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=f'gs://your-bucket/{file_path}')
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US,haw-US",
        enable_automatic_punctuation=True,
        enable_word_time_offsets=True
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=90)
    
    segments = []
    for result in response.results:
        for alternative in result.alternatives:
            if alternative.language_code == 'haw-US':
                start = alternative.words[0].start_time.total_seconds()
                end = alternative.words[-1].end_time.total_seconds()
                segments.append({'start': start, 'end': end, 'language': 'haw'})
    return segments

def clean_audio(file_path, output_path):
    y, sr = librosa.load(file_path)
    segments = detect_language_segments(file_path.split('/')[-1])
    
    for i, segment in enumerate(segments):
        if segment['language'] == 'haw':
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            segment_audio = y[start_sample:end_sample]
            librosa.output.write_wav(f'{output_path}/haw_segment_{i}.wav', segment_audio, sr)

if __name__ == "__main__":
    clean_audio('data/raw/your_audio_file.wav', 'data/cleaned')