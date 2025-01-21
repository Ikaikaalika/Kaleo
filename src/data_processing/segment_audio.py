from pydub import AudioSegment

def segment_audio(input_file, output_dir, segments):
    audio = AudioSegment.from_wav(input_file)
    for i, segment in enumerate(segments):
        if segment['language'] == 'haw':
            segment_audio = audio[segment['start']*1000:segment['end']*1000]
            segment_audio.export(f"{output_dir}/haw_segment_{i}.wav", format="wav")

# Example usage:
# segment_audio('data/raw/file.wav', 'data/cleaned', [{'start': 0, 'end': 5, 'language': 'haw'}, ...])