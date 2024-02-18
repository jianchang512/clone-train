
# 执行 ffmpeg
import json
import os
import subprocess
import sys
from datetime import timedelta

from faster_whisper import WhisperModel
from pydub.silence import detect_nonsilent

from TTS.config import FASTERMODEL_DIR,TTSMODEL_DIR,device
from pydub import AudioSegment

def runffmpeg(arg):
    cmd = ["ffmpeg", "-hide_banner", "-ignore_unknown","-vsync", "vfr"]+arg
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         encoding="utf-8",
                         text=True,
                         creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
    print(f"runffmpeg: {' '.join(cmd)}")

    while True:
        try:
            # 等待0.1未结束则异常
            outs, errs = p.communicate(timeout=0.5)
            errs = str(errs)
            if errs:
                errs = errs.replace('\\\\', '\\').replace('\r', ' ').replace('\n', ' ')
                errs = errs[errs.find("Error"):]
            # 如果结束从此开始执行
            if p.returncode == 0:
                # 成功
                return True
            # 失败
            raise Exception(f'ffmpeg error:{errs=}')
        except subprocess.TimeoutExpired as e:
            pass
        except Exception as e:
            raise Exception(str(e))

model = None

def recogn(wavfile,language="zh"):
    global model
    if model is None:
        model = WhisperModel("large-v3", device=device,
                         # compute_type=config.settings['cuda_com_type'],
                         download_root=FASTERMODEL_DIR,
                         local_files_only=False)

    prompt='使用简体中文转录。'
    segments, _ = model.transcribe(wavfile,language=language,initial_prompt=None if language!='zh' else prompt)
    text=""
    i=0
    for t in segments:
        if t.text==prompt:
            continue
        text += t.text + " "
    return text.strip()
# 按照10s分割
# def split_audio(input_audio, target_directory, chunk_length_ms=10000,language="zh"):
#     print(f'input split')
#     # Load the audio file
#     audio = AudioSegment.from_wav(input_audio)
#
#     # Create the directory if it does not exist
#     if not os.path.exists(target_directory):
#         os.makedirs(target_directory)
#
#     # Initialize an empty list to store the file names
#     chunk_filenames = []
#
#     # Calculate the number of chunks
#     number_of_chunks = len(audio) // chunk_length_ms
#
#
#
#     # Split the audio and save the chunks
#     for i in range(number_of_chunks):
#         # Calculate the start and end times for the chunk
#         start_time = i * chunk_length_ms
#         end_time = start_time + chunk_length_ms
#
#         # Extract the chunk
#         chunk = audio[start_time:end_time]
#
#         # Create a filename for the chunk
#         chunk_filename = os.path.join(target_directory, f"{i}.wav")
#
#         # Export the chunk
#         chunk.export(chunk_filename, format="wav")
#         try:
#             print(f'{chunk_filename=}')
#             text=recogn(chunk_filename)
#             print(f'{text=}')
#             obj={
#                 "wav":chunk_filename,
#                 "text":text,
#                 "language":language
#             }
#             chunk_filenames.append(obj)
#         except Exception as e:
#             print(f'{chunk_filename=},{str(e)}')
#             continue
#     return chunk_filenames



def split_audio(wavfile,target_dir,language="zh"):
    result=[]    
    chunk_length_ms=10000
    audio = AudioSegment.from_wav(wavfile)

    # Create the directory if it does not exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Initialize an empty list to store the file names
    chunk_filenames = []

    # Calculate the number of chunks
    number_of_chunks = len(audio) // chunk_length_ms



    # Split the audio and save the chunks
    for i in range(number_of_chunks):
        # Calculate the start and end times for the chunk
        start_time = i * chunk_length_ms
        end_time = start_time + chunk_length_ms

        # Extract the chunk
        chunk = audio[start_time:end_time]

        # Create a filename for the chunk
        chunk_filename = os.path.join(target_dir, f"{i}.wav")

        # Export the chunk
        chunk.export(chunk_filename, format="wav")
        try:
            print(f'{chunk_filename=}')
            text=recogn(chunk_filename)
            print(f'{text=}')
            obj={
                "wav":f"{i}",
                "text":text,
                "language":language,
                "duration":10000,
                "length":len(text)
            }
            result.append(obj)
        except Exception as e:
            print(f'{chunk_filename=},{str(e)}')
    return result

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


# split audio by silence
def shorten_voice(normalized_sound):
    normalized_sound = match_target_amplitude(normalized_sound, -20.0)
    max_interval = 10000
    buffer = 500
    nonsilent_data = []
    audio_chunks = detect_nonsilent(normalized_sound, min_silence_len=500,
                                    silence_thresh=-20 - 25)
    # print(audio_chunks)
    for i, chunk in enumerate(audio_chunks):
        start_time, end_time = chunk
        n = 0
        while end_time - start_time >= max_interval:
            n += 1
            # new_end = start_time + max_interval+buffer
            new_end = start_time + max_interval + buffer
            new_start = start_time
            nonsilent_data.append((new_start, new_end, True))
            start_time += max_interval
        nonsilent_data.append((start_time, end_time, False))
    return nonsilent_data

def split_audio2(wavfile,target_dir,language="zh"):
    normalized_sound = AudioSegment.from_wav(wavfile)  # -20.0
    nonslient_file = f'{target_dir}/detected_voice.json'
    if os.path.exists(nonslient_file) and os.path.getsize(nonslient_file):
        with open(nonslient_file, 'r') as infile:
            nonsilent_data = json.load(infile)
    else:
        nonsilent_data = shorten_voice(normalized_sound)
        with open(nonslient_file, 'w') as outfile:
            json.dump(nonsilent_data, outfile)
    result=[]
    pre=0
    for i, duration in enumerate(nonsilent_data):
        start_time, end_time, buffered = duration
        if start_time+3000 > end_time:
            continue

        chunk_filename = os.path.join(target_dir, f"{i}.wav")
        audio_chunk = normalized_sound[start_time:end_time]
        audio_chunk.export(chunk_filename, format="wav")
        time_dur=end_time-start_time
        try:
            print(f'{chunk_filename=}')
            text=recogn(chunk_filename)
            print(f'{text=}')
            obj={
                "wav":chunk_filename,
                "text":text,
                "language":language,
                "duration":time_dur
            }
            result.append(obj)
        except Exception as e:
            print(f'{chunk_filename=},{str(e)}')
    return result

# run ffprobe 获取视频元信息
def runffprobe(cmd):
    try:
        cmd[-1]=os.path.normpath(rf'{cmd[-1]}')
        p = subprocess.Popen(['ffprobe']+cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,encoding="utf-8", text=True,
                             creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW)
        out, errs = p.communicate()
        if p.returncode == 0:
            return out.strip()
        raise Exception(f'ffprobe error:{str(errs)}')
    except subprocess.CalledProcessError as e:
        raise Exception(f'ffprobe call error:{str(e)}')
    except Exception as e:
        raise Exception(f'ffprobe except error:{str(e)}')

# 获取视频信息
def get_audio_time(audio_file):
    # 如果存在缓存并且没有禁用缓存
    out = runffprobe(['-v','quiet','-print_format','json','-show_format','-show_streams',audio_file])
    if out is False:
        raise Exception(f'ffprobe error:dont get video information')
    out = json.loads(out)
    return float(out['format']['duration'])

def ms_to_time_string(*, ms=0, seconds=None):
    # 计算小时、分钟、秒和毫秒
    if seconds is None:
        td = timedelta(milliseconds=ms)
    else:
        td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    # 格式化为字符串
    time_string = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    return time_string