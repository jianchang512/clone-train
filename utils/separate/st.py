import math
import os
import time
import traceback


import librosa
import soundfile as sf
import torch
from .vr import AudioPre

from ..cfg import UVR5_DIR,device
from ..tools import get_audio_time, ms_to_time_string, runffmpeg


def uvr(*,model_name=None, save_root=None, inp_path=None):
    infos = []
    try:
        func = AudioPre
        pre_fun = func(
            agg=10,
            model_path=os.path.join(UVR5_DIR, f"{model_name}.pth"),
            device=device,
            is_half=False
        )
        done = 0
        try:
            y, sr = librosa.load(inp_path, sr=None)
            info = sf.info(inp_path)
            channels = info.channels
            if channels == 2 and sr == 44100:
                need_reformat = 0
                pre_fun._path_audio_(
                    inp_path, save_root
                )
                done = 1
            else:
                need_reformat = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (
                os.path.join(os.environ["TEMP"]),
                f'{os.path.basename(inp_path)}-{time.time()}',
            )
            runffmpeg([
                "-y",
                "-i",
                inp_path,
                "-ar",
                "44100",
                tmp_path
            ])
            inp_path = tmp_path
        try:
            if done == 0:
                pre_fun._path_audio_(
                    inp_path, save_root
                )
            infos.append("%s->Success" % (os.path.basename(inp_path)))
            yield "\n".join(infos)
        except:
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            del pre_fun.model
            del pre_fun
        except:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            traceback.print_exc()
    yield "\n".join(infos)



def start(audio,path,duration=0):
    dist=100
    try:
        # duration总时长秒
        if duration<=dist:
            gr = uvr(model_name="HP2", save_root=path, inp_path=audio)
            print(next(gr))
            print(next(gr))
            return
        length=math.ceil(duration/dist)
        result=[]
        for i in range(length):
            #创建新目录，存放vocal.wav
            save_root=os.path.join(path,f'{i}')
            os.makedirs(save_root,exist_ok=True)
            #新音频存放
            inp_path=os.path.join(path,f'{i}.wav')
            print(f'{inp_path=}')
            cmd=['-y','-i',audio,'-ss',ms_to_time_string(seconds=i*dist)]
            if i<length-1:
                #不是最后一个
                cmd+=['-t',f'{dist}']
            cmd.append(inp_path)
            print(f'{cmd=}')
            runffmpeg(cmd)
            # continue
            gr = uvr(model_name="HP2", save_root=save_root, inp_path=inp_path)
            print(next(gr))
            print(next(gr))
            file_=os.path.join(save_root,'vocal.wav')
            result.append(f"file '{file_}'")
        concat_txt=os.path.join(path,'1.txt')
        with open(concat_txt,'w',encoding='utf-8') as f:
            f.write("\n".join(result))
        runffmpeg(['-y','-f','concat','-safe','0','-i',concat_txt,os.path.join(path,'vocal.wav')])


    except Exception as e:
        msg=f"separate vocal and background music:{str(e)}"
        print(msg)
        raise Exception(msg)


