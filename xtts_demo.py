import argparse
import os
import sys
import tempfile

import gradio as gr
import librosa.display
import numpy as np

import os
import torch
import torchaudio
import traceback
from TTS.demos.xtts_ft_demo.utils.formatter import format_audio_list
from TTS.demos.xtts_ft_demo.utils.gpt_train import train_gpt

from TTS.demos.xtts_ft_demo.utils.cfg import TTSMODEL_DIR

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import datetime
import shutil

wav_path=None
copying=False

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

XTTS_MODEL = None
def load_model(xtts_checkpoint, xtts_config, xtts_vocab):
    global XTTS_MODEL
    clear_gpu_cache()
    if not xtts_checkpoint or not xtts_config or not xtts_vocab:
        gr.Error('训练尚未结束，请稍等')
        return "训练尚未结束，请稍等"
    config = XttsConfig()
    config.load_json(xtts_config)
    XTTS_MODEL = Xtts.init_from_config(config)
    print("Loading XTTS model! ")
    XTTS_MODEL.load_checkpoint(config, checkpoint_path=xtts_checkpoint, vocab_path=xtts_vocab, use_deepspeed=False)
    if torch.cuda.is_available():
        XTTS_MODEL.cuda()

    print("Model Loaded!")
    return "模型已加载!"

def run_tts(lang, tts_text, speaker_audio_file):
    if XTTS_MODEL is None:
        gr.Error("模型还未训练完毕或尚未加载，请稍等")
        return "模型还未训练完毕或尚未加载，请稍等 !!", None, None
    if speaker_audio_file and not speaker_audio_file.endswith(".wav"):
        speaker_audio_file+='.wav'
    if not speaker_audio_file or  not os.path.exists(speaker_audio_file):
        gr.Error('必须填写参考音频')
        return '必须填写参考音频',None,None
    gpt_cond_latent, speaker_embedding = XTTS_MODEL.get_conditioning_latents(audio_path=speaker_audio_file, gpt_cond_len=XTTS_MODEL.config.gpt_cond_len, max_ref_length=XTTS_MODEL.config.max_ref_len, sound_norm_refs=XTTS_MODEL.config.sound_norm_refs)
    out = XTTS_MODEL.inference(
        text=tts_text,
        language=lang,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=XTTS_MODEL.config.temperature, # Add custom parameters here
        length_penalty=XTTS_MODEL.config.length_penalty,
        repetition_penalty=XTTS_MODEL.config.repetition_penalty,
        top_k=XTTS_MODEL.config.top_k,
        top_p=XTTS_MODEL.config.top_p,
    )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = fp.name
        torchaudio.save(out_path, out["wav"], 24000)

    return "已创建好了声音 !", out_path, speaker_audio_file




# define a logger to redirect 
class Logger:
    def __init__(self, filename="log.out"):
        self.log_file = filename
        self.terminal = sys.stdout
        self.log = open(self.log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False

# redirect stdout and stderr to a file
sys.stdout = Logger()
sys.stderr = sys.stdout


# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def read_logs():
    sys.stdout.flush()
    with open(sys.stdout.log_file, "r") as f:
        return f.read()


if __name__ == "__main__":
    date=datetime.datetime.now()
    
    parser = argparse.ArgumentParser(
        description="""XTTS fine-tuning demo\n\n"""
        """
        Example runs:
        set http_proxy=http://127.0.0.1:10809

        python TTS/demos/xtts_ft_demo/xtts_demo.py --port 
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to run the gradio demo. Default: 5003",
        default=5003,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        help="Output path (where data and checkpoints will be saved) Default: /tmp/xtts_ft/",
        default=TTSMODEL_DIR,
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs to train. Default: 10",
        default=1
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size. Default: 4",
        default=2,
    )
    parser.add_argument(
        "--grad_acumm",
        type=int,
        help="Grad accumulation steps. Default: 1",
        default=1,
    )
    parser.add_argument(
        "--max_audio_length",
        type=int,
        help="Max permitted audio size in seconds. Default: 11",
        default=10,
    )

    args = parser.parse_args()

    with gr.Blocks(css="ul.options[role='listbox']{background:#ffffff}") as demo:
        with gr.Tab("开始训练"):
            model_name= gr.Textbox(
                        label="模型名称(仅限英文/数字/下划线):",
                        value=f"model{date.day}{date.hour}{date.minute}",
            )
            upload_file = gr.File(
                file_count="multiple",
                label="选择训练素材音频文件，仅包含同一个人声，无背景噪声(wav, mp3, and flac)",
            )
            with gr.Row() as r1:
                lang = gr.Dropdown(
                    label="音频发声语言",
                    value="zh",
                    choices=[
                        "zh",
                        "en",
                        "es",
                        "fr",
                        "de",
                        "it",
                        "pt",
                        "pl",
                        "tr",
                        "ru",
                        "nl",
                        "cs",
                        "ar",
                        "hu",
                        "ko",
                        "ja"
                    ],
                )
                progress_data = gr.Label(
                    label="进度:"
                )
            logs = gr.Textbox(
                label="日志:",
                interactive=False,
            )
            demo.load(read_logs, None, logs, every=1)

            prompt_compute_btn = gr.Button(value="开始训练")
            with gr.Row():
                with gr.Column() as col1:
                    xtts_checkpoint = gr.Textbox(
                        label="训练后模型保存路径:",
                        value="",
                        interactive=False
                    )
                    xtts_config = gr.Textbox(
                        label="训练后模型配置文件:",
                        value="",
                        interactive=False
                    )

                    xtts_vocab = gr.Textbox(
                        label="vocab文件:",
                        value="",
                        interactive=False
                    )
            
            with gr.Row():
                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="参考音频:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="文字语言",
                        value="zh",
                        choices=[
                            "zh",
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="输入要合成的文字.",
                        value="你好啊，我亲爱的朋友.",
                    )
                    tts_btn = gr.Button(value="使用训练的模型生成声音/测试效果")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="进度:"
                    )
                    tts_output_audio = gr.Audio(label="生成的声音.")
                    reference_audio = gr.Audio(label="作为参考的音频.")
            
            move_btn = gr.Button(value="在clone-voice中使用它")
            copy_label=gr.Label(label="")
            
            
            def update_refer(interface_components,new_label=""):
                # Update the label of the textbox
                interface_components["speaker_reference_audio"].label = new_label
                # Rebuild and return the updated components
                return interface_components

            
            
            def train_model(language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length):
                global wav_path
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    return "不存在有效的csv文件 !", "", "", "", ""
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path=output_path, max_audio_length=max_audio_length)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"训练出错了: {error}", "", "", "", ""

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                print("训练完毕!")
                clear_gpu_cache()
                wav_path=os.path.dirname(speaker_wav)
                update_refer({"speaker_reference_audio": speaker_reference_audio},f"你也可以从 {wav_path} 中选择一个质量最好的音频,将名称填在此处代替默认所选，将能优化效果")
                return "训练完毕!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav
        
            def preprocess_dataset(audio_path, language,  progress=gr.Progress(track_tqdm=True)):
                clear_gpu_cache()
                out_path = os.path.join(args.out_path, "dataset")
                os.makedirs(out_path, exist_ok=True)
                
                try:
                    train_meta, eval_meta, audio_total_size = format_audio_list(audio_path, target_language=language, out_path=out_path, gradio_progress=progress)
                except:
                    traceback.print_exc()
                    error = traceback.format_exc()
                    return f"处理训练数据出错了! \n Error summary: {error}", "", "","",""

                clear_gpu_cache()

                # if audio total len is less than 2 minutes raise an error
                if audio_total_size < 120:
                    message = "素材总时长不得小于2分钟!"
                    print(message)
                    return message, "", "","",""

                print("数据处理完毕，开始训练!")
                msg, config_path, vocab_file, ft_xtts_checkpoint, speaker_wav=train_model(language, train_meta, eval_meta, args.num_epochs, args.batch_size, args.grad_acumm, args.out_path, args.max_audio_length)
                progress_data, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio
                msg=load_model(
                    ft_xtts_checkpoint,
                    config_path,
                    vocab_file
                )
                
                return msg, config_path, vocab_file, ft_xtts_checkpoint, speaker_wav
            
            def move_to_clone(model_name,model_file,vocab,cfg,audio_file):
                global copying
                if not wav_path or not os.path.exists(wav_path):
                    gr.Warning("必须填写参考音频")
                    return "必须填写参考音频"
                if copying:
                    gr.Info('正在复制到clone中...')
                    return "正在复制到clone中"
                gr.Info('开始复制到clone自定义模型下，请耐心等待提示完成')
                copying=True
                print(f'{model_name=}')
                print(f'{model_file=}')
                print(f'{vocab=}')
                print(f'{cfg=}')
                print(f'{audio_file=}')
                model_dir=os.path.join(os.getcwd(),f'models/mymodels/{model_name}')
                os.makedirs(model_dir,exist_ok=True)
                shutil.copy2(model_file,os.path.join(model_dir,'model.pth'))
                shutil.copy2(vocab,os.path.join(model_dir,'vocab.json'))
                shutil.copy2(cfg,os.path.join(model_dir,'config.json'))
                shutil.copy2(audio_file,os.path.join(model_dir,'base.wav'))
                gr.Info('已复制到clone自定义模型目录下了，可以去使用咯')
                copying=False
                return "已复制到clone自定义模型目录下了，可以去使用咯"
            
            move_btn.click(
                fn=move_to_clone,
                inputs=[
                    model_name,
                    xtts_checkpoint,
                    xtts_vocab,
                    xtts_config,
                    speaker_reference_audio
                ],
                outputs=[
                    copy_label
                ]
            )
            
            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    lang
                ],
                outputs=[
                    progress_data, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio
                ],
            )

           

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                ],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )
            
            

            
            
        '''    
        with gr.Tab("使用已训练好的模型"):
            with gr.Row():
                with gr.Column() as col1:
                    xtts_checkpoint = gr.Textbox(
                        label="模型路径:",
                        value="",
                    )
                    xtts_config = gr.Textbox(
                        label="模型配置文件:",
                        value="",
                    )

                    xtts_vocab = gr.Textbox(
                        label="vocab文件:",
                        value="",
                    )
                    #load_btn = gr.Button(value="Step 3 - Load Fine-tuned XTTS model")

                with gr.Column() as col2:
                    speaker_reference_audio = gr.Textbox(
                        label="参考音频:",
                        value="",
                    )
                    tts_language = gr.Dropdown(
                        label="文字语言",
                        value="zh",
                        choices=[
                            "zh",
                            "en",
                            "es",
                            "fr",
                            "de",
                            "it",
                            "pt",
                            "pl",
                            "tr",
                            "ru",
                            "nl",
                            "cs",
                            "ar",
                            "hu",
                            "ko",
                            "ja",
                        ]
                    )
                    tts_text = gr.Textbox(
                        label="输入要合成的文字.",
                        value="你好啊，我亲爱的朋友.",
                    )
                    tts_btn = gr.Button(value="生成声音")

                with gr.Column() as col3:
                    progress_gen = gr.Label(
                        label="进度:"
                    )
                    tts_output_audio = gr.Audio(label="生成的声音.")
                    reference_audio = gr.Audio(label="参考音频.")

            prompt_compute_btn.click(
                fn=preprocess_dataset,
                inputs=[
                    upload_file,
                    lang
                ],
                outputs=[
                    progress_data, xtts_config, xtts_vocab, xtts_checkpoint, speaker_reference_audio
                ],
            )

           

            tts_btn.click(
                fn=run_tts,
                inputs=[
                    tts_language,
                    tts_text,
                    speaker_reference_audio,
                ],
                outputs=[progress_gen, tts_output_audio, reference_audio],
            )
        '''
    demo.launch(
        share=True,
        debug=False,
        server_port=args.port,
        server_name="0.0.0.0"
    )
