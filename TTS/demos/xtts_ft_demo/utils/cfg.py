import os
import sys

import torch

rootdir=os.getcwd()
device="cuda" if torch.cuda.is_available() else "cpu"

TMP_DIR=os.path.join(rootdir,'tmp')
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR,exist_ok=True)

DATASET_DIR=os.path.join(rootdir,'dataset')
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR,exist_ok=True)

TTSMODEL_DIR=os.path.join(rootdir,'models/tts')
if not os.path.exists(TTSMODEL_DIR):
    os.makedirs(TTSMODEL_DIR,exist_ok=True)

FASTERMODEL_DIR=os.path.join(rootdir,'models/faster')
if not os.path.exists(FASTERMODEL_DIR):
    os.makedirs(FASTERMODEL_DIR,exist_ok=True)

UVR5_DIR=os.path.join(rootdir,'uvr5_weights')
# ffmpeg
if sys.platform == 'win32':
    os.environ['PATH'] = rootdir + f';{rootdir}\\ffmpeg;' + os.environ['PATH']
else:
    os.environ['PATH'] = rootdir + f':{rootdir}/ffmpeg:' + os.environ['PATH']
