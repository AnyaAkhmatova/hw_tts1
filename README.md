# TTS project (part 1)

Проект - домашнее задание hw_tts1 по курсу dla. Предназначен для воспроизведения акустической модели FastSpeech2. 

____

За основу репозитория взят темплейт https://github.com/WrathOfGrapes/asr_project_template.git. Структура проекта изменена, большинство базовых классов были удалены, некоторые - перенесены в директории с классами-наследниками. Изменена структура модуля datasets, код для этой части взят из семинарского ноутбука. Директории augmentations, batch_sampler, collate_fn - удалены, все необходимое для работы с датасетом содержится в ./hw_tts/datasets/dataset.py. model и loss содержат код для FastSpeech2. metric, а также специальные модули для решения задачи asr были удалены. trainer был адаптирован под задачу tts (акустическая модель). configs были немного видоизменены. logger и utils были оставлены почти без изменений. 

В папке inference содержится конфиг (config.json) и данные (inference.txt) для инференса. 

train.py был подкорректирован под задачу. inference.py был написан для генерации аудио с помощью полученной акустической модели и вокодера waveglow, предоставленного преподавателями курса (как в семинарском ноутбуке).

Папки с датасетом ./fastspeech2_dataset/ и финальными моделями ./final_model/ скачиваются с google drive, скрипт для скачивания приведен ниже.

WaveGlow и необходимые для его работы пакеты скачиваются отдельно (как в семинарском ноутбуке), скрипт для скачивания приведен ниже.

Dockerfile не валиден, необходимые пакеты устанавливаются с помощью requirements.txt.

making_dataset.ipynb содержит код, с помощью которого был создан датасет. inference.ipynb содержит пример запуска инференса (единственное уточнение, путь к модели теперь либо ./final_model/exp5/model_best.pth, либо ./final_model/exp6/model_best.pth).

____

Устанавливать библиотеки нужно с помощью requirements.txt. Dockerfile невалидный.

Guide по установке:
```
git clone https://github.com/AnyaAkhmatova/hw_tts1.git
```
Из директории ./hw_tts1/ (устанавливаем библиотеки и нашу маленькую библиотечку, скачиваем датасет, финальные модели (результаты экспериментов exp5, exp6, логи, конфиги, чекпоинты, ноутбуки из kaggle), скачиваем waveglow и код из https://github.com/xcmyz/FastSpeech.git для инференса):

```
pip install -r requirements.txt
pip install .

gdown https://drive.google.com/u/0/uc?id=1-4cIK7IXOlpQYNqFoyF3RLMiy14JufGn
unzip fastspeech2_dataset.zip
rm -rf fastspeech2_dataset.zip

gdown https://drive.google.com/u/0/uc?id=1KFOUQ8Nv2h-3-yQH06KdcVtpt1b_og52
unzip final_model.zip
rm -rf final_model.zip

gdown https://drive.google.com/u/0/uc?id=1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx
mkdir -p waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt waveglow/pretrained_model/waveglow_256channels.pt

git clone https://github.com/xcmyz/FastSpeech.git
mv ./FastSpeech/text .
mv ./FastSpeech/audio .
mv ./FastSpeech/waveglow/* waveglow/
mv ./FastSpeech/utils.py .
mv ./FastSpeech/glow.py .
mv ./FastSpeech/hparams.py .
rm -rf ./FastSpeech/
```

Wandb:

```
import wandb

wandb.login()
```

Запуск train:

```
!python3 train.py -c ./hw_tts/configs/exp5.json
```

Запуск inference:

```
!python3 inference.py -c ./inference/config.json -r ./final_model/exp6/model_best.pth
```

Комментарий: инференс не работает в kaggle, для waveglow нужна librosa==0.8.0, в kaggle с этой версией возникают конфликты, мне их разрешить не удалось, поэтому инференс запускала в colab.

____

W&B Report: https://wandb.ai/team-from-wonderland/tts1_project/reports/HW_TTS1-Report--Vmlldzo2MDQ3NDM1.

____

Бонусов нет.

____


Выполнила Ахматова Аня, группа 201.

