import argparse
import json
import os
from pathlib import Path

import torch

import hw_tts.model as module_model
from hw_tts.utils.parse_config import ConfigParser

import numpy as np
import pandas as pd
import soundfile as sf

import waveglow
import text
import audio
import utils

import wandb

import warnings 
warnings.filterwarnings("ignore")


def synthesis(model, cur_text, device, alpha=1.0, beta=1.0, gamma=1.0):
    cur_text = np.array(cur_text)
    cur_text = np.stack([cur_text])
    src_pos = np.array([i+1 for i in range(cur_text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(cur_text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    model.eval()
    with torch.no_grad():
        mel, _, _, _, _, _ = model.forward(sequence, src_pos, alpha=alpha, beta=beta, gamma=gamma)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)


def get_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        texts = []
        for line in f.readlines():
            texts.append(line)

    texts_encoded = [text.text_to_sequence(cur_text, ["english_cleaners"]) for cur_text in texts]
    return texts, texts_encoded


def main(config):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()

    input_filename = config["inference"]["input_filename"]

    output_dir = config["inference"]["output_dir"]
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    duration_part = config["inference"]["duration_part"]
    pitch_part = config["inference"]["pitch_part"]
    energy_part = config["inference"]["energy_part"]

    wandb_logging = config["inference"]["wandb_logging"]
    if wandb_logging:
        wandb.init(
                project=config['trainer'].get('wandb_project'),
                config=config
            )

    df = pd.DataFrame(columns=["generated audio", "alpha", "beta", "gamma", "text"])

    texts, texts_encoded = get_data(input_filename)
    idx = 0
    for alpha in duration_part:
        for beta in pitch_part:
            for gamma in energy_part:
                for i, token_ids in enumerate(texts_encoded):
                    mel, mel_cuda = synthesis(model, token_ids, device, alpha, beta, gamma)

                    audio.tools.inv_mel_spec(
                        mel, f"{output_dir}/{i}_alpha={alpha}_beta={beta}_gamma={gamma}.wav"
                    )

                    waveglow.inference.inference(
                        mel_cuda, WaveGlow, f"{output_dir}/{i}_alpha={alpha}_beta={beta}_gamma={gamma}_waveglow.wav"
                    )

                    if wandb_logging:
                        cur_audio, sr = sf.read(f"{output_dir}/{i}_alpha={alpha}_beta={beta}_gamma={gamma}_waveglow.wav")
                        df.loc[idx] = [wandb.Audio(cur_audio.reshape(-1, 1), sample_rate=sr), alpha, beta, gamma, texts[i]]
                        idx += 1

    if wandb_logging:
        wandb.log({"inference_results": wandb.Table(dataframe=df)})
        print('Table is added to wandb')


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    assert config.config.get("inference", {}) is not None

    main(config)
