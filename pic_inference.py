import task
import torch
import fairseq
from fairseq import utils
from fairseq_cli import generate
from PIL import Image
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import deit
import trocr_models
import time

def init(model_path, beam=5):

    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        arg_overrides={
            "beam": beam,
            "task": "text_recognition",
            "data": "",
            "fp16": False,
        },
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model[0].to(device)

    img_transform = transforms.Compose(
        [
            transforms.Resize((384, 384), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ]
    )

    generator = task.build_generator(
        model,
        cfg.generation,
        extra_gen_cls_kwargs={"lm_model": None, "lm_weight": None},
    )

    bpe = task.build_bpe(cfg.bpe)

    return model, cfg, task, generator, bpe, img_transform, device


def preprocess(img_path, img_transform):
    im = Image.open(img_path).convert("RGB").resize((384, 384))
    im = img_transform(im).unsqueeze(0).to(device).float()

    sample = {
        "net_input": {"imgs": im},
    }

    return sample


def get_text(cfg, generator, model, sample, bpe):
    decoder_output = task.inference_step(
        generator, model, sample, prefix_tokens=None, constraints=None
    )
    decoder_output = decoder_output[0][0]  # top1

    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
        hypo_tokens=decoder_output["tokens"].int().cpu(),
        src_str="",
        alignment=decoder_output["alignment"],
        align_dict=None,
        tgt_dict=model[0].decoder.dictionary,
        remove_bpe=cfg.common_eval.post_process,
        extra_symbols_to_ignore=generate.get_symbols_to_strip_from_output(generator),
    )

    detok_hypo_str = bpe.decode(hypo_str)

    return detok_hypo_str


if __name__ == "__main__":
    model_path = "saved_models/ft_name_insi_checkpoint_best.pt"
    data_dir = "data/name_insi"
    output_dir = "output"
    img_dir = os.path.join(data_dir, "image")
    val_data_file = os.path.join(data_dir, "gt_valid.txt")
    plot = True

    with open(val_data_file, "r") as f:
        lines = f.readlines()
    img_paths = []
    for line in lines:
        img_path = os.path.join(img_dir, line.split("\t")[0])
        img_paths.append(img_path)
    beam = 5

    model, cfg, task, generator, bpe, img_transform, device = init(model_path, beam)

    f = open(os.path.dirname(img_dir) + "/results.txt", "w")
    count = 0
    tt = 0
    for img_path in tqdm(img_paths):
        start = time.time()
        sample = preprocess(img_path, img_transform)
        text = get_text(cfg, generator, model, sample, bpe)
        tt_time = time.time() - start
        tt += tt_time
        if plot:
            os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
            img = Image.open(img_path)
            plt.imshow(img)
            plt.title(text)
            plt.savefig(os.path.join(output_dir, "plots", os.path.basename(img_path)))

        line = os.path.basename(img_path) + "\t" + text + "\n"
        f.write(line)
        count += 1
        if count == 0:
            break
    f.close()
    print("done:", tt / count)
