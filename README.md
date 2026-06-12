# KPDM

Official repository for KPDM: Key Phrase Dynamic Masking for Robust Text-to-Image Person Retrieval (AAAI 2026).

## Inference

### Requirements

```
torch>=2.0.0
torchvision>=0.15.0
Pillow
ftfy
regex
prettytable
easydict
PyYAML
```

### Data Preparation

Download a dataset (e.g. [RSTPReid](https://github.com/NjtechCVLab/RSTPReid)) and place it under `./data/`:

```
data/
├── RSTPReid/
│   ├── imgs/
│   └── data_captions.json
└── bpe_simple_vocab_16e6.txt.gz
```

The BPE vocabulary can be downloaded from [OpenAI CLIP](https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz).

### Checkpoints

Pretrained checkpoints and configs are available at [huggingface.co/you93887/KPDM](https://huggingface.co/you93887/KPDM).

| Dataset    | Checkpoint            |
| ---------- | --------------------- |
| CUHK-PEDES | `kpdm_cuhk_pedes.pth` |
| ICFG-PEDES | `kpdm_icfg_pedes.pth` |
| RSTPReid   | `kpdm_rstpreid.pth`   |

### Run

```bash
python test.py \
    --config_file configs/rstpreid.yaml \
    --checkpoint path/to/kpdm_rstpreid.pth
```


## Citation

```bibtex
@inproceedings{you2026kpdm,
  title={KPDM: Key Phrase Dynamic Masking for Robust Text-to-Image Person Retrieval},
  author={You, Shaofeng and Miao, Tianle and Chen, Qihang and Li, Xin and Cheng, Zhuo and Luo, Dapeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={12099--12107},
  year={2026}
}
```
