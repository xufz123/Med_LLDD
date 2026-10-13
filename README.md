# Low-Level Dataset Distillation for Medical Image Enhancement

The implementation of our ECCV2026 paper "Low-Level Dataset Distillation for Medical Image Enhancement".

## Abstract

Medical image enhancement is clinically important, but existing methods often rely on large-scale datasets to learn complex image-to-image mappings. Such reliance leads to high training and storage costs and hinders practical deployment.

Dataset distillation (DD) provides a promising solution by synthesizing compact datasets that preserve the training behavior of the original data. However, existing DD methods mainly focus on high-level tasks, where multiple samples share a common semantic label and the distilled data can compress shared representations. In contrast, low-level medical image enhancement involves dense image-to-image mappings, where each sample corresponds to a unique pixel-level target. This fundamental difference makes low-level DD significantly more challenging and inherently underdetermined.

To address this issue, we propose the first DD framework for low-level medical image enhancement. We construct a shared anatomical prior from a representative subject. This prior serves as the initialization and constrains the solution space. We further personalize the prior through a Structure-Preserving Personalized Generation (SPG) module, which incorporates subject-specific structural characteristics while maintaining fidelity. To capture task-specific degradation effects, we use the distilled dataset to form paired high- and low-quality samples for optimization.

More importantly, we align the optimization behavior of networks trained on the distilled dataset with that of networks trained on the full dataset through subject-aware gradient matching. This design ensures that the distilled data preserve essential training dynamics rather than only appearance-level similarity. Finally, only the distilled dataset is shared for downstream use, so raw patient data remain inaccessible and privacy protection is improved. Extensive experiments across multiple modalities and tasks demonstrate the effectiveness of our method.

## Get Started

### 1. Clone the repository

```git
git clone https://github.com/xufz123/Med_LLDD.git
```

### 2. Set up the environment

```
cd Med_LLDD
pip install -r requirements.txt
```

## Data

We evaluate our method on CT and MRI data using the following datasets:

- **Mayo CT:** The [NIH-AAPM-Mayo Clinic Low-Dose CT Grand Challenge dataset](https://www.aapm.org/grandchallenge/lowdosect/) provides CT images and projection data.
- **CC-359 MRI:** The [Calgary-Campinas-359 (CC-359) dataset](https://www.ccdataset.com/) is a public brain MRI dataset.

For the super-resolution code in this repository, prepare the high-resolution images and organize them as follows:

```text
datasets/
|-- medical_ct/
|   |-- train/
|   |   |-- ...
|   `-- test/
|       |-- ...
```

## Training

Run the following command from the repository root to distill the Mayo CT dataset for 4x super-resolution using the default SRCNN model:

```bash
python main.py \
    --data_path datasets/medical_ct \
    --save_path results/results_medical_ct \
    --spc 5 --dpc 2 --vpc 1
```

Replace `--data_path` and `--save_path` with your dataset and output paths. The dataset root contain the `train` and `test` folders described above.

| Argument | Description |
| --- | --- |
| `--data_path` | Root directory of the CT dataset. |
| `--save_path` | Output directory for the distilled representation and generator checkpoints. |
| `--spc` | The number of raw images used for initialization or selection. |
| `--dpc` | The dimension of subject-specific adjustment code. |
| `--vpc` | The number of images per subject. |

## Citation

If our work is valuable to you, please cite our work:

```bibtex
@inproceedings{xu2026low,
  title={Low-Level Dataset Distillation for Medical Image Enhancement},
  author={Xu, Fengzhi and Yang, Ziyuan and Sun, Mengyu and Zhou, Joey Tianyi and Zhang, Yi},
  booktitle={European Conference on Computer Vision},
  pages={442--459},
  year={2026},
  organization={Springer}
}
```

