---
task_categories:
- image-segmentation
language:
- en
tags:
- medical
size_categories:
- 1K<n<10K
---
# LLD-MMRI-MedSAM2 Dataset

<div align="center">
 <table align="center">
   <tr>
     <td><a href="https://arxiv.org/abs/2504.03600" target="_blank"><img src="https://img.shields.io/badge/arXiv-Paper-FF6B6B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a></td>
     <td><a href="https://medsam2.github.io/" target="_blank"><img src="https://img.shields.io/badge/Project-Page-4285F4?style=for-the-badge&logoColor=white" alt="Project"></a></td>
     <td><a href="https://github.com/bowang-lab/MedSAM2" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a></td>
     <td><a href="https://huggingface.co/wanglab/MedSAM2" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Model-FFBF00?style=for-the-badge&logo=huggingface&logoColor=white" alt="HuggingFace Model"></a></td>
   </tr>
   <tr>
     <td><a href="https://medsam-datasetlist.github.io/" target="_blank"><img src="https://img.shields.io/badge/Dataset-List-00B89E?style=for-the-badge" alt="Dataset List"></a></td>
     <td><a href="https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2" target="_blank"><img src="https://img.shields.io/badge/Dataset-CT__DeepLesion-28A745?style=for-the-badge" alt="CT_DeepLesion-MedSAM2"></a></td>
     <td><a href="https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2" target="_blank"><img src="https://img.shields.io/badge/Dataset-LLD--MMRI-FF6B6B?style=for-the-badge" alt="LLD-MMRI-MedSAM2"></a></td>
     <td><a href="https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2" target="_blank"><img src="https://img.shields.io/badge/3D_Slicer-Plugin-e2006a?style=for-the-badge" alt="3D Slicer"></a></td>
   </tr>
   <tr>
     <td><a href="https://github.com/bowang-lab/MedSAM2/blob/main/app.py" target="_blank"><img src="https://img.shields.io/badge/Gradio-Demo-F9D371?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio App"></a></td>
     <td><a href="https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Colab-CT--Seg--Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="CT-Seg-Demo"></a></td>
     <td><a href="https://colab.research.google.com/drive/16niRHqdDZMCGV7lKuagNq_r_CEHtKY1f?usp=sharing" target="_blank"><img src="https://img.shields.io/badge/Colab-Video--Seg--Demo-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Video-Seg-Demo"></a></td>
     <td><a href="https://github.com/bowang-lab/MedSAM2?tab=readme-ov-file#bibtex" target="_blank"><img src="https://img.shields.io/badge/Paper-BibTeX-9370DB?style=for-the-badge&logoColor=white" alt="BibTeX"></a></td>
   </tr>
 </table>
</div>


## Authors

<p align="center">
  <a href="https://scholar.google.com.hk/citations?hl=en&user=bW1UV4IAAAAJ&view_op=list_works&sortby=pubdate">Jun Ma</a><sup>* 1,2</sup>, 
  <a href="https://scholar.google.com/citations?user=8IE0CfwAAAAJ&hl=en">Zongxin Yang</a><sup>* 3</sup>, 
  Sumin Kim<sup>2,4,5</sup>, 
  Bihui Chen<sup>2,4,5</sup>, 
  <a href="https://scholar.google.com.hk/citations?user=U-LgNOwAAAAJ&hl=en&oi=sra">Mohammed Baharoon</a><sup>2,3,5</sup>,<br>
  <a href="https://scholar.google.com.hk/citations?user=4qvKTooAAAAJ&hl=en&oi=sra">Adibvafa Fallahpour</a><sup>2,4,5</sup>, 
  <a href="https://scholar.google.com.hk/citations?user=UlTJ-pAAAAAJ&hl=en&oi=sra">Reza Asakereh</a><sup>4,7</sup>, 
  Hongwei Lyu<sup>4</sup>, 
  <a href="https://wanglab.ai/index.html">Bo Wang</a><sup>† 1,2,4,5,6</sup>
</p>

<p align="center">
  <sup>*</sup> Equal contribution &nbsp;&nbsp;&nbsp; <sup>†</sup> Corresponding author
</p>

<p align="center">
  <sup>1</sup>AI Collaborative Centre, University Health Network, Toronto, Canada<br>
  <sup>2</sup>Vector Institute for Artificial Intelligence, Toronto, Canada<br>
  <sup>3</sup>Department of Biomedical Informatics, Harvard Medical School, Harvard University, Boston, USA<br>
  <sup>4</sup>Peter Munk Cardiac Centre, University Health Network, Toronto, Canada<br>
  <sup>5</sup>Department of Computer Science, University of Toronto, Toronto, Canada<br>
  <sup>6</sup>Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada<br>
  <sup>7</sup>Roche Canada and Genentech
</p>

## About

[LLD-MMRI](https://github.com/LMMMEng/LLD-MMRI-Dataset) dataset contains diverse liver lesions from 498 unique patients, including hepatocellular carcinoma, intrahepatic cholangiocarcinoma, liver metastases (HM), hepatic cysts (HC), hepatic hemangioma, focal nodular hyperplasia,
and hepatic abscess. Each lesion has eight MRI scans: non-contrast, arterial, venous, delay, T2-weighted imaging, diffusionweighted imaging, T1 in-phase, and T1 out-of-phase, resulting in 3984 cases in total. 
We annotated all the 3984 lesions with [MedSAM2](https://github.com/bowang-lab/MedSAM2) in a human-in-the-loop pipeline. 

```py
# Install required package
pip install huggingface_hub

# Download the files
from huggingface_hub import snapshot_download

local_path = snapshot_download(
  repo_id="wanglab/LLD-MMRI-MedSAM2",
  repo_type="dataset",
  local_dir="./LLD-MMRI-MedSAM2"
)

# Check where data is saved
print(f"Dataset downloaded to a specific folder: {local_path}")
```
**Note.** If you are rate limited, in your terminal, use huggingface-cli login to authenticate for higher download limits.

## Citation
Please cite both LLD-MMRI and MedSAM2 when using this dataset. 

```bash
@article{LLD-MMRI,
  title={Sdr-former: A siamese dual-resolution transformer for liver lesion classification using 3d multi-phase imaging},
  author={Lou, Meng and Ying, Hanning and Liu, Xiaoqing and Zhou, Hong-Yu and Zhang, Yuqin and Yu, Yizhou},
  journal={Neural Networks},
  pages={107228},
  year={2025}
}

@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.63609},
    year={2025}
}
```