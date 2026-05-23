<h1 align="center">
  [ICML 26] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models
</h1>

<p align="center">
<a href="https://arxiv.org/pdf/2602.02544"><img  src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
<a href="https://opensource.org/license/mit/"><img  src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>



<p align="center">
<a>Wenhao Sun</a>,
<a>Rong-Cheng Tu</a>,
<a>Yifu Ding</a>,
<a>Zhao Jin</a>,
<a>Jingyi Liao</a>,
<a>Yongcheng Jing</a>,
<a>Dacheng Tao</a>
<br>
<em>Nanyang Technological University</em>
</p>


> [!TIP]
> **ICML26** 
> SPA-Cache finds token updates in a cheap, low-dimensional subspace and dynamically skipping updates for stable layers. It gives DLMs a 2x to 4x speedup over existing caches and up to 8x throughput over vanilla decoding.



## 🔧 Setup
Install Pytorch, we have tested the code with PyTorch 2.7.1 and CUDA 12.8. But it should work with other versions as well. You can install PyTorch using the following command:
```
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

Install the dependencies:
```
python -m pip install -r requirements.txt
```

## 🚀 Quick Start
You can find the detailed scripts for benchmark in the `scripts` folder:
- LLaDA: [scripts/llada/eval_gsm8k.sh](scripts/llada/eval_gsm8k.sh)
- Dream: [scripts/dream/eval_gsm8k.sh](scripts/dream/eval_gsm8k.sh)


> - See the source code `scripts/<model_name>/eval.py` or use `python scripts/<model_name>/eval.py --help` command for more detailed explanations of the arguments.


## 📄 Citation
If you find our work useful, please consider citing our paper:
```bibtex
@article{sun2026spa,
  author = {Wenhao Sun and Rong{-}Cheng Tu and Yifu Ding and Zhao Jin and Jingyi Liao and Yongcheng Jing and Dacheng Tao},
  title = {SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models},
  journal = {CoRR},
  volume = {abs/2602.02544},
  year = {2026}
}
```