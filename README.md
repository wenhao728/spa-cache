<h1 align="center">
  SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models
</h1>

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
