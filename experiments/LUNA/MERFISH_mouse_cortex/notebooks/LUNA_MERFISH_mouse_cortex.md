```python
# Pretty inline figures + autoreload
%load_ext autoreload
%autoreload 2
%config InlineBackend.figure_format = 'retina'

```


```python
ls
```

    config.yaml  [0m[01;34mgeneral[0m/     Luna_Run.ipynb  prompt.txt  [01;34mtrain[0m/
    [01;34mexperiment[0m/  __init__.py  [01;34mmodel[0m/          [01;34mtest[0m/       [01;34mvalidation[0m/



```python
%pip install --upgrade pip

# match your Torch 2.0.1 + CUDA 11.8
%pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# main PyG metapackage
%pip install torch_geometric

# LUNA imports this unconditionally; keep it disabled via env vars
%pip install wandb

# (if lightning/hydra not present)
%pip install pytorch-lightning==2.2.5 hydra-core==1.3.2 omegaconf==2.3.0 scipy==1.9.1

```

    Requirement already satisfied: pip in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (22.0.2)
    Collecting pip
      Using cached pip-25.2-py3-none-any.whl (1.8 MB)
    Installing collected packages: pip
      Attempting uninstall: pip
        Found existing installation: pip 22.0.2
        Uninstalling pip-22.0.2:
          Successfully uninstalled pip-22.0.2
    Successfully installed pip-25.2
    Note: you may need to restart the kernel to use updated packages.
    Looking in links: https://data.pyg.org/whl/torch-2.0.1+cu118.html
    Collecting pyg_lib
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcu118/pyg_lib-0.4.0%2Bpt20cu118-cp310-cp310-linux_x86_64.whl (2.6 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.6/2.6 MB[0m [31m43.4 MB/s[0m  [33m0:00:00[0m
    [?25hCollecting torch_scatter
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_scatter-2.1.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl (10.2 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m10.2/10.2 MB[0m [31m12.1 MB/s[0m  [33m0:00:01[0m[31m11.5 MB/s[0m eta [36m0:00:01[0m
    [?25hCollecting torch_sparse
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_sparse-0.6.18%2Bpt20cu118-cp310-cp310-linux_x86_64.whl (4.9 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m7.3 MB/s[0m  [33m0:00:01[0m7.6 MB/s[0m eta [36m0:00:01[0m:01[0m
    [?25hCollecting torch_cluster
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_cluster-1.6.3%2Bpt20cu118-cp310-cp310-linux_x86_64.whl (3.3 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.3/3.3 MB[0m [31m4.9 MB/s[0m  [33m0:00:00[0mm [31m5.9 MB/s[0m eta [36m0:00:01[0m
    [?25hCollecting torch_spline_conv
      Downloading https://data.pyg.org/whl/torch-2.0.0%2Bcu118/torch_spline_conv-1.2.2%2Bpt20cu118-cp310-cp310-linux_x86_64.whl (886 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m886.6/886.6 kB[0m [31m3.7 MB/s[0m  [33m0:00:00[0mm [31m?[0m eta [36m-:--:--[0m
    [?25hRequirement already satisfied: scipy in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_sparse) (1.9.1)
    Requirement already satisfied: numpy<1.25.0,>=1.18.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scipy->torch_sparse) (1.24.4)
    Installing collected packages: torch_spline_conv, torch_scatter, pyg_lib, torch_sparse, torch_cluster
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5/5[0m [torch_cluster]m [32m4/5[0m [torch_cluster]
    [1A[2KSuccessfully installed pyg_lib-0.4.0+pt20cu118 torch_cluster-1.6.3+pt20cu118 torch_scatter-2.1.2+pt20cu118 torch_sparse-0.6.18+pt20cu118 torch_spline_conv-1.2.2+pt20cu118
    Note: you may need to restart the kernel to use updated packages.
    Collecting torch_geometric
      Downloading torch_geometric-2.6.1-py3-none-any.whl.metadata (63 kB)
    Requirement already satisfied: aiohttp in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (3.12.15)
    Requirement already satisfied: fsspec in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (2025.7.0)
    Requirement already satisfied: jinja2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (3.1.4)
    Requirement already satisfied: numpy in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (1.24.4)
    Requirement already satisfied: psutil>=5.8.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (7.0.0)
    Collecting pyparsing (from torch_geometric)
      Using cached pyparsing-3.2.3-py3-none-any.whl.metadata (5.0 kB)
    Requirement already satisfied: requests in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (2.32.5)
    Requirement already satisfied: tqdm in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch_geometric) (4.67.1)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.4.0)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (5.0.1)
    Requirement already satisfied: attrs>=17.3.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (6.6.4)
    Requirement already satisfied: propcache>=0.2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp->torch_geometric) (1.20.1)
    Requirement already satisfied: typing-extensions>=4.1.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from multidict<7.0,>=4.5->aiohttp->torch_geometric) (4.12.2)
    Requirement already satisfied: idna>=2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from yarl<2.0,>=1.17.0->aiohttp->torch_geometric) (3.4)
    Requirement already satisfied: MarkupSafe>=2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from jinja2->torch_geometric) (2.1.5)
    Requirement already satisfied: charset_normalizer<4,>=2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests->torch_geometric) (2.1.1)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests->torch_geometric) (1.26.13)
    Requirement already satisfied: certifi>=2017.4.17 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests->torch_geometric) (2022.12.7)
    Downloading torch_geometric-2.6.1-py3-none-any.whl (1.1 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m27.7 MB/s[0m  [33m0:00:00[0m
    [?25hUsing cached pyparsing-3.2.3-py3-none-any.whl (111 kB)
    Installing collected packages: pyparsing, torch_geometric
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2/2[0m [torch_geometric][32m1/2[0m [torch_geometric]
    [1A[2KSuccessfully installed pyparsing-3.2.3 torch_geometric-2.6.1
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: wandb in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (0.21.1)
    Requirement already satisfied: click!=8.0.0,>=7.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (8.2.1)
    Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (3.1.45)
    Requirement already satisfied: packaging in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (25.0)
    Requirement already satisfied: platformdirs in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (4.3.8)
    Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (6.32.0)
    Requirement already satisfied: pydantic<3 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.11.7)
    Requirement already satisfied: pyyaml in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (6.0.2)
    Requirement already satisfied: requests<3,>=2.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.32.5)
    Requirement already satisfied: sentry-sdk>=2.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.35.0)
    Requirement already satisfied: typing-extensions<5,>=4.8 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (4.12.2)
    Requirement already satisfied: annotated-types>=0.6.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (0.7.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (0.4.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2.1.1)
    Requirement already satisfied: idna<4,>=2.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (1.26.13)
    Requirement already satisfied: certifi>=2017.4.17 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2022.12.7)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb) (5.0.2)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: pytorch-lightning==2.2.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (2.2.5)
    Requirement already satisfied: hydra-core==1.3.2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (1.3.2)
    Requirement already satisfied: omegaconf==2.3.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (2.3.0)
    Requirement already satisfied: scipy==1.9.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (1.9.1)
    Requirement already satisfied: numpy>=1.17.2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (1.24.4)
    Requirement already satisfied: torch>=1.13.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (2.0.1+cu118)
    Requirement already satisfied: tqdm>=4.57.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (4.67.1)
    Requirement already satisfied: PyYAML>=5.4 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (6.0.2)
    Requirement already satisfied: fsspec>=2022.5.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (2025.7.0)
    Requirement already satisfied: torchmetrics>=0.7.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (1.8.1)
    Requirement already satisfied: packaging>=20.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (25.0)
    Requirement already satisfied: typing-extensions>=4.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (4.12.2)
    Requirement already satisfied: lightning-utilities>=0.8.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pytorch-lightning==2.2.5) (0.15.2)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from hydra-core==1.3.2) (4.9.3)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (3.12.15)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (1.4.0)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (5.0.1)
    Requirement already satisfied: attrs>=17.3.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (6.6.4)
    Requirement already satisfied: propcache>=0.2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (1.20.1)
    Requirement already satisfied: idna>=2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from yarl<2.0,>=1.17.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2022.5.0->pytorch-lightning==2.2.5) (3.4)
    Requirement already satisfied: setuptools in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from lightning-utilities>=0.8.0->pytorch-lightning==2.2.5) (59.6.0)
    Requirement already satisfied: filelock in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch>=1.13.0->pytorch-lightning==2.2.5) (3.13.1)
    Requirement already satisfied: sympy in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch>=1.13.0->pytorch-lightning==2.2.5) (1.13.3)
    Requirement already satisfied: networkx in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch>=1.13.0->pytorch-lightning==2.2.5) (3.3)
    Requirement already satisfied: jinja2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch>=1.13.0->pytorch-lightning==2.2.5) (3.1.4)
    Requirement already satisfied: triton==2.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch>=1.13.0->pytorch-lightning==2.2.5) (2.0.0)
    Requirement already satisfied: cmake in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from triton==2.0.0->torch>=1.13.0->pytorch-lightning==2.2.5) (3.25.0)
    Requirement already satisfied: lit in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from triton==2.0.0->torch>=1.13.0->pytorch-lightning==2.2.5) (15.0.7)
    Requirement already satisfied: MarkupSafe>=2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from jinja2->torch>=1.13.0->pytorch-lightning==2.2.5) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from sympy->torch>=1.13.0->pytorch-lightning==2.2.5) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.



```python

```


```python
%pip install torch_geometric
```


```python
import os, socket, torch, sys
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["HYDRA_FULL_ERROR"] = "1"  # nicer tracebacks from hydra

print("Host:", socket.gethostname())
print("Python:", sys.version.split()[0])
print("CUDA visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

```

    Host: farm22-gpu0306
    Python: 3.10.12
    CUDA visible: True
    GPU: NVIDIA H100 80GB HBM3



```python
%pip install scanpy wandb colorcet squidpy hydra-core linear_attention_transformer

```

    Collecting scanpy
      Using cached scanpy-1.11.4-py3-none-any.whl.metadata (9.2 kB)
    Requirement already satisfied: wandb in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (0.21.1)
    Collecting colorcet
      Using cached colorcet-3.1.0-py3-none-any.whl.metadata (6.3 kB)
    Collecting squidpy
      Using cached squidpy-1.6.5-py3-none-any.whl.metadata (9.0 kB)
    Requirement already satisfied: hydra-core in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (1.3.2)
    Collecting linear_attention_transformer
      Using cached linear_attention_transformer-0.19.1-py3-none-any.whl.metadata (787 bytes)
    Collecting anndata>=0.8 (from scanpy)
      Using cached anndata-0.11.4-py3-none-any.whl.metadata (9.3 kB)
    Collecting h5py>=3.7.0 (from scanpy)
      Using cached h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.7 kB)
    Collecting joblib (from scanpy)
      Using cached joblib-1.5.1-py3-none-any.whl.metadata (5.6 kB)
    Collecting legacy-api-wrap>=1.4.1 (from scanpy)
      Using cached legacy_api_wrap-1.4.1-py3-none-any.whl.metadata (2.1 kB)
    Collecting matplotlib>=3.7.5 (from scanpy)
      Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
    Collecting natsort (from scanpy)
      Using cached natsort-8.4.0-py3-none-any.whl.metadata (21 kB)
    Requirement already satisfied: networkx>=2.7.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (3.3)
    Collecting numba>=0.57.1 (from scanpy)
      Using cached numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.8 kB)
    Requirement already satisfied: numpy>=1.24.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (1.24.4)
    Requirement already satisfied: packaging>=21.3 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (25.0)
    Requirement already satisfied: pandas>=1.5.3 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (2.3.2)
    Collecting patsy!=1.0.0 (from scanpy)
      Using cached patsy-1.0.1-py2.py3-none-any.whl.metadata (3.3 kB)
    Collecting pynndescent>=0.5.13 (from scanpy)
      Using cached pynndescent-0.5.13-py3-none-any.whl.metadata (6.8 kB)
    Collecting scikit-learn>=1.1.3 (from scanpy)
      Using cached scikit_learn-1.7.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
    Requirement already satisfied: scipy>=1.8.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (1.9.1)
    Collecting seaborn>=0.13.2 (from scanpy)
      Using cached seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting session-info2 (from scanpy)
      Downloading session_info2-0.2.1-py3-none-any.whl.metadata (3.4 kB)
    Collecting statsmodels>=0.14.5 (from scanpy)
      Using cached statsmodels-0.14.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (9.5 kB)
    Requirement already satisfied: tqdm in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (4.67.1)
    Requirement already satisfied: typing-extensions in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from scanpy) (4.12.2)
    Collecting umap-learn>=0.5.6 (from scanpy)
      Using cached umap_learn-0.5.9.post2-py3-none-any.whl.metadata (25 kB)
    Requirement already satisfied: click!=8.0.0,>=7.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (8.2.1)
    Requirement already satisfied: gitpython!=3.1.29,>=1.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (3.1.45)
    Requirement already satisfied: platformdirs in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (4.3.8)
    Requirement already satisfied: protobuf!=4.21.0,!=5.28.0,<7,>=3.19.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (6.32.0)
    Requirement already satisfied: pydantic<3 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.11.7)
    Requirement already satisfied: pyyaml in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (6.0.2)
    Requirement already satisfied: requests<3,>=2.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.32.5)
    Requirement already satisfied: sentry-sdk>=2.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from wandb) (2.35.0)
    Requirement already satisfied: annotated-types>=0.6.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (0.7.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pydantic<3->wandb) (0.4.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2.1.1)
    Requirement already satisfied: idna<4,>=2.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (1.26.13)
    Requirement already satisfied: certifi>=2017.4.17 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from requests<3,>=2.0.0->wandb) (2022.12.7)
    Requirement already satisfied: aiohttp>=3.8.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from squidpy) (3.12.15)
    Collecting cycler>=0.11.0 (from squidpy)
      Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting dask-image>=0.5.0 (from squidpy)
      Using cached dask_image-2024.5.3-py3-none-any.whl.metadata (2.8 kB)
    Collecting dask<=2024.11.2,>=2021.02.0 (from dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached dask-2024.11.2-py3-none-any.whl.metadata (3.7 kB)
    Collecting docrep>=0.3.1 (from squidpy)
      Using cached docrep-0.3.2-py3-none-any.whl
    Requirement already satisfied: fsspec>=2021.11.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from squidpy) (2025.7.0)
    Collecting matplotlib-scalebar>=0.8.0 (from squidpy)
      Using cached matplotlib_scalebar-0.9.0-py3-none-any.whl.metadata (18 kB)
    Collecting omnipath>=1.0.7 (from squidpy)
      Using cached omnipath-1.0.12-py3-none-any.whl.metadata (7.0 kB)
    Requirement already satisfied: pillow>=8.0.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from squidpy) (11.0.0)
    Collecting scikit-image>=0.20 (from squidpy)
      Using cached scikit_image-0.25.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (14 kB)
    Collecting spatialdata>=0.2.5 (from squidpy)
      Downloading spatialdata-0.5.0-py3-none-any.whl.metadata (10 kB)
    Collecting tifffile!=2022.4.22 (from squidpy)
      Using cached tifffile-2025.5.10-py3-none-any.whl.metadata (31 kB)
    Collecting validators>=0.18.2 (from squidpy)
      Using cached validators-0.35.0-py3-none-any.whl.metadata (3.9 kB)
    Collecting xarray>=2024.10.0 (from squidpy)
      Using cached xarray-2025.6.1-py3-none-any.whl.metadata (12 kB)
    Collecting zarr<3.0.0,>=2.6.1 (from squidpy)
      Using cached zarr-2.18.3-py3-none-any.whl.metadata (5.7 kB)
    Collecting cloudpickle>=3.0.0 (from dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached cloudpickle-3.1.1-py3-none-any.whl.metadata (7.1 kB)
    Collecting partd>=1.4.0 (from dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached partd-1.4.2-py3-none-any.whl.metadata (4.6 kB)
    Collecting toolz>=0.10.0 (from dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached toolz-1.0.0-py3-none-any.whl.metadata (5.1 kB)
    Collecting importlib-metadata>=4.13.0 (from dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached importlib_metadata-8.7.0-py3-none-any.whl.metadata (4.8 kB)
    Collecting asciitree (from zarr<3.0.0,>=2.6.1->squidpy)
      Using cached asciitree-0.3.3-py3-none-any.whl
    Collecting numcodecs>=0.10.0 (from zarr<3.0.0,>=2.6.1->squidpy)
      Using cached numcodecs-0.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.9 kB)
    Collecting fasteners (from zarr<3.0.0,>=2.6.1->squidpy)
      Downloading fasteners-0.20-py3-none-any.whl.metadata (4.8 kB)
    Requirement already satisfied: omegaconf<2.4,>=2.2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from hydra-core) (2.3.0)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from hydra-core) (4.9.3)
    Collecting axial-positional-embedding (from linear_attention_transformer)
      Using cached axial_positional_embedding-0.3.12-py3-none-any.whl.metadata (4.3 kB)
    Collecting einops (from linear_attention_transformer)
      Using cached einops-0.8.1-py3-none-any.whl.metadata (13 kB)
    Collecting linformer>=0.1.0 (from linear_attention_transformer)
      Using cached linformer-0.2.3-py3-none-any.whl.metadata (602 bytes)
    Collecting local-attention (from linear_attention_transformer)
      Using cached local_attention-1.11.2-py3-none-any.whl.metadata (929 bytes)
    Collecting product-key-memory>=0.1.5 (from linear_attention_transformer)
      Downloading product_key_memory-0.2.11-py3-none-any.whl.metadata (717 bytes)
    Requirement already satisfied: torch in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from linear_attention_transformer) (2.0.1+cu118)
    Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (2.6.1)
    Requirement already satisfied: aiosignal>=1.4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (1.4.0)
    Requirement already satisfied: async-timeout<6.0,>=4.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (5.0.1)
    Requirement already satisfied: attrs>=17.3.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (25.3.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (1.7.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (6.6.4)
    Requirement already satisfied: propcache>=0.2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (0.3.2)
    Requirement already satisfied: yarl<2.0,>=1.17.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from aiohttp>=3.8.1->squidpy) (1.20.1)
    Collecting array-api-compat!=1.5,>1.4 (from anndata>=0.8->scanpy)
      Using cached array_api_compat-1.12.0-py3-none-any.whl.metadata (2.5 kB)
    Requirement already satisfied: exceptiongroup in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from anndata>=0.8->scanpy) (1.3.0)
    Collecting pims>=0.4.1 (from dask-image>=0.5.0->squidpy)
      Using cached PIMS-0.7-py3-none-any.whl
    Collecting dask-expr<1.2,>=1.1 (from dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached dask_expr-1.1.21-py3-none-any.whl.metadata (2.6 kB)
    INFO: pip is looking at multiple versions of dask-expr to determine which version is compatible with other requirements. This could take a while.
      Using cached dask_expr-1.1.20-py3-none-any.whl.metadata (2.6 kB)
      Using cached dask_expr-1.1.19-py3-none-any.whl.metadata (2.6 kB)
    Collecting pyarrow>=14.0.1 (from dask-expr<1.2,>=1.1->dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached pyarrow-21.0.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (3.3 kB)
    Requirement already satisfied: six in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from docrep>=0.3.1->squidpy) (1.17.0)
    Requirement already satisfied: gitdb<5,>=4.0.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from gitpython!=3.1.29,>=1.0.0->wandb) (4.0.12)
    Requirement already satisfied: smmap<6,>=3.0.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.29,>=1.0.0->wandb) (5.0.2)
    Collecting zipp>=3.20 (from importlib-metadata>=4.13.0->dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
    Collecting contourpy>=1.0.1 (from matplotlib>=3.7.5->scanpy)
      Using cached contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.5 kB)
    Collecting fonttools>=4.22.0 (from matplotlib>=3.7.5->scanpy)
      Using cached fonttools-4.59.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (108 kB)
    Collecting kiwisolver>=1.3.1 (from matplotlib>=3.7.5->scanpy)
      Using cached kiwisolver-1.4.9-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl.metadata (6.3 kB)
    Requirement already satisfied: pyparsing>=2.3.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from matplotlib>=3.7.5->scanpy) (3.2.3)
    Requirement already satisfied: python-dateutil>=2.7 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from matplotlib>=3.7.5->scanpy) (2.9.0.post0)
    Collecting llvmlite<0.45,>=0.44.0dev0 (from numba>=0.57.1->scanpy)
      Using cached llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.8 kB)
    Collecting inflect>=4.1.0 (from omnipath>=1.0.7->squidpy)
      Using cached inflect-7.5.0-py3-none-any.whl.metadata (24 kB)
    Collecting wrapt>=1.12.0 (from omnipath>=1.0.7->squidpy)
      Downloading wrapt-1.17.3-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (6.4 kB)
    Collecting more_itertools>=8.5.0 (from inflect>=4.1.0->omnipath>=1.0.7->squidpy)
      Using cached more_itertools-10.7.0-py3-none-any.whl.metadata (37 kB)
    Collecting typeguard>=4.0.1 (from inflect>=4.1.0->omnipath>=1.0.7->squidpy)
      Using cached typeguard-4.4.4-py3-none-any.whl.metadata (3.3 kB)
    Requirement already satisfied: pytz>=2020.1 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pandas>=1.5.3->scanpy) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from pandas>=1.5.3->scanpy) (2025.2)
    Collecting locket (from partd>=1.4.0->dask<=2024.11.2,>=2021.02.0->dask[array]<=2024.11.2,>=2021.02.0->squidpy)
      Using cached locket-1.0.0-py2.py3-none-any.whl.metadata (2.8 kB)
    Collecting imageio (from pims>=0.4.1->dask-image>=0.5.0->squidpy)
      Using cached imageio-2.37.0-py3-none-any.whl.metadata (5.2 kB)
    Collecting slicerator>=0.9.8 (from pims>=0.4.1->dask-image>=0.5.0->squidpy)
      Using cached slicerator-1.1.0-py3-none-any.whl.metadata (1.9 kB)
    Collecting colt5-attention>=0.10.14 (from product-key-memory>=0.1.5->linear_attention_transformer)
      Downloading CoLT5_attention-0.11.1-py3-none-any.whl.metadata (737 bytes)
    Collecting hyper-connections>=0.1.8 (from local-attention->linear_attention_transformer)
      Downloading hyper_connections-0.2.1-py3-none-any.whl.metadata (6.0 kB)
    Collecting torch (from linear_attention_transformer)
      Using cached torch-2.8.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (30 kB)
    Collecting scipy>=1.8.1 (from scanpy)
      Using cached scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
    Collecting lazy-loader>=0.4 (from scikit-image>=0.20->squidpy)
      Using cached lazy_loader-0.4-py3-none-any.whl.metadata (7.6 kB)
    Collecting threadpoolctl>=3.1.0 (from scikit-learn>=1.1.3->scanpy)
      Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
    Collecting datashader (from spatialdata>=0.2.5->squidpy)
      Downloading datashader-0.18.2-py3-none-any.whl.metadata (7.6 kB)
    Collecting geopandas>=0.14 (from spatialdata>=0.2.5->squidpy)
      Using cached geopandas-1.1.1-py3-none-any.whl.metadata (2.3 kB)
    Collecting multiscale-spatial-image>=2.0.3 (from spatialdata>=0.2.5->squidpy)
      Downloading multiscale_spatial_image-2.0.3-py3-none-any.whl.metadata (25 kB)
    Collecting ome-zarr>=0.8.4 (from spatialdata>=0.2.5->squidpy)
      Downloading ome_zarr-0.12.2-py3-none-any.whl.metadata (2.9 kB)
    Collecting pooch (from spatialdata>=0.2.5->squidpy)
      Using cached pooch-1.8.2-py3-none-any.whl.metadata (10 kB)
    Collecting rich (from spatialdata>=0.2.5->squidpy)
      Using cached rich-14.1.0-py3-none-any.whl.metadata (18 kB)
    Requirement already satisfied: setuptools in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from spatialdata>=0.2.5->squidpy) (59.6.0)
    Collecting shapely>=2.0.1 (from spatialdata>=0.2.5->squidpy)
      Using cached shapely-2.1.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.7 kB)
    Collecting spatial-image>=1.2.3 (from spatialdata>=0.2.5->squidpy)
      Downloading spatial_image-1.2.3-py3-none-any.whl.metadata (7.0 kB)
    Collecting xarray-schema (from spatialdata>=0.2.5->squidpy)
      Using cached xarray_schema-0.0.3-py3-none-any.whl.metadata (4.3 kB)
    Collecting xarray-spatial>=0.3.5 (from spatialdata>=0.2.5->squidpy)
      Using cached xarray_spatial-0.4.0-py3-none-any.whl.metadata (16 kB)
    Collecting pyogrio>=0.7.2 (from geopandas>=0.14->spatialdata>=0.2.5->squidpy)
      Downloading pyogrio-0.11.1-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (5.3 kB)
    Collecting pyproj>=3.5.0 (from geopandas>=0.14->spatialdata>=0.2.5->squidpy)
      Using cached pyproj-3.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (31 kB)
    Collecting xarray-dataclass>=3.0.0 (from multiscale-spatial-image>=2.0.3->spatialdata>=0.2.5->squidpy)
      Using cached xarray_dataclass-3.0.0-py3-none-any.whl.metadata (14 kB)
    INFO: pip is looking at multiple versions of ome-zarr to determine which version is compatible with other requirements. This could take a while.
    Collecting ome-zarr>=0.8.4 (from spatialdata>=0.2.5->squidpy)
      Downloading ome_zarr-0.12.1-py3-none-any.whl.metadata (2.9 kB)
      Downloading ome_zarr-0.12.0-py3-none-any.whl.metadata (2.9 kB)
      Using cached ome_zarr-0.11.1-py3-none-any.whl.metadata (2.9 kB)
    Collecting s3fs (from fsspec[s3]!=2021.07.0,!=2023.9.0,>=0.8->ome-zarr>=0.8.4->spatialdata>=0.2.5->squidpy)
      Using cached s3fs-2025.7.0-py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: filelock in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch->linear_attention_transformer) (3.13.1)
    Requirement already satisfied: sympy>=1.13.3 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch->linear_attention_transformer) (1.13.3)
    Requirement already satisfied: jinja2 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from torch->linear_attention_transformer) (3.1.4)
    Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch->linear_attention_transformer)
      Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch->linear_attention_transformer)
      Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch->linear_attention_transformer)
      Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch->linear_attention_transformer)
      Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-cublas-cu12==12.8.4.1 (from torch->linear_attention_transformer)
      Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cufft-cu12==11.3.3.83 (from torch->linear_attention_transformer)
      Using cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-curand-cu12==10.3.9.90 (from torch->linear_attention_transformer)
      Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch->linear_attention_transformer)
      Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch->linear_attention_transformer)
      Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-cusparselt-cu12==0.7.1 (from torch->linear_attention_transformer)
      Downloading nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
    Collecting nvidia-nccl-cu12==2.27.3 (from torch->linear_attention_transformer)
      Downloading nvidia_nccl_cu12-2.27.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
    Collecting nvidia-nvtx-cu12==12.8.90 (from torch->linear_attention_transformer)
      Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
    Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch->linear_attention_transformer)
      Using cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
    Collecting nvidia-cufile-cu12==1.13.1.3 (from torch->linear_attention_transformer)
      Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
    Collecting triton==3.4.0 (from torch->linear_attention_transformer)
      Downloading triton-3.4.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from sympy>=1.13.3->torch->linear_attention_transformer) (1.3.0)
    Collecting typing-extensions (from scanpy)
      Using cached typing_extensions-4.14.1-py3-none-any.whl.metadata (3.0 kB)
    Collecting numpy>=1.24.1 (from scanpy)
      Using cached numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
    Collecting multipledispatch (from datashader->spatialdata>=0.2.5->squidpy)
      Using cached multipledispatch-1.0.0-py3-none-any.whl.metadata (3.8 kB)
    Collecting param (from datashader->spatialdata>=0.2.5->squidpy)
      Using cached param-2.2.1-py3-none-any.whl.metadata (6.6 kB)
    Collecting pyct (from datashader->spatialdata>=0.2.5->squidpy)
      Using cached pyct-0.5.0-py2.py3-none-any.whl.metadata (7.4 kB)
    Requirement already satisfied: MarkupSafe>=2.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from jinja2->torch->linear_attention_transformer) (2.1.5)
    Collecting markdown-it-py>=2.2.0 (from rich->spatialdata>=0.2.5->squidpy)
      Downloading markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages (from rich->spatialdata>=0.2.5->squidpy) (2.19.2)
    Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->spatialdata>=0.2.5->squidpy)
      Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Collecting aiobotocore<3.0.0,>=2.5.4 (from s3fs->fsspec[s3]!=2021.07.0,!=2023.9.0,>=0.8->ome-zarr>=0.8.4->spatialdata>=0.2.5->squidpy)
      Downloading aiobotocore-2.24.1-py3-none-any.whl.metadata (25 kB)
    Collecting aioitertools<1.0.0,>=0.5.1 (from aiobotocore<3.0.0,>=2.5.4->s3fs->fsspec[s3]!=2021.07.0,!=2023.9.0,>=0.8->ome-zarr>=0.8.4->spatialdata>=0.2.5->squidpy)
      Using cached aioitertools-0.12.0-py3-none-any.whl.metadata (3.8 kB)
    Collecting botocore<1.39.12,>=1.39.9 (from aiobotocore<3.0.0,>=2.5.4->s3fs->fsspec[s3]!=2021.07.0,!=2023.9.0,>=0.8->ome-zarr>=0.8.4->spatialdata>=0.2.5->squidpy)
      Downloading botocore-1.39.11-py3-none-any.whl.metadata (5.7 kB)
    Collecting jmespath<2.0.0,>=0.7.1 (from aiobotocore<3.0.0,>=2.5.4->s3fs->fsspec[s3]!=2021.07.0,!=2023.9.0,>=0.8->ome-zarr>=0.8.4->spatialdata>=0.2.5->squidpy)
      Using cached jmespath-1.0.1-py3-none-any.whl.metadata (7.6 kB)
    Using cached scanpy-1.11.4-py3-none-any.whl (2.1 MB)
    Using cached colorcet-3.1.0-py3-none-any.whl (260 kB)
    Using cached squidpy-1.6.5-py3-none-any.whl (161 kB)
    Using cached dask-2024.11.2-py3-none-any.whl (1.3 MB)
    Using cached zarr-2.18.3-py3-none-any.whl (210 kB)
    Downloading linear_attention_transformer-0.19.1-py3-none-any.whl (12 kB)
    Using cached anndata-0.11.4-py3-none-any.whl (144 kB)
    Using cached array_api_compat-1.12.0-py3-none-any.whl (58 kB)
    Using cached cloudpickle-3.1.1-py3-none-any.whl (20 kB)
    Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Using cached dask_image-2024.5.3-py3-none-any.whl (59 kB)
    Using cached dask_expr-1.1.19-py3-none-any.whl (244 kB)
    Using cached h5py-3.14.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
    Using cached importlib_metadata-8.7.0-py3-none-any.whl (27 kB)
    Using cached legacy_api_wrap-1.4.1-py3-none-any.whl (10.0 kB)
    Downloading linformer-0.2.3-py3-none-any.whl (6.2 kB)
    Downloading matplotlib-3.10.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (8.7 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m8.7/8.7 MB[0m [31m90.6 MB/s[0m  [33m0:00:00[0m
    [?25hUsing cached contourpy-1.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (325 kB)
    Using cached fonttools-4.59.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (4.8 MB)
    Using cached kiwisolver-1.4.9-cp310-cp310-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (1.6 MB)
    Using cached matplotlib_scalebar-0.9.0-py3-none-any.whl (16 kB)
    Using cached numba-0.61.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (3.8 MB)
    Using cached llvmlite-0.44.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (42.4 MB)
    Using cached numcodecs-0.13.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (8.6 MB)
    Using cached omnipath-1.0.12-py3-none-any.whl (51 kB)
    Using cached inflect-7.5.0-py3-none-any.whl (35 kB)
    Using cached more_itertools-10.7.0-py3-none-any.whl (65 kB)
    Using cached partd-1.4.2-py3-none-any.whl (18 kB)
    Using cached patsy-1.0.1-py2.py3-none-any.whl (232 kB)
    Downloading product_key_memory-0.2.11-py3-none-any.whl (6.5 kB)
    Downloading CoLT5_attention-0.11.1-py3-none-any.whl (18 kB)
    Downloading einops-0.8.1-py3-none-any.whl (64 kB)
    Downloading local_attention-1.11.2-py3-none-any.whl (9.5 kB)
    Downloading hyper_connections-0.2.1-py3-none-any.whl (16 kB)
    Using cached pyarrow-21.0.0-cp310-cp310-manylinux_2_28_x86_64.whl (42.7 MB)
    Using cached pynndescent-0.5.13-py3-none-any.whl (56 kB)
    Using cached joblib-1.5.1-py3-none-any.whl (307 kB)
    Using cached scikit_image-0.25.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (14.8 MB)
    Using cached imageio-2.37.0-py3-none-any.whl (315 kB)
    Using cached lazy_loader-0.4-py3-none-any.whl (12 kB)
    Using cached scikit_learn-1.7.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (9.7 MB)
    Using cached scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.7 MB)
    Using cached seaborn-0.13.2-py3-none-any.whl (294 kB)
    Using cached slicerator-1.1.0-py3-none-any.whl (10 kB)
    Downloading spatialdata-0.5.0-py3-none-any.whl (185 kB)
    Using cached geopandas-1.1.1-py3-none-any.whl (338 kB)
    Downloading multiscale_spatial_image-2.0.3-py3-none-any.whl (29 kB)
    Using cached ome_zarr-0.11.1-py3-none-any.whl (40 kB)
    Downloading pyogrio-0.11.1-cp310-cp310-manylinux_2_28_x86_64.whl (27.5 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m27.5/27.5 MB[0m [31m90.9 MB/s[0m  [33m0:00:00[0ms[0m eta [36m0:00:01[0m
    [?25hUsing cached pyproj-3.7.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (9.3 MB)
    Using cached shapely-2.1.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.1 MB)
    Downloading spatial_image-1.2.3-py3-none-any.whl (8.7 kB)
    Using cached statsmodels-0.14.5-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (10.7 MB)
    Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
    Using cached tifffile-2025.5.10-py3-none-any.whl (226 kB)
    Using cached toolz-1.0.0-py3-none-any.whl (56 kB)
    Using cached torch-2.8.0-cp310-cp310-manylinux_2_28_x86_64.whl (888.0 MB)
    Using cached nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl (594.3 MB)
    Using cached nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (10.2 MB)
    Using cached nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (88.0 MB)
    Using cached nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (954 kB)
    Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl (706.8 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m706.8/706.8 MB[0m [31m72.0 MB/s[0m  [33m0:00:05[0m[0m eta [36m0:00:01[0m0:01[0m:01[0m
    [?25hUsing cached nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (193.1 MB)
    Using cached nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (1.2 MB)
    Using cached nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl (63.6 MB)
    Using cached nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl (267.5 MB)
    Using cached nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (288.2 MB)
    Downloading nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl (287.2 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m287.2/287.2 MB[0m [31m163.5 MB/s[0m  [33m0:00:01[0m[0m eta [36m0:00:01[0m0:01[0m:01[0m
    [?25hDownloading nvidia_nccl_cu12-2.27.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (322.4 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m322.4/322.4 MB[0m [31m145.5 MB/s[0m  [33m0:00:02[0m[0m eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hUsing cached nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl (39.3 MB)
    Using cached nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (89 kB)
    Downloading triton-3.4.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (155.4 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m155.4/155.4 MB[0m [31m92.8 MB/s[0m  [33m0:00:01[0m[0m eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hUsing cached typeguard-4.4.4-py3-none-any.whl (34 kB)
    Using cached typing_extensions-4.14.1-py3-none-any.whl (43 kB)
    Using cached umap_learn-0.5.9.post2-py3-none-any.whl (90 kB)
    Using cached validators-0.35.0-py3-none-any.whl (44 kB)
    Downloading wrapt-1.17.3-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl (81 kB)
    Downloading xarray-2025.6.1-py3-none-any.whl (1.3 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m35.1 MB/s[0m  [33m0:00:00[0m
    [?25hUsing cached xarray_dataclass-3.0.0-py3-none-any.whl (16 kB)
    Using cached numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.8 MB)
    Using cached xarray_spatial-0.4.0-py3-none-any.whl (2.0 MB)
    Downloading datashader-0.18.2-py3-none-any.whl (18.3 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m18.3/18.3 MB[0m [31m95.6 MB/s[0m  [33m0:00:00[0m
    [?25hUsing cached zipp-3.23.0-py3-none-any.whl (10 kB)
    Downloading axial_positional_embedding-0.3.12-py3-none-any.whl (6.7 kB)
    Downloading fasteners-0.20-py3-none-any.whl (18 kB)
    Using cached locket-1.0.0-py2.py3-none-any.whl (4.4 kB)
    Using cached multipledispatch-1.0.0-py3-none-any.whl (12 kB)
    Using cached natsort-8.4.0-py3-none-any.whl (38 kB)
    Using cached param-2.2.1-py3-none-any.whl (119 kB)
    Using cached pooch-1.8.2-py3-none-any.whl (64 kB)
    Using cached pyct-0.5.0-py2.py3-none-any.whl (15 kB)
    Using cached rich-14.1.0-py3-none-any.whl (243 kB)
    Downloading markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
    Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Downloading s3fs-2025.7.0-py3-none-any.whl (30 kB)
    Downloading aiobotocore-2.24.1-py3-none-any.whl (85 kB)
    Using cached aioitertools-0.12.0-py3-none-any.whl (24 kB)
    Downloading botocore-1.39.11-py3-none-any.whl (13.9 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m13.9/13.9 MB[0m [31m97.9 MB/s[0m  [33m0:00:00[0m
    [?25hUsing cached jmespath-1.0.1-py3-none-any.whl (20 kB)
    Downloading session_info2-0.2.1-py3-none-any.whl (16 kB)
    Using cached xarray_schema-0.0.3-py3-none-any.whl (10 kB)
    Installing collected packages: slicerator, nvidia-cusparselt-cu12, multipledispatch, asciitree, zipp, wrapt, validators, typing-extensions, triton, toolz, threadpoolctl, session-info2, pyproj, pyarrow, param, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufile-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, numpy, natsort, more_itertools, mdurl, locket, llvmlite, legacy-api-wrap, lazy-loader, kiwisolver, joblib, jmespath, fonttools, fasteners, einops, docrep, cycler, colorcet, cloudpickle, array-api-compat, aioitertools, typeguard, tifffile, shapely, scipy, pyogrio, pyct, pooch, patsy, partd, nvidia-cusparse-cu12, nvidia-cufft-cu12, nvidia-cudnn-cu12, numcodecs, numba, markdown-it-py, importlib-metadata, imageio, h5py, contourpy, botocore, zarr, xarray, statsmodels, scikit-learn, scikit-image, rich, pims, nvidia-cusolver-cu12, matplotlib, inflect, geopandas, dask, anndata, xarray-schema, xarray-dataclass, torch, seaborn, pynndescent, omnipath, matplotlib-scalebar, datashader, dask-expr, xarray-spatial, umap-learn, spatial-image, linformer, hyper-connections, axial-positional-embedding, aiobotocore, scanpy, s3fs, multiscale-spatial-image, local-attention, dask-image, colt5-attention, product-key-memory, ome-zarr, spatialdata, linear_attention_transformer, squidpy
    [2K  Attempting uninstall: typing-extensions0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  6/104[0m [validators]]
    [2K    Found existing installation: typing_extensions 4.12.2━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  6/104[0m [validators]
    [2K    Uninstalling typing_extensions-4.12.2:8;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  6/104[0m [validators]
    [2K      Successfully uninstalled typing_extensions-4.12.25;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  7/104[0m [typing-extensions]
    [2K  Attempting uninstall: triton249;38;114m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  7/104[0m [typing-extensions]
    [2K    Found existing installation: triton 2.0.0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  7/104[0m [typing-extensions]
    [2K    Uninstalling triton-2.0.0:249;38;114m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  7/104[0m [typing-extensions]
    [2K      Successfully uninstalled triton-2.0.0[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m  7/104[0m [typing-extensions]
    [2K  Attempting uninstall: numpy[0m[38;2;249;38;114m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 23/104[0m [nvidia-cublas-cu12]-cu12]
    [2K    Found existing installation: numpy 1.24.44m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 23/104[0m [nvidia-cublas-cu12]
    [2K    Uninstalling numpy-1.24.4:[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 24/104[0m [numpy]dia-cublas-cu12]
    [2K      Successfully uninstalled numpy-1.24.47m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 24/104[0m [numpy]
    [2K   [38;2;249;38;114m━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 24/104[0m [numpy][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~umpy.libs'.
      You can safely remove it manually.[0m[33m
    [2K   [38;2;249;38;114m━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m 24/104[0m [numpy][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~umpy'.
      You can safely remove it manually.[0m[33m
    [2K  Attempting uninstall: scipy━━━━━━━━━[0m[38;2;249;38;114m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━[0m [32m 46/104[0m [shapely]d]ls]ap]
    [2K    Found existing installation: scipy 1.9.1;249;38;114m╸[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━━[0m [32m 46/104[0m [shapely]
    [2K    Uninstalling scipy-1.9.1:━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━[0m [32m 47/104[0m [scipy]pely]
    [2K      Successfully uninstalled scipy-1.9.1m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━━━[0m [32m 47/104[0m [scipy]
    [2K  Attempting uninstall: torch━━━━━━━━━━━━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 78/104[0m [xarray-dataclass]]12]data]
    [2K    Found existing installation: torch 2.0.1+cu118m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 78/104[0m [xarray-dataclass]
    [2K    Uninstalling torch-2.0.1+cu118:━━━━━━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 79/104[0m [torch]aclass]
    [2K      Successfully uninstalled torch-2.0.1+cu118━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 79/104[0m [torch]
    [2K   [38;2;249;38;114m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 79/104[0m [torch][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~vfuser'.
      You can safely remove it manually.[0m[33m
    [2K   [38;2;249;38;114m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━[0m [32m 79/104[0m [torch][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~orch'.
      You can safely remove it manually.[0m[33m
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m104/104[0m [squidpy]m103/104[0m [squidpy]transformer]aldata]ion]al-image]
    [1A[2K[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    torchaudio 2.0.2+cu118 requires torch==2.0.1, but you have torch 2.8.0 which is incompatible.
    torchvision 0.15.2+cu118 requires torch==2.0.1, but you have torch 2.8.0 which is incompatible.[0m[31m
    [0mSuccessfully installed aiobotocore-2.24.1 aioitertools-0.12.0 anndata-0.11.4 array-api-compat-1.12.0 asciitree-0.3.3 axial-positional-embedding-0.3.12 botocore-1.39.11 cloudpickle-3.1.1 colorcet-3.1.0 colt5-attention-0.11.1 contourpy-1.3.2 cycler-0.12.1 dask-2024.11.2 dask-expr-1.1.19 dask-image-2024.5.3 datashader-0.18.2 docrep-0.3.2 einops-0.8.1 fasteners-0.20 fonttools-4.59.1 geopandas-1.1.1 h5py-3.14.0 hyper-connections-0.2.1 imageio-2.37.0 importlib-metadata-8.7.0 inflect-7.5.0 jmespath-1.0.1 joblib-1.5.1 kiwisolver-1.4.9 lazy-loader-0.4 legacy-api-wrap-1.4.1 linear_attention_transformer-0.19.1 linformer-0.2.3 llvmlite-0.44.0 local-attention-1.11.2 locket-1.0.0 markdown-it-py-4.0.0 matplotlib-3.10.5 matplotlib-scalebar-0.9.0 mdurl-0.1.2 more_itertools-10.7.0 multipledispatch-1.0.0 multiscale-spatial-image-2.0.3 natsort-8.4.0 numba-0.61.2 numcodecs-0.13.1 numpy-2.2.6 nvidia-cublas-cu12-12.8.4.1 nvidia-cuda-cupti-cu12-12.8.90 nvidia-cuda-nvrtc-cu12-12.8.93 nvidia-cuda-runtime-cu12-12.8.90 nvidia-cudnn-cu12-9.10.2.21 nvidia-cufft-cu12-11.3.3.83 nvidia-cufile-cu12-1.13.1.3 nvidia-curand-cu12-10.3.9.90 nvidia-cusolver-cu12-11.7.3.90 nvidia-cusparse-cu12-12.5.8.93 nvidia-cusparselt-cu12-0.7.1 nvidia-nccl-cu12-2.27.3 nvidia-nvjitlink-cu12-12.8.93 nvidia-nvtx-cu12-12.8.90 ome-zarr-0.11.1 omnipath-1.0.12 param-2.2.1 partd-1.4.2 patsy-1.0.1 pims-0.7 pooch-1.8.2 product-key-memory-0.2.11 pyarrow-21.0.0 pyct-0.5.0 pynndescent-0.5.13 pyogrio-0.11.1 pyproj-3.7.1 rich-14.1.0 s3fs-2025.7.0 scanpy-1.11.4 scikit-image-0.25.2 scikit-learn-1.7.1 scipy-1.15.3 seaborn-0.13.2 session-info2-0.2.1 shapely-2.1.1 slicerator-1.1.0 spatial-image-1.2.3 spatialdata-0.5.0 squidpy-1.6.5 statsmodels-0.14.5 threadpoolctl-3.6.0 tifffile-2025.5.10 toolz-1.0.0 torch-2.8.0 triton-3.4.0 typeguard-4.4.4 typing-extensions-4.14.1 umap-learn-0.5.9.post2 validators-0.35.0 wrapt-1.17.3 xarray-2025.6.1 xarray-dataclass-3.0.0 xarray-schema-0.0.3 xarray-spatial-0.4.0 zarr-2.18.3 zipp-3.23.0
    Note: you may need to restart the kernel to use updated packages.



```python
%pip -q install wandb
import wandb, os
print("wandb imported, WANDB_DISABLED =", os.environ.get("WANDB_DISABLED"))

```

    Note: you may need to restart the kernel to use updated packages.
    wandb imported, WANDB_DISABLED = true



```python
# --- Paths you will keep AFTER the repo clean-up below ---
import os, re, glob, json, socket, sys, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns, colorcet as cc

BASE_EXP = "/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/MERFISH_mouse_cortex"
FIG_DIR   = f"{BASE_EXP}/figs"
TABLE_DIR = f"{BASE_EXP}/tables"
CKPT_DIR  = f"{BASE_EXP}/checkpoints"
RES_TRAINTEST_DIR = f"{BASE_EXP}/results/train_and_test"
RES_TESTONLY_DIR  = f"{BASE_EXP}/results/test_only"

for d in (FIG_DIR, TABLE_DIR, CKPT_DIR, RES_TRAINTEST_DIR, RES_TESTONLY_DIR):
    os.makedirs(d, exist_ok=True)

def epoch_from_dir(d):
    m = re.search(r"epoch_(\d+)$", d)
    return int(m.group(1)) if m else None

print("Host:", socket.gethostname())
print("Python:", sys.version.split()[0])
print("BASE_EXP:", BASE_EXP)

```

    Host: farm22-gpu0306
    Python: 3.10.12
    BASE_EXP: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/MERFISH_mouse_cortex



```python
# Resolve dataset root (handles both layouts you saw)
cand1 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex"
cand2 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex/MERFISH_mouse_cortex"
root_directory = cand1 if os.path.exists(os.path.join(cand1, "MERFISH_mouse_cortex_train.csv")) else cand2

train_csv = f"{root_directory}/MERFISH_mouse_cortex_train.csv"
test_csv  = f"{root_directory}/MERFISH_mouse_cortex_test.csv"
assert os.path.exists(train_csv) and os.path.exists(test_csv), (train_csv, test_csv)

tr = pd.read_csv(train_csv, index_col=0)
te = pd.read_csv(test_csv,  index_col=0)

# What LUNA uses in your config
NUM_GENES = 254                       # 0..253 inclusive
GENE_COLS = list(tr.columns[:NUM_GENES])
REQ_COLS  = ["coord_X", "coord_Y", "cell_section", "cell_class"]

for name, df in [("train", tr), ("test", te)]:
    missing1 = set(REQ_COLS) - set(df.columns)
    missing2 = set(GENE_COLS) - set(df.columns)
    assert not missing1 and not missing2, f"{name} missing: base={missing1}, genes?={missing2}"

assert te.columns[:NUM_GENES].equals(tr.columns[:NUM_GENES]), "Train/Test gene columns mismatch"

print("Dataset OK")
print("Train:", tr.shape, "Test:", te.shape, "| Genes:", len(GENE_COLS))
print("Train sections:", tr['cell_section'].nunique(), "Test sections:", te['cell_section'].nunique())

```

    Dataset OK
    Train: (158379, 262) Test: (118036, 262) | Genes: 254
    Train sections: 33 Test sections: 31


### What is fed to the model: training step


```python
# Construct the arrays exactly as the model sees them
X_train = tr[GENE_COLS].astype("float32").to_numpy()
Y_train = tr[["coord_X","coord_Y"]].astype("float32").to_numpy()

train_summary = {
    "X_train_shape": list(X_train.shape),
    "Y_train_shape": list(Y_train.shape),
    "n_sections": int(tr["cell_section"].nunique()),
    "n_classes": int(tr["cell_class"].nunique()),
}
with open(os.path.join(TABLE_DIR, "train_feed_summary.json"), "w") as f:
    json.dump(train_summary, f, indent=2)

pd.DataFrame({
    "column": ["X (genes)", "Y (coords)"],
    "dtype":  [str(X_train.dtype), str(Y_train.dtype)],
    "shape":  [str(X_train.shape), str(Y_train.shape)]
}).to_csv(os.path.join(TABLE_DIR, "train_feed_shapes.csv"), index=False)

print(train_summary)

```

    {'X_train_shape': [158379, 254], 'Y_train_shape': [158379, 2], 'n_sections': 33, 'n_classes': 23}


### What is fed to the model: inference step


```python
# Largest test section -> this is what a single inference graph usually processes
sec_counts = te["cell_section"].value_counts()
sec_name = sec_counts.idxmax()

X_test_sec = te.loc[te["cell_section"]==sec_name, GENE_COLS].astype("float32").to_numpy()
Y_true_sec = te.loc[te["cell_section"]==sec_name, ["coord_X","coord_Y"]].astype("float32").to_numpy()

infer_summary = {
    "section": sec_name,
    "cells_in_section": int(X_test_sec.shape[0]),
    "X_test_section_shape": list(X_test_sec.shape),
    "Y_true_section_shape": list(Y_true_sec.shape),
}
with open(os.path.join(TABLE_DIR, "inference_feed_summary.json"), "w") as f:
    json.dump(infer_summary, f, indent=2)

print(infer_summary)

```

    {'section': 'mouse2_slice99', 'cells_in_section': 5235, 'X_test_section_shape': [5235, 254], 'Y_true_section_shape': [5235, 2]}



```python
# Gene stats
gene_stats = pd.DataFrame({
    "gene": GENE_COLS,
    "mean": tr[GENE_COLS].mean(0).values,
    "std":  tr[GENE_COLS].std(0).values,
    "var":  tr[GENE_COLS].var(0).values,
    "zero_frac": (tr[GENE_COLS]==0).mean(0).values,
})
gene_stats.sort_values("var", ascending=False).to_csv(os.path.join(TABLE_DIR, "gene_stats_train.csv"), index=False)

# Top-30 variable genes (bar)
top = gene_stats.sort_values("var", ascending=False).head(30)
plt.figure(figsize=(10,5))
sns.barplot(data=top, x="var", y="gene")
plt.title("Top 30 variable genes (train)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top30_variable_genes_train.png"), dpi=150)
plt.show()

# Cells per class (train)
cls = tr["cell_class"].value_counts().rename_axis("cell_class").reset_index(name="train_cells")
plt.figure(figsize=(10,6))
sns.barplot(data=cls.head(20), x="train_cells", y="cell_class")
plt.title("Top 20 cell classes (train)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "top20_cell_classes_train.png"), dpi=150)
plt.show()

# Coordinate distribution (train)
plt.figure(figsize=(5,5))
sns.kdeplot(x=tr["coord_X"], y=tr["coord_Y"], fill=True, thresh=0.02)
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Train coordinates density")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "train_coord_density.png"), dpi=150)
plt.show()

```


    
![png](output_14_0.png)
    



    
![png](output_14_1.png)
    



    
![png](output_14_2.png)
    



```python
import os, pandas as pd

# Try both possible layouts and pick the one that exists
cand1 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex"
cand2 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex/MERFISH_mouse_cortex"
root_directory = cand1 if os.path.exists(os.path.join(cand1, "MERFISH_mouse_cortex_train.csv")) else cand2

train_csv = f"{root_directory}/MERFISH_mouse_cortex_train.csv"
test_csv  = f"{root_directory}/MERFISH_mouse_cortex_test.csv"
assert os.path.exists(train_csv) and os.path.exists(test_csv), (train_csv, test_csv)

train_data = pd.read_csv(train_csv, index_col=0)
test_data  = pd.read_csv(test_csv,  index_col=0)

required_columns = ['coord_X', 'coord_Y', 'cell_section', 'cell_class']
for name, df in [("train", train_data), ("test", test_data)]:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")

number_of_gene = 254  # per LUNA’s example for MERFISH mouse cortex
assert train_data.columns[:number_of_gene].equals(test_data.columns[:number_of_gene]), "Gene columns mismatch"
print("✓ data OK:", train_data.shape, test_data.shape, "\nroot_directory =", root_directory)

```

    ✓ data OK: (158379, 262) (118036, 262) 
    root_directory = /nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex



```python
from hydra import initialize, compose
initialize(config_path="../configs", version_base="1.3")
cfg = compose(config_name="config")
cfg

```




    {'validation': {'if_validate': False, 'batch_size': 8, 'save_model_every_n_epochs': 250, 'check_val_every_n_epochs': 100, 'check_val_monitor': 'val_loss/absolute_rssd', 'save_top_k_models': 40, 'early_stopping': True, 'early_stopping_patience': 100}, 'test': {'save_dir': './', 'checkpoints_parent_dir': None, 'checkpoints_name_list': 'all', 'batch_size': 1, 'epoch_index': None, 'checkpoint_path': None, 'test_save_parent_path': None, 'checkpoint_name': 'all'}, 'general': {'name': 'MERFISH_mouse_cortex', 'wandb': 'online', 'mode': 'train_and_test', 'seed': 0, 'enable_progress_bar': True, 'local_saved_path': None}, 'train': {'n_epochs': 1000, 'batch_size': 6, 'lr': 0.0005, 'fast_dev_run': False, 'weight_decay': 1e-12}, 'model': {'n_layers': 8, 'diffusion_noise_schedule': 'cosine', 'diffusion_steps': 1000, 'nu': {'p': 2}, 'hidden_mlp_dims': {'X': 256, 'y': 256, 'pos': 64}, 'hidden_dims': {'dx': 256, 'dy': 1, 'num_heads': 16, 'dim_ffX': 256, 'dim_ffy': 256, 'dd': 64, 'output_features_to_pos_dims': 4}}, 'distribute': {'gpus_per_node': 1}, 'dataset': {'dataset_name': 'merfish_mouse_cortex', 'gene_columns_start': 0, 'gene_columns_end': 254, 'train_data_path': '/data/merfish_mouse_cortex_train.csv', 'test_data_path': '/data/merfish_mouse_cortex_test.csv', 'validation_data_path': None, 'maximum_graph_size': {'train': None, 'test': None, 'validation': None}}}




```python
EXP_DIR = "/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA"
!python3 main.py \
    general.name=MERFISH_mouse_cortex \
    general.wandb=disabled \
    dataset.gene_columns_start=0 \
    dataset.gene_columns_end=254 \
    distribute.gpus_per_node=1 \
    train.batch_size=6 \
    train.n_epochs=1000 \
    dataset.train_data_path={train_csv} \
    dataset.test_data_path={test_csv} \
    test.save_dir={EXP_DIR}

```

    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torchvision/image.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
      warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: libcudart.so.11.0: cannot open shared object file: No such file or directory
      warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:86: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:97: UserWarning: An issue occurred while importing 'torch-cluster'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_cluster/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-cluster'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:113: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_spline_conv/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_sparse/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
    Seed set to 0
    train
    None
    [     0   2378   4801  11532  16526  22657  27704  35113  41872  48191
      53971  59620  61817  67907  69940  76409  81975  87244  91175  95380
     100493 104376 108650 113043 117387 120493 124342 128401 132560 136919
     141710 147457 152718 158379]
    test
    None
    [     0   1706   2866   8046  13067  18222  22408  27399  31928  36978
      41367  42037  45770  50087  53647  58694  62746  66310  70391  74242
      78602  82382  85678  87715  90515  93707  96612  99928 103985 107855
     112801 118036]
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/configuration_validator.py:74: You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.
    You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    [rank: 0] Seed set to 0
    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
    ----------------------------------------------------------------------------------------------------
    distributed_backend=nccl
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    Missing logger folder: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/lightning_logs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name       | Type         | Params
    --------------------------------------------
    0 | train_loss | LossFunction | 0     
    1 | val_loss   | LossFunction | 0     
    2 | model      | Model        | 8.8 M 
    --------------------------------------------
    8.8 M     Trainable params
    0         Non-trainable params
    8.8 M     Total params
    35.220    Total estimated model params size (MB)
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
      warnings.warn(  # warn only once
    Epoch 999: 100%|█████████████████████████| 6/6 [00:02<00:00,  2.97it/s, v_num=0]`Trainer.fit` stopped: `max_epochs=1000` reached.
    Epoch 999: 100%|█████████████████████████| 6/6 [00:03<00:00,  1.91it/s, v_num=0]
    Directory: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints
    Testing checkpoint: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    Epoch index: 999
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    [rank: 0] Seed set to 0
    Restoring states from the checkpoint path at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from the checkpoint at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:492: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
    Testing DataLoader 0:   0%|                              | 0/31 [00:00<?, ?it/s]Sampling. The number of nodes to sample is 3851.
    Sampling on one graph took 9.081207752227783 seconds.
    
      0%|                                                  | 0/3851 [00:00<?, ?it/s][A
      7%|██▋                                   | 273/3851 [00:00<00:01, 2729.01it/s][A
     14%|█████▍                                | 551/3851 [00:00<00:01, 2756.19it/s][A
     22%|████████▏                             | 829/3851 [00:00<00:01, 2765.70it/s][A
     29%|██████████▋                          | 1109/3851 [00:00<00:00, 2776.75it/s][A
     36%|█████████████▎                       | 1391/3851 [00:00<00:00, 2792.27it/s][A
     43%|████████████████                     | 1671/3851 [00:00<00:00, 2665.53it/s][A
     51%|██████████████████▊                  | 1953/3851 [00:00<00:00, 2713.09it/s][A
     58%|█████████████████████▍               | 2234/3851 [00:00<00:00, 2742.66it/s][A
     65%|████████████████████████▏            | 2514/3851 [00:00<00:00, 2757.71it/s][A
     73%|██████████████████████████▊          | 2796/3851 [00:01<00:00, 2776.00it/s][A
     80%|█████████████████████████████▌       | 3077/3851 [00:01<00:00, 2784.74it/s][A
     87%|████████████████████████████████▎    | 3359/3851 [00:01<00:00, 2793.42it/s][A
    100%|█████████████████████████████████████| 3851/3851 [00:01<00:00, 2769.49it/s][A
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:14<07:21,  0.07it/s]Sampling. The number of nodes to sample is 670.
    Sampling on one graph took 8.55120038986206 seconds.
    
      0%|                                                   | 0/670 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████| 670/670 [00:00<00:00, 4992.14it/s][A
    Error executing job with overrides: ['general.name=MERFISH_mouse_cortex', 'general.wandb=disabled', 'dataset.gene_columns_start=0', 'dataset.gene_columns_end=254', 'distribute.gpus_per_node=1', 'train.batch_size=6', 'train.n_epochs=1000', 'dataset.train_data_path=/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex/MERFISH_mouse_cortex_train.csv', 'dataset.test_data_path=/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex/MERFISH_mouse_cortex_test.csv', 'test.save_dir=/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA']
    Traceback (most recent call last):
      File "/nfs/team361/mv11/LUNA/main.py", line 32, in main
        test_model(cfg, datamodule, dataset_infos)
      File "/nfs/team361/mv11/LUNA/main.py", line 71, in test_model
        test_single_checkpoint(
      File "/nfs/team361/mv11/LUNA/main.py", line 109, in test_single_checkpoint
        trainer.test(model, ckpt_path=checkpoint_path, dataloaders=dataloader_test)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 754, in test
        return call._call_and_handle_interrupt(
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 43, in _call_and_handle_interrupt
        return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/strategies/launchers/subprocess_script.py", line 105, in launch
        return function(*args, **kwargs)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 794, in _test_impl
        results = self._run(model, ckpt_path=ckpt_path)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 987, in _run
        results = self._run_stage()
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py", line 1026, in _run_stage
        return self._evaluation_loop.run()
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/loops/utilities.py", line 182, in _decorator
        return loop_run(self, *args, **kwargs)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 135, in run
        self._evaluation_step(batch, batch_idx, dataloader_idx, dataloader_iter)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/loops/evaluation_loop.py", line 396, in _evaluation_step
        output = call._call_strategy_hook(trainer, hook_name, *step_args)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/call.py", line 309, in _call_strategy_hook
        output = fn(*args, **kwargs)
      File "/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/strategies/strategy.py", line 425, in test_step
        return self.lightning_module.test_step(*args, **kwargs)
      File "/nfs/team361/mv11/LUNA/diffusion_model.py", line 82, in test_step
        test_step_func(self, data, i)
      File "/nfs/team361/mv11/LUNA/utils/diffusion_model/test/test.py", line 47, in test_step_func
        process_single_batch(self, batches, index, test_save_path)
      File "/nfs/team361/mv11/LUNA/utils/diffusion_model/test/test.py", line 90, in process_single_batch
        process_single_sample(self, batch, positions_pred, test_save_path, sample_index)
      File "/nfs/team361/mv11/LUNA/utils/diffusion_model/test/test.py", line 128, in process_single_sample
        log_dict = perform_evaluation(
      File "/nfs/team361/mv11/LUNA/utils/diffusion_model/test/test.py", line 246, in perform_evaluation
        sum_rssd, mean_rssd, absolute_rssd = compute_RSSD(
      File "/nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py", line 199, in compute_RSSD
        _, rssd, _ = compute_kabsch_algorithm(metadata_true_c, metadata_pred_c)
      File "/nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py", line 154, in compute_kabsch_algorithm
        return compute_kabsch_rotation(metadata_true_filtered, metadata_pred_filtered)
      File "/nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py", line 43, in compute_kabsch_rotation
        rot, rssd, sens = R.align_vectors(
      File "_rotation.pyx", line 3544, in scipy.spatial.transform._rotation.Rotation.align_vectors
    ValueError: Cannot return sensitivity matrix with an infinite weight or one vector pair
    
    Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:24<12:18,  0.04it/s]
    [rank0]:[W822 11:40:52.549556009 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())



```python
import scipy, sys
print("SciPy:", scipy.__version__)

```

    SciPy: 1.15.3



```python
%pip install --force-reinstall "scipy==1.9.1"
import IPython, os; IPython.get_ipython().kernel.do_shutdown(True)  # restarts kernel

```

    Collecting scipy==1.9.1
      Downloading scipy-1.9.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (2.2 kB)
    Collecting numpy<1.25.0,>=1.18.5 (from scipy==1.9.1)
      Downloading numpy-1.24.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (5.6 kB)
    Downloading scipy-1.9.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (43.9 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m43.9/43.9 MB[0m [31m99.4 MB/s[0m  [33m0:00:00[0ms[0m eta [36m0:00:01[0m
    [?25hDownloading numpy-1.24.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (17.3 MB)
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m17.3/17.3 MB[0m [31m81.0 MB/s[0m  [33m0:00:00[0mta [36m0:00:01[0m
    [?25hInstalling collected packages: numpy, scipy
    [2K  Attempting uninstall: numpy
    [2K    Found existing installation: numpy 2.2.6
    [2K    Uninstalling numpy-2.2.6:━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0/2[0m [numpy]
    [2K      Successfully uninstalled numpy-2.2.6━━━━━━━━━━━━[0m [32m0/2[0m [numpy]
    [2K   [38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0/2[0m [numpy][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~-mpy.libs'.
      You can safely remove it manually.[0m[33m
    [2K   [38;5;237m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m0/2[0m [numpy][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~-mpy'.
      You can safely remove it manually.[0m[33m
    [2K  Attempting uninstall: scipy
    [2K    Found existing installation: scipy 1.15.3━━━━━[0m [32m0/2[0m [numpy]
    [2K    Uninstalling scipy-1.15.3:━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━[0m [32m1/2[0m [scipy]
    [2K      Successfully uninstalled scipy-1.15.30m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━[0m [32m1/2[0m [scipy]
    [2K   [38;2;249;38;114m━━━━━━━━━━━━━━━━━━━━[0m[38;5;237m╺[0m[38;5;237m━━━━━━━━━━━━━━━━━━━[0m [32m1/2[0m [scipy][33m  WARNING: Failed to remove contents in a temporary directory '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/~cipy'.
      You can safely remove it manually.[0m[33m
    [2K   [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2/2[0m [scipy]
    [1A[2K[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    scikit-image 0.25.2 requires scipy>=1.11.4, but you have scipy 1.9.1 which is incompatible.
    torchvision 0.15.2+cu118 requires torch==2.0.1, but you have torch 2.8.0 which is incompatible.
    xarray-dataclass 3.0.0 requires numpy>=2.0.0, but you have numpy 1.24.4 which is incompatible.[0m[31m
    [0mSuccessfully installed numpy-1.24.4 scipy-1.9.1
    Note: you may need to restart the kernel to use updated packages.





    {'status': 'ok', 'restart': True}




```python

```

# Test


```python
import torch, torch_geometric as tg
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
print("PyG:", tg.__version__)

```

    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: libcudart.so.11.0: cannot open shared object file: No such file or directory
      warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:86: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:97: UserWarning: An issue occurred while importing 'torch-cluster'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_cluster/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-cluster'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:113: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_spline_conv/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_sparse/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-sparse'. "


    CUDA available: True
    GPU: NVIDIA H100 80GB HBM3
    PyG: 2.6.1



```python
import os

# Keep W&B off
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"

# Data paths (auto-handle nested unzip)
cand1 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex"
cand2 = "/nfs/team361/mv11/DATASETS/luna/MERFISH_mouse_cortex/MERFISH_mouse_cortex"
root_directory = cand1 if os.path.exists(os.path.join(cand1, "MERFISH_mouse_cortex_train.csv")) else cand2
train_csv = f"{root_directory}/MERFISH_mouse_cortex_train.csv"
test_csv  = f"{root_directory}/MERFISH_mouse_cortex_test.csv"

EXP_DIR = "/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA"
CKPT_PARENT = "/nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints"

# Run just the test phase over all checkpoints in that folder
!python3 main.py \
    general.name=MERFISH_mouse_cortex_test_only \
    general.mode=test_only \
    general.wandb=disabled \
    dataset.gene_columns_start=0 \
    dataset.gene_columns_end=254 \
    dataset.train_data_path={train_csv} \
    dataset.test_data_path={test_csv} \
    test.save_dir={EXP_DIR} \
    test.checkpoints_parent_dir={CKPT_PARENT} \
    test.checkpoints_name_list=all

```

    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torchvision/io/image.py:13: UserWarning: Failed to load image Python extension: '/nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torchvision/image.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev'If you don't plan on using image functionality from `torchvision.io`, you can ignore this warning. Otherwise, there might be something wrong with your environment. Did you have `libjpeg` or `libpng` installed before building `torchvision` from source?
      warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:68: UserWarning: An issue occurred while importing 'pyg-lib'. Disabling its usage. Stacktrace: libcudart.so.11.0: cannot open shared object file: No such file or directory
      warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:86: UserWarning: An issue occurred while importing 'torch-scatter'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_scatter/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-scatter'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:97: UserWarning: An issue occurred while importing 'torch-cluster'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_cluster/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-cluster'. "
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:113: UserWarning: An issue occurred while importing 'torch-spline-conv'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_spline_conv/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_geometric/typing.py:124: UserWarning: An issue occurred while importing 'torch-sparse'. Disabling its usage. Stacktrace: /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch_sparse/_version_cuda.so: undefined symbol: _ZN3c1017RegisterOperatorsD1Ev
      warnings.warn(f"An issue occurred while importing 'torch-sparse'. "
    Seed set to 0
    train
    None
    [     0   2378   4801  11532  16526  22657  27704  35113  41872  48191
      53971  59620  61817  67907  69940  76409  81975  87244  91175  95380
     100493 104376 108650 113043 117387 120493 124342 128401 132560 136919
     141710 147457 152718 158379]
    test
    None
    [     0   1706   2866   8046  13067  18222  22408  27399  31928  36978
      41367  42037  45770  50087  53647  58694  62746  66310  70391  74242
      78602  82382  85678  87715  90515  93707  96612  99928 103985 107855
     112801 118036]
    Directory: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    Testing checkpoint: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=499.ckpt
    Epoch index: 499
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/logger_connector/logger_connector.py:75: Starting from v1.9.0, `tensorboardX` has been removed as a dependency of the `pytorch_lightning` package, due to potential conflicts with other packages in the ML ecosystem. For this reason, `logger=True` will use `CSVLogger` as the default logger, unless the `tensorboard` or `tensorboardX` packages are found. Please `pip install lightning[extra]` or one of them to enable TensorBoard support by default
    You are using a CUDA device ('NVIDIA H100 80GB HBM3') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
    [rank: 0] Seed set to 0
    Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/1
    ----------------------------------------------------------------------------------------------------
    distributed_backend=nccl
    All distributed processes registered. Starting with 1 processes
    ----------------------------------------------------------------------------------------------------
    
    Missing logger folder: /nfs/team361/mv11/outputs/2025-08-22/11-55-04-MERFISH_mouse_cortex_test_only/lightning_logs
    Restoring states from the checkpoint path at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=499.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from the checkpoint at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=499.ckpt
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:492: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
      warnings.warn(  # warn only once
    Testing DataLoader 0:   0%|                              | 0/31 [00:00<?, ?it/s]Sampling. The number of nodes to sample is 3851.
    Sampling on one graph took 9.1297128200531 seconds.
    
      0%|                                                  | 0/3851 [00:00<?, ?it/s][A
      4%|█▍                                    | 141/3851 [00:00<00:02, 1409.10it/s][A
      7%|██▊                                   | 285/3851 [00:00<00:02, 1424.87it/s][A
     11%|████▏                                 | 429/3851 [00:00<00:02, 1428.36it/s][A
     15%|█████▋                                | 573/3851 [00:00<00:02, 1432.40it/s][A
     19%|███████                               | 718/3851 [00:00<00:02, 1435.84it/s][A
     22%|████████▌                             | 862/3851 [00:00<00:02, 1423.71it/s][A
     26%|█████████▋                           | 1006/3851 [00:00<00:01, 1427.77it/s][A
     30%|███████████                          | 1151/3851 [00:00<00:01, 1431.81it/s][A
     34%|████████████▍                        | 1296/3851 [00:00<00:01, 1435.39it/s][A
     37%|█████████████▊                       | 1441/3851 [00:01<00:01, 1439.25it/s][A
     41%|███████████████▏                     | 1585/3851 [00:01<00:01, 1381.84it/s][A
     45%|████████████████▌                    | 1730/3851 [00:01<00:01, 1400.14it/s][A
     49%|██████████████████                   | 1875/3851 [00:01<00:01, 1414.38it/s][A
     52%|███████████████████▍                 | 2020/3851 [00:01<00:01, 1424.10it/s][A
     56%|████████████████████▊                | 2166/3851 [00:01<00:01, 1432.07it/s][A
     60%|██████████████████████▏              | 2311/3851 [00:01<00:01, 1436.63it/s][A
     64%|███████████████████████▌             | 2456/3851 [00:01<00:00, 1440.43it/s][A
     68%|████████████████████████▉            | 2601/3851 [00:01<00:00, 1443.11it/s][A
     71%|██████████████████████████▍          | 2747/3851 [00:01<00:00, 1445.64it/s][A
     75%|███████████████████████████▊         | 2893/3851 [00:02<00:00, 1447.94it/s][A
     79%|█████████████████████████████▏       | 3038/3851 [00:02<00:00, 1440.39it/s][A
     83%|██████████████████████████████▌      | 3183/3851 [00:02<00:00, 1442.82it/s][A
     86%|███████████████████████████████▉     | 3328/3851 [00:02<00:00, 1444.47it/s][A
     90%|█████████████████████████████████▎   | 3473/3851 [00:02<00:00, 1388.74it/s][A
     94%|██████████████████████████████████▋  | 3613/3851 [00:02<00:00, 1349.38it/s][A
    100%|█████████████████████████████████████| 3851/3851 [00:02<00:00, 1417.35it/s][A
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:16<08:10,  0.06it/s]Sampling. The number of nodes to sample is 670.
    Sampling on one graph took 8.50302791595459 seconds.
    
      0%|                                                   | 0/670 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████| 670/670 [00:00<00:00, 3790.41it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:   6%|█▍                    | 2/31 [00:25<06:10,  0.08it/s]Sampling. The number of nodes to sample is 4360.
    Sampling on one graph took 9.580984115600586 seconds.
    
      0%|                                                  | 0/4360 [00:00<?, ?it/s][A
      3%|█▏                                    | 131/4360 [00:00<00:03, 1301.00it/s][A
      6%|██▎                                   | 263/4360 [00:00<00:03, 1306.83it/s][A
      9%|███▍                                  | 394/4360 [00:00<00:03, 1300.11it/s][A
     12%|████▌                                 | 526/4360 [00:00<00:02, 1304.76it/s][A
     15%|█████▋                                | 658/4360 [00:00<00:02, 1307.54it/s][A
     18%|██████▉                               | 789/4360 [00:00<00:02, 1251.13it/s][A
     21%|████████                              | 920/4360 [00:00<00:02, 1269.69it/s][A
     24%|████████▉                            | 1052/4360 [00:00<00:02, 1282.93it/s][A
     27%|██████████                           | 1184/4360 [00:00<00:02, 1292.43it/s][A
     30%|███████████▏                         | 1316/4360 [00:01<00:02, 1299.30it/s][A
     33%|████████████▎                        | 1448/4360 [00:01<00:02, 1303.40it/s][A
     36%|█████████████▍                       | 1580/4360 [00:01<00:02, 1306.21it/s][A
     39%|██████████████▌                      | 1712/4360 [00:01<00:02, 1307.95it/s][A
     42%|███████████████▋                     | 1844/4360 [00:01<00:01, 1309.45it/s][A
     45%|████████████████▊                    | 1976/4360 [00:01<00:01, 1311.69it/s][A
     48%|█████████████████▉                   | 2108/4360 [00:01<00:01, 1312.40it/s][A
     51%|███████████████████                  | 2240/4360 [00:01<00:01, 1310.71it/s][A
     54%|████████████████████▏                | 2372/4360 [00:01<00:01, 1310.74it/s][A
     57%|█████████████████████▏               | 2504/4360 [00:01<00:01, 1258.99it/s][A
     60%|██████████████████████▎              | 2636/4360 [00:02<00:01, 1275.66it/s][A
     63%|███████████████████████▍             | 2768/4360 [00:02<00:01, 1286.49it/s][A
     67%|████████████████████████▌            | 2900/4360 [00:02<00:01, 1293.86it/s][A
     70%|█████████████████████████▋           | 3032/4360 [00:02<00:01, 1301.30it/s][A
     73%|██████████████████████████▊          | 3164/4360 [00:02<00:00, 1304.84it/s][A
     76%|███████████████████████████▉         | 3296/4360 [00:02<00:00, 1308.85it/s][A
     79%|█████████████████████████████        | 3427/4360 [00:02<00:00, 1306.45it/s][A
     82%|██████████████████████████████▏      | 3558/4360 [00:02<00:00, 1306.78it/s][A
     85%|███████████████████████████████▎     | 3690/4360 [00:02<00:00, 1307.87it/s][A
     88%|████████████████████████████████▍    | 3821/4360 [00:02<00:00, 1308.18it/s][A
     91%|█████████████████████████████████▌   | 3952/4360 [00:03<00:00, 1308.42it/s][A
     94%|██████████████████████████████████▋  | 4084/4360 [00:03<00:00, 1309.37it/s][A
     97%|███████████████████████████████████▊ | 4215/4360 [00:03<00:00, 1255.34it/s][A
    100%|█████████████████████████████████████| 4360/4360 [00:03<00:00, 1294.52it/s][A
    Testing DataLoader 0:  10%|██▏                   | 3/31 [00:43<06:46,  0.07it/s]Sampling. The number of nodes to sample is 5180.
    Sampling on one graph took 10.494909286499023 seconds.
    
      0%|                                                  | 0/5180 [00:00<?, ?it/s][A
      2%|▊                                     | 112/5180 [00:00<00:04, 1110.42it/s][A
      4%|█▋                                    | 224/5180 [00:00<00:04, 1114.58it/s][A
      7%|██▍                                   | 337/5180 [00:00<00:04, 1117.53it/s][A
      9%|███▎                                  | 450/5180 [00:00<00:04, 1118.93it/s][A
     11%|████▏                                 | 563/5180 [00:00<00:04, 1120.07it/s][A
     13%|████▉                                 | 676/5180 [00:00<00:04, 1122.39it/s][A
     15%|█████▊                                | 789/5180 [00:00<00:03, 1121.71it/s][A
     17%|██████▌                               | 902/5180 [00:00<00:03, 1122.42it/s][A
     20%|███████▎                             | 1015/5180 [00:00<00:03, 1123.30it/s][A
     22%|████████                             | 1128/5180 [00:01<00:03, 1122.02it/s][A
     24%|████████▊                            | 1241/5180 [00:01<00:03, 1122.72it/s][A
     26%|█████████▋                           | 1354/5180 [00:01<00:03, 1119.54it/s][A
     28%|██████████▍                          | 1466/5180 [00:01<00:03, 1075.00it/s][A
     30%|███████████▎                         | 1579/5180 [00:01<00:03, 1089.86it/s][A
     33%|████████████                         | 1692/5180 [00:01<00:03, 1100.03it/s][A
     35%|████████████▉                        | 1805/5180 [00:01<00:03, 1106.89it/s][A
     37%|█████████████▋                       | 1918/5180 [00:01<00:02, 1112.74it/s][A
     39%|██████████████▌                      | 2031/5180 [00:01<00:02, 1115.94it/s][A
     41%|███████████████▎                     | 2144/5180 [00:01<00:02, 1118.60it/s][A
     44%|████████████████                     | 2257/5180 [00:02<00:02, 1119.25it/s][A
     46%|████████████████▉                    | 2369/5180 [00:02<00:02, 1118.54it/s][A
     48%|█████████████████▋                   | 2482/5180 [00:02<00:02, 1120.03it/s][A
     50%|██████████████████▌                  | 2595/5180 [00:02<00:02, 1121.49it/s][A
     52%|███████████████████▎                 | 2708/5180 [00:02<00:02, 1122.48it/s][A
     54%|████████████████████▏                | 2821/5180 [00:02<00:02, 1122.42it/s][A
     57%|████████████████████▉                | 2934/5180 [00:02<00:02, 1121.87it/s][A
     59%|█████████████████████▊               | 3047/5180 [00:02<00:01, 1073.68it/s][A
     61%|██████████████████████▌              | 3160/5180 [00:02<00:01, 1087.28it/s][A
     63%|███████████████████████▍             | 3273/5180 [00:02<00:01, 1097.51it/s][A
     65%|████████████████████████▏            | 3386/5180 [00:03<00:01, 1104.20it/s][A
     68%|████████████████████████▉            | 3498/5180 [00:03<00:01, 1108.69it/s][A
     70%|█████████████████████████▊           | 3610/5180 [00:03<00:01, 1111.54it/s][A
     72%|██████████████████████████▌          | 3723/5180 [00:03<00:01, 1114.67it/s][A
     74%|███████████████████████████▍         | 3835/5180 [00:03<00:01, 1116.23it/s][A
     76%|████████████████████████████▏        | 3948/5180 [00:03<00:01, 1117.59it/s][A
     78%|█████████████████████████████        | 4060/5180 [00:03<00:01, 1118.21it/s][A
     81%|█████████████████████████████▊       | 4173/5180 [00:03<00:00, 1119.08it/s][A
     83%|██████████████████████████████▌      | 4285/5180 [00:03<00:00, 1117.39it/s][A
     85%|███████████████████████████████▍     | 4398/5180 [00:03<00:00, 1118.64it/s][A
     87%|████████████████████████████████▏    | 4510/5180 [00:04<00:00, 1035.13it/s][A
     89%|████████████████████████████████▉    | 4619/5180 [00:04<00:00, 1049.66it/s][A
     91%|█████████████████████████████████▊   | 4731/5180 [00:04<00:00, 1067.43it/s][A
     94%|██████████████████████████████████▌  | 4844/5180 [00:04<00:00, 1083.36it/s][A
     96%|███████████████████████████████████▍ | 4957/5180 [00:04<00:00, 1096.06it/s][A
    100%|█████████████████████████████████████| 5180/5180 [00:04<00:00, 1106.98it/s][A
    Testing DataLoader 0:  13%|██▊                   | 4/31 [01:05<07:22,  0.06it/s]Sampling. The number of nodes to sample is 4946.
    Sampling on one graph took 10.31455111503601 seconds.
    
      0%|                                                  | 0/4946 [00:00<?, ?it/s][A
      2%|▉                                     | 117/4946 [00:00<00:04, 1169.50it/s][A
      5%|█▊                                    | 235/4946 [00:00<00:04, 1172.48it/s][A
      7%|██▋                                   | 353/4946 [00:00<00:03, 1170.59it/s][A
     10%|███▌                                  | 471/4946 [00:00<00:03, 1172.26it/s][A
     12%|████▌                                 | 589/4946 [00:00<00:03, 1119.45it/s][A
     14%|█████▍                                | 707/4946 [00:00<00:03, 1136.96it/s][A
     17%|██████▎                               | 825/4946 [00:00<00:03, 1147.84it/s][A
     19%|███████▏                              | 943/4946 [00:00<00:03, 1154.92it/s][A
     21%|███████▉                             | 1061/4946 [00:00<00:03, 1160.96it/s][A
     24%|████████▊                            | 1179/4946 [00:01<00:03, 1164.10it/s][A
     26%|█████████▋                           | 1297/4946 [00:01<00:03, 1167.07it/s][A
     29%|██████████▌                          | 1415/4946 [00:01<00:03, 1169.09it/s][A
     31%|███████████▍                         | 1532/4946 [00:01<00:02, 1169.27it/s][A
     33%|████████████▎                        | 1650/4946 [00:01<00:02, 1170.55it/s][A
     36%|█████████████▏                       | 1768/4946 [00:01<00:02, 1172.08it/s][A
     38%|██████████████                       | 1886/4946 [00:01<00:02, 1171.87it/s][A
     41%|██████████████▉                      | 2004/4946 [00:01<00:02, 1173.14it/s][A
     43%|███████████████▊                     | 2122/4946 [00:01<00:02, 1122.89it/s][A
     45%|████████████████▊                    | 2240/4946 [00:01<00:02, 1137.04it/s][A
     48%|█████████████████▋                   | 2357/4946 [00:02<00:02, 1145.43it/s][A
     50%|██████████████████▌                  | 2474/4946 [00:02<00:02, 1152.56it/s][A
     52%|███████████████████▍                 | 2592/4946 [00:02<00:02, 1158.31it/s][A
     55%|████████████████████▎                | 2708/4946 [00:02<00:01, 1158.49it/s][A
     57%|█████████████████████▏               | 2826/4946 [00:02<00:01, 1162.43it/s][A
     60%|██████████████████████               | 2944/4946 [00:02<00:01, 1165.52it/s][A
     62%|██████████████████████▉              | 3062/4946 [00:02<00:01, 1167.45it/s][A
     64%|███████████████████████▊             | 3180/4946 [00:02<00:01, 1168.53it/s][A
     67%|████████████████████████▋            | 3298/4946 [00:02<00:01, 1170.18it/s][A
     69%|█████████████████████████▌           | 3416/4946 [00:02<00:01, 1170.90it/s][A
     71%|██████████████████████████▍          | 3534/4946 [00:03<00:01, 1130.49it/s][A
     74%|███████████████████████████▎         | 3651/4946 [00:03<00:01, 1139.26it/s][A
     76%|████████████████████████████▏        | 3769/4946 [00:03<00:01, 1150.67it/s][A
     79%|█████████████████████████████        | 3887/4946 [00:03<00:00, 1157.43it/s][A
     81%|█████████████████████████████▉       | 4005/4946 [00:03<00:00, 1162.21it/s][A
     83%|██████████████████████████████▊      | 4123/4946 [00:03<00:00, 1165.83it/s][A
     86%|███████████████████████████████▋     | 4241/4946 [00:03<00:00, 1167.81it/s][A
     88%|████████████████████████████████▌    | 4359/4946 [00:03<00:00, 1170.74it/s][A
     91%|█████████████████████████████████▍   | 4477/4946 [00:03<00:00, 1168.51it/s][A
     93%|██████████████████████████████████▎  | 4595/4946 [00:03<00:00, 1171.21it/s][A
     95%|███████████████████████████████████▎ | 4713/4946 [00:04<00:00, 1172.71it/s][A
    100%|█████████████████████████████████████| 4946/4946 [00:04<00:00, 1161.07it/s][A
    Testing DataLoader 0:  16%|███▌                  | 5/31 [01:26<07:28,  0.06it/s]Sampling. The number of nodes to sample is 2037.
    Sampling on one graph took 8.545300722122192 seconds.
    
      0%|                                                  | 0/2037 [00:00<?, ?it/s][A
     11%|████▎                                 | 230/2037 [00:00<00:00, 2291.09it/s][A
     23%|████████▌                             | 461/2037 [00:00<00:00, 2300.98it/s][A
     34%|████████████▉                         | 692/2037 [00:00<00:00, 2303.38it/s][A
     45%|█████████████████▏                    | 923/2037 [00:00<00:00, 2182.74it/s][A
     56%|████████████████████▊                | 1143/2037 [00:00<00:00, 2112.78it/s][A
     67%|████████████████████████▉            | 1373/2037 [00:00<00:00, 2170.79it/s][A
     79%|█████████████████████████████        | 1603/2037 [00:00<00:00, 2208.85it/s][A
    100%|█████████████████████████████████████| 2037/2037 [00:00<00:00, 2226.95it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  19%|████▎                 | 6/31 [01:37<06:44,  0.06it/s]Sampling. The number of nodes to sample is 5050.
    Sampling on one graph took 10.39351224899292 seconds.
    
      0%|                                                  | 0/5050 [00:00<?, ?it/s][A
      2%|▊                                     | 115/5050 [00:00<00:04, 1147.50it/s][A
      5%|█▋                                    | 231/5050 [00:00<00:04, 1151.34it/s][A
      7%|██▌                                   | 347/5050 [00:00<00:04, 1151.18it/s][A
      9%|███▍                                  | 463/5050 [00:00<00:04, 1088.88it/s][A
     11%|████▎                                 | 573/5050 [00:00<00:04, 1049.82it/s][A
     14%|█████▏                                | 688/5050 [00:00<00:04, 1081.17it/s][A
     16%|██████                                | 804/5050 [00:00<00:03, 1104.51it/s][A
     18%|██████▉                               | 920/5050 [00:00<00:03, 1118.98it/s][A
     20%|███████▌                             | 1035/5050 [00:00<00:03, 1126.93it/s][A
     23%|████████▍                            | 1151/5050 [00:01<00:03, 1135.07it/s][A
     25%|█████████▎                           | 1267/5050 [00:01<00:03, 1141.60it/s][A
     27%|██████████▏                          | 1383/5050 [00:01<00:03, 1147.11it/s][A
     30%|██████████▉                          | 1499/5050 [00:01<00:03, 1148.52it/s][A
     32%|███████████▊                         | 1615/5050 [00:01<00:02, 1151.15it/s][A
     34%|████████████▋                        | 1731/5050 [00:01<00:02, 1151.92it/s][A
     37%|█████████████▌                       | 1847/5050 [00:01<00:02, 1112.05it/s][A
     39%|██████████████▎                      | 1959/5050 [00:01<00:02, 1074.47it/s][A
     41%|███████████████▏                     | 2075/5050 [00:01<00:02, 1097.87it/s][A
     43%|████████████████                     | 2191/5050 [00:01<00:02, 1114.11it/s][A
     46%|████████████████▉                    | 2307/5050 [00:02<00:02, 1125.69it/s][A
     48%|█████████████████▊                   | 2423/5050 [00:02<00:02, 1134.44it/s][A
     50%|██████████████████▌                  | 2538/5050 [00:02<00:02, 1138.58it/s][A
     53%|███████████████████▍                 | 2654/5050 [00:02<00:02, 1144.09it/s][A
     55%|████████████████████▎                | 2770/5050 [00:02<00:01, 1147.51it/s][A
     57%|█████████████████████▏               | 2886/5050 [00:02<00:01, 1149.05it/s][A
     59%|█████████████████████▉               | 3002/5050 [00:02<00:01, 1150.64it/s][A
     62%|██████████████████████▊              | 3118/5050 [00:02<00:01, 1150.73it/s][A
     64%|███████████████████████▋             | 3234/5050 [00:02<00:01, 1151.88it/s][A
     66%|████████████████████████▌            | 3350/5050 [00:02<00:01, 1151.88it/s][A
     69%|█████████████████████████▍           | 3466/5050 [00:03<00:01, 1152.40it/s][A
     71%|██████████████████████████▏          | 3582/5050 [00:03<00:01, 1153.04it/s][A
     73%|███████████████████████████          | 3698/5050 [00:03<00:01, 1153.43it/s][A
     76%|███████████████████████████▉         | 3814/5050 [00:03<00:01, 1154.19it/s][A
     78%|████████████████████████████▊        | 3930/5050 [00:03<00:00, 1154.27it/s][A
     80%|█████████████████████████████▋       | 4046/5050 [00:03<00:00, 1154.74it/s][A
     82%|██████████████████████████████▍      | 4162/5050 [00:03<00:00, 1153.47it/s][A
     85%|███████████████████████████████▎     | 4278/5050 [00:03<00:00, 1154.30it/s][A
     87%|████████████████████████████████▏    | 4394/5050 [00:03<00:00, 1154.38it/s][A
     89%|█████████████████████████████████    | 4510/5050 [00:03<00:00, 1154.96it/s][A
     92%|█████████████████████████████████▉   | 4626/5050 [00:04<00:00, 1154.85it/s][A
     94%|██████████████████████████████████▋  | 4742/5050 [00:04<00:00, 1154.47it/s][A
     96%|███████████████████████████████████▌ | 4858/5050 [00:04<00:00, 1153.64it/s][A
    100%|█████████████████████████████████████| 5050/5050 [00:04<00:00, 1135.01it/s][A
    Testing DataLoader 0:  23%|████▉                 | 7/31 [01:58<06:45,  0.06it/s]Sampling. The number of nodes to sample is 5047.
    Sampling on one graph took 10.42126727104187 seconds.
    
      0%|                                                  | 0/5047 [00:00<?, ?it/s][A
      2%|▊                                     | 115/5047 [00:00<00:04, 1145.50it/s][A
      5%|█▋                                    | 231/5047 [00:00<00:04, 1150.54it/s][A
      7%|██▌                                   | 347/5047 [00:00<00:04, 1077.27it/s][A
      9%|███▍                                  | 456/5047 [00:00<00:04, 1044.59it/s][A
     11%|████▎                                 | 572/5047 [00:00<00:04, 1082.36it/s][A
     14%|█████▏                                | 688/5047 [00:00<00:03, 1105.85it/s][A
     16%|██████                                | 804/5047 [00:00<00:03, 1121.51it/s][A
     18%|██████▉                               | 920/5047 [00:00<00:03, 1131.52it/s][A
     21%|███████▌                             | 1035/5047 [00:00<00:03, 1135.94it/s][A
     23%|████████▍                            | 1151/5047 [00:01<00:03, 1140.89it/s][A
     25%|█████████▎                           | 1267/5047 [00:01<00:03, 1145.18it/s][A
     27%|██████████▏                          | 1383/5047 [00:01<00:03, 1148.45it/s][A
     30%|██████████▉                          | 1499/5047 [00:01<00:03, 1150.45it/s][A
     32%|███████████▊                         | 1615/5047 [00:01<00:02, 1151.02it/s][A
     34%|████████████▋                        | 1731/5047 [00:01<00:02, 1150.92it/s][A
     37%|█████████████▌                       | 1847/5047 [00:01<00:02, 1151.38it/s][A
     39%|██████████████▍                      | 1963/5047 [00:01<00:02, 1103.57it/s][A
     41%|███████████████▏                     | 2079/5047 [00:01<00:02, 1118.34it/s][A
     43%|████████████████                     | 2195/5047 [00:01<00:02, 1128.82it/s][A
     46%|████████████████▉                    | 2311/5047 [00:02<00:02, 1136.45it/s][A
     48%|█████████████████▊                   | 2427/5047 [00:02<00:02, 1142.45it/s][A
     50%|██████████████████▋                  | 2543/5047 [00:02<00:02, 1144.74it/s][A
     53%|███████████████████▍                 | 2659/5047 [00:02<00:02, 1147.36it/s][A
     55%|████████████████████▎                | 2775/5047 [00:02<00:01, 1149.63it/s][A
     57%|█████████████████████▏               | 2891/5047 [00:02<00:01, 1145.50it/s][A
     60%|██████████████████████               | 3007/5047 [00:02<00:01, 1147.60it/s][A
     62%|██████████████████████▉              | 3123/5047 [00:02<00:01, 1149.60it/s][A
     64%|███████████████████████▋             | 3239/5047 [00:02<00:01, 1150.15it/s][A
     66%|████████████████████████▌            | 3355/5047 [00:02<00:01, 1123.84it/s][A
     69%|█████████████████████████▍           | 3468/5047 [00:03<00:01, 1110.20it/s][A
     71%|██████████████████████████▎          | 3584/5047 [00:03<00:01, 1122.82it/s][A
     73%|███████████████████████████▏         | 3700/5047 [00:03<00:01, 1132.56it/s][A
     76%|███████████████████████████▉         | 3816/5047 [00:03<00:01, 1139.96it/s][A
     78%|████████████████████████████▊        | 3932/5047 [00:03<00:00, 1144.36it/s][A
     80%|█████████████████████████████▋       | 4048/5047 [00:03<00:00, 1147.43it/s][A
     83%|██████████████████████████████▌      | 4164/5047 [00:03<00:00, 1149.93it/s][A
     85%|███████████████████████████████▍     | 4280/5047 [00:03<00:00, 1150.78it/s][A
     87%|████████████████████████████████▏    | 4396/5047 [00:03<00:00, 1151.01it/s][A
     89%|█████████████████████████████████    | 4512/5047 [00:03<00:00, 1152.35it/s][A
     92%|█████████████████████████████████▉   | 4628/5047 [00:04<00:00, 1153.16it/s][A
     94%|██████████████████████████████████▊  | 4744/5047 [00:04<00:00, 1154.44it/s][A
     96%|███████████████████████████████████▋ | 4860/5047 [00:04<00:00, 1106.93it/s][A
    100%|█████████████████████████████████████| 5047/5047 [00:04<00:00, 1133.90it/s][A
    Testing DataLoader 0:  26%|█████▋                | 8/31 [02:19<06:41,  0.06it/s]Sampling. The number of nodes to sample is 5235.
    Sampling on one graph took 10.607892751693726 seconds.
    
      0%|                                                  | 0/5235 [00:00<?, ?it/s][A
      2%|▊                                     | 111/5235 [00:00<00:04, 1107.84it/s][A
      4%|█▌                                    | 223/5235 [00:00<00:04, 1112.34it/s][A
      6%|██▍                                   | 335/5235 [00:00<00:04, 1114.54it/s][A
      9%|███▏                                  | 447/5235 [00:00<00:04, 1114.60it/s][A
     11%|████                                  | 559/5235 [00:00<00:04, 1114.39it/s][A
     13%|████▊                                 | 671/5235 [00:00<00:04, 1113.82it/s][A
     15%|█████▋                                | 783/5235 [00:00<00:04, 1063.36it/s][A
     17%|██████▍                               | 895/5235 [00:00<00:04, 1079.21it/s][A
     19%|███████                              | 1007/5235 [00:00<00:03, 1090.02it/s][A
     21%|███████▉                             | 1119/5235 [00:01<00:03, 1098.02it/s][A
     24%|████████▋                            | 1231/5235 [00:01<00:03, 1103.96it/s][A
     26%|█████████▍                           | 1343/5235 [00:01<00:03, 1107.63it/s][A
     28%|██████████▎                          | 1455/5235 [00:01<00:03, 1110.82it/s][A
     30%|███████████                          | 1567/5235 [00:01<00:03, 1109.07it/s][A
     32%|███████████▊                         | 1679/5235 [00:01<00:03, 1112.19it/s][A
     34%|████████████▋                        | 1791/5235 [00:01<00:03, 1112.59it/s][A
     36%|█████████████▍                       | 1903/5235 [00:01<00:02, 1113.21it/s][A
     38%|██████████████▏                      | 2015/5235 [00:01<00:02, 1113.82it/s][A
     41%|███████████████                      | 2127/5235 [00:01<00:02, 1114.99it/s][A
     43%|███████████████▊                     | 2239/5235 [00:02<00:02, 1115.39it/s][A
     45%|████████████████▌                    | 2351/5235 [00:02<00:02, 1070.66it/s][A
     47%|█████████████████▍                   | 2463/5235 [00:02<00:02, 1083.29it/s][A
     49%|██████████████████▏                  | 2575/5235 [00:02<00:02, 1092.48it/s][A
     51%|██████████████████▉                  | 2687/5235 [00:02<00:02, 1098.56it/s][A
     53%|███████████████████▊                 | 2799/5235 [00:02<00:02, 1102.39it/s][A
     56%|████████████████████▌                | 2911/5235 [00:02<00:02, 1105.87it/s][A
     58%|█████████████████████▎               | 3023/5235 [00:02<00:01, 1109.10it/s][A
     60%|██████████████████████▏              | 3134/5235 [00:02<00:01, 1101.24it/s][A
     62%|██████████████████████▉              | 3246/5235 [00:02<00:01, 1104.40it/s][A
     64%|███████████████████████▋             | 3358/5235 [00:03<00:01, 1107.01it/s][A
     66%|████████████████████████▌            | 3470/5235 [00:03<00:01, 1109.62it/s][A
     68%|█████████████████████████▎           | 3582/5235 [00:03<00:01, 1111.40it/s][A
     71%|██████████████████████████           | 3694/5235 [00:03<00:01, 1070.69it/s][A
     73%|██████████████████████████▊          | 3802/5235 [00:03<00:01, 1040.69it/s][A
     75%|███████████████████████████▋         | 3914/5235 [00:03<00:01, 1062.20it/s][A
     77%|████████████████████████████▍        | 4026/5235 [00:03<00:01, 1077.18it/s][A
     79%|█████████████████████████████▏       | 4138/5235 [00:03<00:01, 1087.38it/s][A
     81%|██████████████████████████████       | 4249/5235 [00:03<00:00, 1093.81it/s][A
     83%|██████████████████████████████▊      | 4360/5235 [00:03<00:00, 1097.62it/s][A
     85%|███████████████████████████████▌     | 4471/5235 [00:04<00:00, 1099.16it/s][A
     88%|████████████████████████████████▍    | 4582/5235 [00:04<00:00, 1101.90it/s][A
     90%|█████████████████████████████████▏   | 4693/5235 [00:04<00:00, 1102.00it/s][A
     92%|█████████████████████████████████▉   | 4804/5235 [00:04<00:00, 1102.24it/s][A
     94%|██████████████████████████████████▋  | 4915/5235 [00:04<00:00, 1101.26it/s][A
     96%|███████████████████████████████████▌ | 5026/5235 [00:04<00:00, 1063.23it/s][A
    100%|█████████████████████████████████████| 5235/5235 [00:04<00:00, 1091.74it/s][A
    Testing DataLoader 0:  29%|██████▍               | 9/31 [02:42<06:36,  0.06it/s]Sampling. The number of nodes to sample is 3870.
    Sampling on one graph took 8.857812643051147 seconds.
    
      0%|                                                  | 0/3870 [00:00<?, ?it/s][A
      4%|█▍                                    | 143/3870 [00:00<00:02, 1427.89it/s][A
      7%|██▊                                   | 286/3870 [00:00<00:02, 1428.39it/s][A
     11%|████▏                                 | 429/3870 [00:00<00:02, 1339.27it/s][A
     15%|█████▌                                | 572/3870 [00:00<00:02, 1373.45it/s][A
     18%|██████▉                               | 710/3870 [00:00<00:02, 1322.96it/s][A
     22%|████████▍                             | 853/3870 [00:00<00:02, 1357.28it/s][A
     26%|█████████▊                            | 997/3870 [00:00<00:02, 1381.28it/s][A
     29%|██████████▉                          | 1140/3870 [00:00<00:01, 1396.19it/s][A
     33%|████████████▎                        | 1284/3870 [00:00<00:01, 1406.98it/s][A
     37%|█████████████▋                       | 1427/3870 [00:01<00:01, 1412.35it/s][A
     41%|███████████████                      | 1571/3870 [00:01<00:01, 1418.37it/s][A
     44%|████████████████▍                    | 1714/3870 [00:01<00:01, 1421.87it/s][A
     48%|█████████████████▊                   | 1858/3870 [00:01<00:01, 1424.62it/s][A
     52%|███████████████████▏                 | 2001/3870 [00:01<00:01, 1426.21it/s][A
     55%|████████████████████▌                | 2145/3870 [00:01<00:01, 1427.50it/s][A
     59%|█████████████████████▊               | 2288/3870 [00:01<00:01, 1428.23it/s][A
     63%|███████████████████████▎             | 2432/3870 [00:01<00:01, 1428.93it/s][A
     67%|████████████████████████▌            | 2575/3870 [00:01<00:00, 1368.40it/s][A
     70%|█████████████████████████▉           | 2713/3870 [00:01<00:00, 1328.75it/s][A
     74%|███████████████████████████▎         | 2857/3870 [00:02<00:00, 1358.04it/s][A
     78%|████████████████████████████▋        | 3000/3870 [00:02<00:00, 1377.95it/s][A
     81%|██████████████████████████████       | 3143/3870 [00:02<00:00, 1392.60it/s][A
     85%|███████████████████████████████▍     | 3286/3870 [00:02<00:00, 1402.26it/s][A
     89%|████████████████████████████████▊    | 3429/3870 [00:02<00:00, 1409.36it/s][A
     92%|██████████████████████████████████▏  | 3572/3870 [00:02<00:00, 1414.71it/s][A
     96%|███████████████████████████████████▌ | 3715/3870 [00:02<00:00, 1418.87it/s][A
    100%|█████████████████████████████████████| 3870/3870 [00:02<00:00, 1398.51it/s][A
    Testing DataLoader 0:  32%|██████▊              | 10/31 [02:57<06:13,  0.06it/s]Sampling. The number of nodes to sample is 4317.
    Sampling on one graph took 9.685555696487427 seconds.
    
      0%|                                                  | 0/4317 [00:00<?, ?it/s][A
      3%|▉                                     | 112/4317 [00:00<00:03, 1118.05it/s][A
      6%|██▏                                   | 243/4317 [00:00<00:03, 1227.11it/s][A
      9%|███▎                                  | 374/4317 [00:00<00:03, 1264.06it/s][A
     12%|████▍                                 | 506/4317 [00:00<00:02, 1282.70it/s][A
     15%|█████▌                                | 638/4317 [00:00<00:02, 1293.77it/s][A
     18%|██████▊                               | 769/4317 [00:00<00:02, 1298.64it/s][A
     21%|███████▉                              | 900/4317 [00:00<00:02, 1300.70it/s][A
     24%|████████▊                            | 1031/4317 [00:00<00:02, 1300.38it/s][A
     27%|█████████▉                           | 1162/4317 [00:00<00:02, 1302.05it/s][A
     30%|███████████                          | 1293/4317 [00:01<00:02, 1301.07it/s][A
     33%|████████████▏                        | 1424/4317 [00:01<00:02, 1296.84it/s][A
     36%|█████████████▎                       | 1555/4317 [00:01<00:02, 1300.04it/s][A
     39%|██████████████▍                      | 1686/4317 [00:01<00:02, 1248.69it/s][A
     42%|███████████████▌                     | 1815/4317 [00:01<00:01, 1258.83it/s][A
     45%|████████████████▋                    | 1946/4317 [00:01<00:01, 1272.47it/s][A
     48%|█████████████████▊                   | 2077/4317 [00:01<00:01, 1281.93it/s][A
     51%|██████████████████▉                  | 2208/4317 [00:01<00:01, 1288.85it/s][A
     54%|████████████████████                 | 2339/4317 [00:01<00:01, 1294.20it/s][A
     57%|█████████████████████▏               | 2470/4317 [00:01<00:01, 1298.83it/s][A
     60%|██████████████████████▎              | 2601/4317 [00:02<00:01, 1300.86it/s][A
     63%|███████████████████████▍             | 2732/4317 [00:02<00:01, 1303.22it/s][A
     66%|████████████████████████▌            | 2863/4317 [00:02<00:01, 1303.48it/s][A
     69%|█████████████████████████▋           | 2994/4317 [00:02<00:01, 1305.17it/s][A
     72%|██████████████████████████▊          | 3125/4317 [00:02<00:00, 1304.92it/s][A
     75%|███████████████████████████▉         | 3256/4317 [00:02<00:00, 1300.62it/s][A
     78%|█████████████████████████████        | 3387/4317 [00:02<00:00, 1248.00it/s][A
     81%|██████████████████████████████       | 3513/4317 [00:02<00:00, 1215.31it/s][A
     84%|███████████████████████████████▏     | 3644/4317 [00:02<00:00, 1240.66it/s][A
     87%|████████████████████████████████▎    | 3776/4317 [00:02<00:00, 1261.35it/s][A
     91%|█████████████████████████████████▍   | 3907/4317 [00:03<00:00, 1273.42it/s][A
     94%|██████████████████████████████████▌  | 4038/4317 [00:03<00:00, 1284.09it/s][A
     97%|███████████████████████████████████▋ | 4169/4317 [00:03<00:00, 1290.95it/s][A
    100%|█████████████████████████████████████| 4317/4317 [00:03<00:00, 1282.47it/s][A
    Testing DataLoader 0:  35%|███████▍             | 11/31 [03:15<05:55,  0.06it/s]Sampling. The number of nodes to sample is 4081.
    Sampling on one graph took 8.961014032363892 seconds.
    
      0%|                                                  | 0/4081 [00:00<?, ?it/s][A
      3%|█                                     | 118/4081 [00:00<00:03, 1174.57it/s][A
      6%|██▎                                   | 255/4081 [00:00<00:02, 1287.31it/s][A
     10%|███▋                                  | 392/4081 [00:00<00:02, 1324.84it/s][A
     13%|████▉                                 | 529/4081 [00:00<00:02, 1341.40it/s][A
     16%|██████▏                               | 667/4081 [00:00<00:02, 1351.91it/s][A
     20%|███████▍                              | 803/4081 [00:00<00:02, 1350.72it/s][A
     23%|████████▊                             | 940/4081 [00:00<00:02, 1356.95it/s][A
     26%|█████████▊                           | 1078/4081 [00:00<00:02, 1362.41it/s][A
     30%|███████████                          | 1215/4081 [00:00<00:02, 1364.58it/s][A
     33%|████████████▎                        | 1353/4081 [00:01<00:01, 1368.00it/s][A
     37%|█████████████▌                       | 1491/4081 [00:01<00:01, 1371.46it/s][A
     40%|██████████████▊                      | 1629/4081 [00:01<00:01, 1367.38it/s][A
     43%|████████████████                     | 1767/4081 [00:01<00:01, 1370.76it/s][A
     47%|█████████████████▎                   | 1905/4081 [00:01<00:01, 1320.03it/s][A
     50%|██████████████████▌                  | 2043/4081 [00:01<00:01, 1336.89it/s][A
     53%|███████████████████▊                 | 2181/4081 [00:01<00:01, 1349.02it/s][A
     57%|█████████████████████                | 2319/4081 [00:01<00:01, 1357.33it/s][A
     60%|██████████████████████▎              | 2457/4081 [00:01<00:01, 1363.17it/s][A
     64%|███████████████████████▌             | 2595/4081 [00:01<00:01, 1365.65it/s][A
     67%|████████████████████████▊            | 2733/4081 [00:02<00:00, 1368.33it/s][A
     70%|██████████████████████████           | 2871/4081 [00:02<00:00, 1371.05it/s][A
     74%|███████████████████████████▎         | 3009/4081 [00:02<00:00, 1372.46it/s][A
     77%|████████████████████████████▌        | 3147/4081 [00:02<00:00, 1372.12it/s][A
     80%|█████████████████████████████▊       | 3285/4081 [00:02<00:00, 1372.64it/s][A
     84%|███████████████████████████████      | 3423/4081 [00:02<00:00, 1371.63it/s][A
     87%|████████████████████████████████▎    | 3561/4081 [00:02<00:00, 1372.62it/s][A
     91%|█████████████████████████████████▌   | 3699/4081 [00:02<00:00, 1318.33it/s][A
     94%|██████████████████████████████████▊  | 3838/4081 [00:02<00:00, 1336.31it/s][A
    100%|█████████████████████████████████████| 4081/4081 [00:03<00:00, 1353.84it/s][A
    Testing DataLoader 0:  39%|████████▏            | 12/31 [03:31<05:34,  0.06it/s]Sampling. The number of nodes to sample is 4052.
    Sampling on one graph took 8.934802770614624 seconds.
    
      0%|                                                  | 0/4052 [00:00<?, ?it/s][A
      3%|█▎                                    | 136/4052 [00:00<00:02, 1359.89it/s][A
      7%|██▌                                   | 274/4052 [00:00<00:02, 1367.50it/s][A
     10%|███▊                                  | 412/4052 [00:00<00:02, 1372.65it/s][A
     14%|█████▏                                | 550/4052 [00:00<00:02, 1374.76it/s][A
     17%|██████▍                               | 688/4052 [00:00<00:02, 1375.99it/s][A
     20%|███████▋                              | 826/4052 [00:00<00:02, 1375.48it/s][A
     24%|█████████                             | 964/4052 [00:00<00:02, 1375.33it/s][A
     27%|██████████                           | 1102/4052 [00:00<00:02, 1375.45it/s][A
     31%|███████████▎                         | 1240/4052 [00:00<00:02, 1374.43it/s][A
     34%|████████████▌                        | 1378/4052 [00:01<00:01, 1374.64it/s][A
     37%|█████████████▊                       | 1516/4052 [00:01<00:01, 1314.65it/s][A
     41%|███████████████                      | 1654/4052 [00:01<00:01, 1332.50it/s][A
     44%|████████████████▎                    | 1792/4052 [00:01<00:01, 1344.94it/s][A
     48%|█████████████████▌                   | 1930/4052 [00:01<00:01, 1352.75it/s][A
     51%|██████████████████▉                  | 2069/4052 [00:01<00:01, 1360.69it/s][A
     54%|████████████████████▏                | 2207/4052 [00:01<00:01, 1365.28it/s][A
     58%|█████████████████████▍               | 2345/4052 [00:01<00:01, 1369.58it/s][A
     61%|██████████████████████▋              | 2483/4052 [00:01<00:01, 1372.48it/s][A
     65%|███████████████████████▉             | 2621/4052 [00:01<00:01, 1374.04it/s][A
     68%|█████████████████████████▏           | 2759/4052 [00:02<00:00, 1374.78it/s][A
     71%|██████████████████████████▍          | 2897/4052 [00:02<00:00, 1374.76it/s][A
     75%|███████████████████████████▋         | 3036/4052 [00:02<00:00, 1376.74it/s][A
     78%|████████████████████████████▉        | 3174/4052 [00:02<00:00, 1376.94it/s][A
     82%|██████████████████████████████▏      | 3312/4052 [00:02<00:00, 1319.30it/s][A
     85%|███████████████████████████████▌     | 3450/4052 [00:02<00:00, 1335.36it/s][A
     89%|████████████████████████████████▊    | 3588/4052 [00:02<00:00, 1347.05it/s][A
     92%|██████████████████████████████████   | 3725/4052 [00:02<00:00, 1353.71it/s][A
     95%|███████████████████████████████████▎ | 3863/4052 [00:02<00:00, 1359.62it/s][A
    100%|█████████████████████████████████████| 4052/4052 [00:02<00:00, 1361.20it/s][A
    Testing DataLoader 0:  42%|████████▊            | 13/31 [03:47<05:15,  0.06it/s]Sampling. The number of nodes to sample is 3564.
    Sampling on one graph took 8.647794723510742 seconds.
    
      0%|                                                  | 0/3564 [00:00<?, ?it/s][A
      4%|█▋                                    | 153/3564 [00:00<00:02, 1524.53it/s][A
      9%|███▎                                  | 307/3564 [00:00<00:02, 1527.61it/s][A
     13%|████▉                                 | 461/3564 [00:00<00:02, 1532.61it/s][A
     17%|██████▌                               | 615/3564 [00:00<00:01, 1533.94it/s][A
     22%|████████▏                             | 769/3564 [00:00<00:01, 1534.65it/s][A
     26%|█████████▊                            | 923/3564 [00:00<00:01, 1534.55it/s][A
     30%|███████████▏                         | 1077/3564 [00:00<00:01, 1536.15it/s][A
     35%|████████████▊                        | 1231/3564 [00:00<00:01, 1535.02it/s][A
     39%|██████████████▍                      | 1385/3564 [00:00<00:01, 1535.20it/s][A
     43%|███████████████▉                     | 1539/3564 [00:01<00:01, 1534.38it/s][A
     48%|█████████████████▌                   | 1693/3564 [00:01<00:01, 1417.12it/s][A
     52%|███████████████████▏                 | 1845/3564 [00:01<00:01, 1445.03it/s][A
     56%|████████████████████▊                | 1999/3564 [00:01<00:01, 1471.39it/s][A
     60%|██████████████████████▎              | 2153/3564 [00:01<00:00, 1489.82it/s][A
     65%|███████████████████████▉             | 2307/3564 [00:01<00:00, 1502.09it/s][A
     69%|█████████████████████████▌           | 2461/3564 [00:01<00:00, 1511.37it/s][A
     73%|███████████████████████████▏         | 2615/3564 [00:01<00:00, 1518.21it/s][A
     78%|████████████████████████████▊        | 2770/3564 [00:01<00:00, 1525.49it/s][A
     82%|██████████████████████████████▎      | 2924/3564 [00:01<00:00, 1529.14it/s][A
     86%|███████████████████████████████▉     | 3078/3564 [00:02<00:00, 1529.65it/s][A
     91%|█████████████████████████████████▌   | 3233/3564 [00:02<00:00, 1533.18it/s][A
     95%|███████████████████████████████████▏ | 3387/3564 [00:02<00:00, 1533.79it/s][A
    100%|█████████████████████████████████████| 3564/3564 [00:02<00:00, 1516.11it/s][A
    Testing DataLoader 0:  45%|█████████▍           | 14/31 [04:01<04:53,  0.06it/s]Sampling. The number of nodes to sample is 4186.
    Sampling on one graph took 9.05336594581604 seconds.
    
      0%|                                                  | 0/4186 [00:00<?, ?it/s][A
      3%|█▏                                    | 132/4186 [00:00<00:03, 1314.75it/s][A
      6%|██▍                                   | 266/4186 [00:00<00:02, 1325.58it/s][A
     10%|███▋                                  | 400/4186 [00:00<00:02, 1330.48it/s][A
     13%|████▊                                 | 534/4186 [00:00<00:02, 1332.98it/s][A
     16%|██████                                | 668/4186 [00:00<00:02, 1333.35it/s][A
     19%|███████▎                              | 802/4186 [00:00<00:02, 1333.76it/s][A
     22%|████████▍                             | 936/4186 [00:00<00:02, 1334.24it/s][A
     26%|█████████▍                           | 1070/4186 [00:00<00:02, 1335.42it/s][A
     29%|██████████▋                          | 1204/4186 [00:00<00:02, 1280.77it/s][A
     32%|███████████▊                         | 1338/4186 [00:01<00:02, 1297.74it/s][A
     35%|█████████████                        | 1472/4186 [00:01<00:02, 1310.20it/s][A
     38%|██████████████▏                      | 1606/4186 [00:01<00:01, 1317.52it/s][A
     42%|███████████████▍                     | 1740/4186 [00:01<00:01, 1322.77it/s][A
     45%|████████████████▌                    | 1874/4186 [00:01<00:01, 1327.37it/s][A
     48%|█████████████████▊                   | 2009/4186 [00:01<00:01, 1332.59it/s][A
     51%|██████████████████▉                  | 2144/4186 [00:01<00:01, 1336.79it/s][A
     54%|████████████████████▏                | 2280/4186 [00:01<00:01, 1341.26it/s][A
     58%|█████████████████████▎               | 2416/4186 [00:01<00:01, 1344.37it/s][A
     61%|██████████████████████▌              | 2551/4186 [00:01<00:01, 1345.07it/s][A
     64%|███████████████████████▋             | 2686/4186 [00:02<00:01, 1346.05it/s][A
     67%|████████████████████████▉            | 2821/4186 [00:02<00:01, 1346.01it/s][A
     71%|██████████████████████████▏          | 2956/4186 [00:02<00:00, 1271.75it/s][A
     74%|███████████████████████████▎         | 3085/4186 [00:02<00:00, 1262.94it/s][A
     77%|████████████████████████████▍        | 3220/4186 [00:02<00:00, 1288.04it/s][A
     80%|█████████████████████████████▋       | 3355/4186 [00:02<00:00, 1305.08it/s][A
     83%|██████████████████████████████▊      | 3491/4186 [00:02<00:00, 1318.51it/s][A
     87%|████████████████████████████████     | 3626/4186 [00:02<00:00, 1327.64it/s][A
     90%|█████████████████████████████████▏   | 3761/4186 [00:02<00:00, 1332.76it/s][A
     93%|██████████████████████████████████▍  | 3896/4186 [00:02<00:00, 1337.72it/s][A
     96%|███████████████████████████████████▋ | 4031/4186 [00:03<00:00, 1341.23it/s][A
    100%|█████████████████████████████████████| 4186/4186 [00:03<00:00, 1324.21it/s][A
    Testing DataLoader 0:  48%|██████████▏          | 15/31 [04:18<04:35,  0.06it/s]Sampling. The number of nodes to sample is 2800.
    Sampling on one graph took 8.491618633270264 seconds.
    
      0%|                                                  | 0/2800 [00:00<?, ?it/s][A
      7%|██▍                                   | 184/2800 [00:00<00:01, 1832.86it/s][A
     13%|█████                                 | 369/2800 [00:00<00:01, 1841.76it/s][A
     20%|███████▌                              | 554/2800 [00:00<00:01, 1841.77it/s][A
     26%|██████████                            | 739/2800 [00:00<00:01, 1843.29it/s][A
     33%|████████████▌                         | 924/2800 [00:00<00:01, 1840.84it/s][A
     40%|██████████████▋                      | 1109/2800 [00:00<00:00, 1839.93it/s][A
     46%|█████████████████                    | 1294/2800 [00:00<00:00, 1842.62it/s][A
     53%|███████████████████▌                 | 1479/2800 [00:00<00:00, 1761.35it/s][A
     59%|█████████████████████▉               | 1664/2800 [00:00<00:00, 1785.34it/s][A
     66%|████████████████████████▍            | 1849/2800 [00:01<00:00, 1802.83it/s][A
     73%|██████████████████████████▉          | 2035/2800 [00:01<00:00, 1819.08it/s][A
     79%|█████████████████████████████▎       | 2220/2800 [00:01<00:00, 1825.87it/s][A
     86%|███████████████████████████████▊     | 2404/2800 [00:01<00:00, 1827.87it/s][A
     92%|██████████████████████████████████▏  | 2590/2800 [00:01<00:00, 1835.10it/s][A
    100%|█████████████████████████████████████| 2800/2800 [00:01<00:00, 1825.02it/s][A
    Testing DataLoader 0:  52%|██████████▊          | 16/31 [04:30<04:13,  0.06it/s]Sampling. The number of nodes to sample is 4529.
    Sampling on one graph took 9.83433985710144 seconds.
    
      0%|                                                  | 0/4529 [00:00<?, ?it/s][A
      2%|▉                                     | 108/4529 [00:00<00:04, 1070.99it/s][A
      5%|█▉                                    | 233/4529 [00:00<00:03, 1170.51it/s][A
      8%|███                                   | 358/4529 [00:00<00:03, 1202.67it/s][A
     11%|████                                  | 483/4529 [00:00<00:03, 1218.20it/s][A
     13%|█████                                 | 609/4529 [00:00<00:03, 1231.44it/s][A
     16%|██████▏                               | 735/4529 [00:00<00:03, 1240.58it/s][A
     19%|███████▏                              | 861/4529 [00:00<00:02, 1246.90it/s][A
     22%|████████▎                             | 987/4529 [00:00<00:02, 1248.53it/s][A
     25%|█████████                            | 1113/4529 [00:00<00:02, 1251.69it/s][A
     27%|██████████                           | 1239/4529 [00:01<00:02, 1252.72it/s][A
     30%|███████████▏                         | 1365/4529 [00:01<00:02, 1254.19it/s][A
     33%|████████████▏                        | 1492/4529 [00:01<00:02, 1256.79it/s][A
     36%|█████████████▏                       | 1619/4529 [00:01<00:02, 1258.19it/s][A
     39%|██████████████▎                      | 1745/4529 [00:01<00:02, 1206.72it/s][A
     41%|███████████████▎                     | 1867/4529 [00:01<00:02, 1174.31it/s][A
     44%|████████████████▎                    | 1994/4529 [00:01<00:02, 1199.38it/s][A
     47%|█████████████████▎                   | 2121/4529 [00:01<00:01, 1217.21it/s][A
     50%|██████████████████▎                  | 2247/4529 [00:01<00:01, 1229.63it/s][A
     52%|███████████████████▍                 | 2374/4529 [00:01<00:01, 1238.70it/s][A
     55%|████████████████████▍                | 2501/4529 [00:02<00:01, 1246.05it/s][A
     58%|█████████████████████▍               | 2628/4529 [00:02<00:01, 1251.04it/s][A
     61%|██████████████████████▌              | 2755/4529 [00:02<00:01, 1254.59it/s][A
     64%|███████████████████████▌             | 2882/4529 [00:02<00:01, 1256.55it/s][A
     66%|████████████████████████▌            | 3009/4529 [00:02<00:01, 1259.07it/s][A
     69%|█████████████████████████▌           | 3136/4529 [00:02<00:01, 1260.17it/s][A
     72%|██████████████████████████▋          | 3263/4529 [00:02<00:01, 1261.00it/s][A
     75%|███████████████████████████▋         | 3390/4529 [00:02<00:00, 1208.53it/s][A
     78%|████████████████████████████▋        | 3512/4529 [00:02<00:00, 1173.19it/s][A
     80%|█████████████████████████████▋       | 3639/4529 [00:02<00:00, 1199.93it/s][A
     83%|██████████████████████████████▊      | 3766/4529 [00:03<00:00, 1219.71it/s][A
     86%|███████████████████████████████▊     | 3893/4529 [00:03<00:00, 1232.35it/s][A
     89%|████████████████████████████████▊    | 4020/4529 [00:03<00:00, 1241.72it/s][A
     92%|█████████████████████████████████▉   | 4147/4529 [00:03<00:00, 1249.48it/s][A
     94%|██████████████████████████████████▉  | 4273/4529 [00:03<00:00, 1251.18it/s][A
     97%|███████████████████████████████████▉ | 4400/4529 [00:03<00:00, 1255.59it/s][A
    100%|█████████████████████████████████████| 4529/4529 [00:03<00:00, 1234.63it/s][A
    Testing DataLoader 0:  55%|███████████▌         | 17/31 [04:49<03:58,  0.06it/s]Sampling. The number of nodes to sample is 5021.
    Sampling on one graph took 10.437591314315796 seconds.
    
      0%|                                                  | 0/5021 [00:00<?, ?it/s][A
      2%|▊                                     | 115/5021 [00:00<00:04, 1142.43it/s][A
      5%|█▋                                    | 230/5021 [00:00<00:04, 1144.96it/s][A
      7%|██▌                                   | 346/5021 [00:00<00:04, 1147.71it/s][A
      9%|███▍                                  | 461/5021 [00:00<00:03, 1145.26it/s][A
     11%|████▎                                 | 576/5021 [00:00<00:04, 1090.28it/s][A
     14%|█████▏                                | 686/5021 [00:00<00:04, 1055.37it/s][A
     16%|██████                                | 802/5021 [00:00<00:03, 1085.49it/s][A
     18%|██████▉                               | 917/5021 [00:00<00:03, 1105.17it/s][A
     21%|███████▌                             | 1032/5021 [00:00<00:03, 1116.13it/s][A
     23%|████████▍                            | 1147/5021 [00:01<00:03, 1125.57it/s][A
     25%|█████████▎                           | 1263/5021 [00:01<00:03, 1133.66it/s][A
     27%|██████████▏                          | 1379/5021 [00:01<00:03, 1138.99it/s][A
     30%|███████████                          | 1495/5021 [00:01<00:03, 1143.18it/s][A
     32%|███████████▊                         | 1611/5021 [00:01<00:02, 1145.36it/s][A
     34%|████████████▋                        | 1727/5021 [00:01<00:02, 1148.09it/s][A
     37%|█████████████▌                       | 1843/5021 [00:01<00:02, 1148.94it/s][A
     39%|██████████████▍                      | 1959/5021 [00:01<00:02, 1149.87it/s][A
     41%|███████████████▎                     | 2075/5021 [00:01<00:02, 1103.50it/s][A
     44%|████████████████                     | 2186/5021 [00:01<00:02, 1071.68it/s][A
     46%|████████████████▉                    | 2302/5021 [00:02<00:02, 1095.83it/s][A
     48%|█████████████████▊                   | 2418/5021 [00:02<00:02, 1113.47it/s][A
     50%|██████████████████▋                  | 2534/5021 [00:02<00:02, 1125.92it/s][A
     53%|███████████████████▌                 | 2651/5021 [00:02<00:02, 1136.29it/s][A
     55%|████████████████████▍                | 2767/5021 [00:02<00:01, 1141.31it/s][A
     57%|█████████████████████▏               | 2883/5021 [00:02<00:01, 1145.26it/s][A
     60%|██████████████████████               | 2999/5021 [00:02<00:01, 1148.36it/s][A
     62%|██████████████████████▉              | 3115/5021 [00:02<00:01, 1151.44it/s][A
     64%|███████████████████████▊             | 3231/5021 [00:02<00:01, 1153.79it/s][A
     67%|████████████████████████▋            | 3347/5021 [00:02<00:01, 1154.41it/s][A
     69%|█████████████████████████▌           | 3463/5021 [00:03<00:01, 1155.29it/s][A
     71%|██████████████████████████▎          | 3579/5021 [00:03<00:01, 1078.54it/s][A
     73%|███████████████████████████▏         | 3689/5021 [00:03<00:01, 1082.64it/s][A
     76%|████████████████████████████         | 3805/5021 [00:03<00:01, 1103.76it/s][A
     78%|████████████████████████████▉        | 3921/5021 [00:03<00:00, 1119.20it/s][A
     80%|█████████████████████████████▋       | 4037/5021 [00:03<00:00, 1129.53it/s][A
     83%|██████████████████████████████▌      | 4153/5021 [00:03<00:00, 1138.17it/s][A
     85%|███████████████████████████████▍     | 4269/5021 [00:03<00:00, 1142.10it/s][A
     87%|████████████████████████████████▎    | 4385/5021 [00:03<00:00, 1145.05it/s][A
     90%|█████████████████████████████████▏   | 4501/5021 [00:03<00:00, 1148.53it/s][A
     92%|██████████████████████████████████   | 4616/5021 [00:04<00:00, 1148.86it/s][A
     94%|██████████████████████████████████▊  | 4732/5021 [00:04<00:00, 1150.83it/s][A
     97%|███████████████████████████████████▋ | 4848/5021 [00:04<00:00, 1151.56it/s][A
    100%|█████████████████████████████████████| 5021/5021 [00:04<00:00, 1126.28it/s][A
    Testing DataLoader 0:  58%|████████████▏        | 18/31 [05:10<03:44,  0.06it/s]Sampling. The number of nodes to sample is 4991.
    Sampling on one graph took 10.436870574951172 seconds.
    
      0%|                                                  | 0/4991 [00:00<?, ?it/s][A
      2%|▉                                     | 116/4991 [00:00<00:04, 1153.57it/s][A
      5%|█▊                                    | 232/4991 [00:00<00:04, 1061.78it/s][A
      7%|██▋                                   | 348/4991 [00:00<00:04, 1102.31it/s][A
      9%|███▌                                  | 464/4991 [00:00<00:04, 1123.30it/s][A
     12%|████▍                                 | 580/4991 [00:00<00:03, 1133.82it/s][A
     14%|█████▎                                | 696/4991 [00:00<00:03, 1141.51it/s][A
     16%|██████▏                               | 812/4991 [00:00<00:03, 1144.68it/s][A
     19%|███████                               | 928/4991 [00:00<00:03, 1148.49it/s][A
     21%|███████▋                             | 1044/4991 [00:00<00:03, 1150.72it/s][A
     23%|████████▌                            | 1160/4991 [00:01<00:03, 1152.35it/s][A
     26%|█████████▍                           | 1276/4991 [00:01<00:03, 1153.42it/s][A
     28%|██████████▎                          | 1392/4991 [00:01<00:03, 1153.50it/s][A
     30%|███████████▏                         | 1508/4991 [00:01<00:03, 1108.29it/s][A
     32%|████████████                         | 1620/4991 [00:01<00:03, 1065.63it/s][A
     35%|████████████▊                        | 1736/4991 [00:01<00:02, 1091.59it/s][A
     37%|█████████████▋                       | 1852/4991 [00:01<00:02, 1110.28it/s][A
     39%|██████████████▌                      | 1968/4991 [00:01<00:02, 1122.24it/s][A
     42%|███████████████▍                     | 2084/4991 [00:01<00:02, 1130.80it/s][A
     44%|████████████████▎                    | 2200/4991 [00:01<00:02, 1137.20it/s][A
     46%|█████████████████▏                   | 2316/4991 [00:02<00:02, 1141.95it/s][A
     49%|██████████████████                   | 2432/4991 [00:02<00:02, 1145.41it/s][A
     51%|██████████████████▉                  | 2547/4991 [00:02<00:02, 1144.56it/s][A
     53%|███████████████████▋                 | 2663/4991 [00:02<00:02, 1147.66it/s][A
     56%|████████████████████▌                | 2779/4991 [00:02<00:01, 1149.43it/s][A
     58%|█████████████████████▍               | 2895/4991 [00:02<00:01, 1150.27it/s][A
     60%|██████████████████████▎              | 3011/4991 [00:02<00:01, 1100.20it/s][A
     63%|███████████████████████▏             | 3122/4991 [00:02<00:01, 1063.97it/s][A
     65%|███████████████████████▉             | 3237/4991 [00:02<00:01, 1088.07it/s][A
     67%|████████████████████████▊            | 3352/4991 [00:02<00:01, 1105.31it/s][A
     69%|█████████████████████████▋           | 3468/4991 [00:03<00:01, 1118.88it/s][A
     72%|██████████████████████████▌          | 3583/4991 [00:03<00:01, 1127.28it/s][A
     74%|███████████████████████████▍         | 3699/4991 [00:03<00:01, 1134.39it/s][A
     76%|████████████████████████████▎        | 3814/4991 [00:03<00:01, 1138.64it/s][A
     79%|█████████████████████████████▏       | 3929/4991 [00:03<00:00, 1141.18it/s][A
     81%|█████████████████████████████▉       | 4044/4991 [00:03<00:00, 1143.78it/s][A
     83%|██████████████████████████████▊      | 4159/4991 [00:03<00:00, 1145.53it/s][A
     86%|███████████████████████████████▋     | 4274/4991 [00:03<00:00, 1137.81it/s][A
     88%|████████████████████████████████▌    | 4389/4991 [00:03<00:00, 1141.09it/s][A
     90%|█████████████████████████████████▍   | 4504/4991 [00:04<00:00, 1089.08it/s][A
     92%|██████████████████████████████████▏  | 4614/4991 [00:04<00:00, 1062.14it/s][A
     95%|███████████████████████████████████  | 4730/4991 [00:04<00:00, 1089.26it/s][A
     97%|███████████████████████████████████▉ | 4847/4991 [00:04<00:00, 1110.24it/s][A
    100%|█████████████████████████████████████| 4991/4991 [00:04<00:00, 1123.65it/s][A
    Testing DataLoader 0:  61%|████████████▊        | 19/31 [05:31<03:29,  0.06it/s]Sampling. The number of nodes to sample is 4389.
    Sampling on one graph took 9.666241645812988 seconds.
    
      0%|                                                  | 0/4389 [00:00<?, ?it/s][A
      3%|█                                     | 129/4389 [00:00<00:03, 1289.30it/s][A
      6%|██▏                                   | 259/4389 [00:00<00:03, 1292.27it/s][A
      9%|███▎                                  | 389/4389 [00:00<00:03, 1293.54it/s][A
     12%|████▍                                 | 519/4389 [00:00<00:02, 1294.89it/s][A
     15%|█████▌                                | 649/4389 [00:00<00:02, 1289.06it/s][A
     18%|██████▋                               | 779/4389 [00:00<00:02, 1291.87it/s][A
     21%|███████▊                              | 909/4389 [00:00<00:02, 1231.09it/s][A
     24%|████████▋                            | 1033/4389 [00:00<00:02, 1199.55it/s][A
     27%|█████████▊                           | 1164/4389 [00:00<00:02, 1230.81it/s][A
     30%|██████████▉                          | 1295/4389 [00:01<00:02, 1252.42it/s][A
     32%|████████████                         | 1425/4389 [00:01<00:02, 1266.22it/s][A
     35%|█████████████                        | 1556/4389 [00:01<00:02, 1277.66it/s][A
     38%|██████████████▏                      | 1687/4389 [00:01<00:02, 1285.19it/s][A
     41%|███████████████▎                     | 1818/4389 [00:01<00:01, 1290.71it/s][A
     44%|████████████████▍                    | 1949/4389 [00:01<00:01, 1294.54it/s][A
     47%|█████████████████▌                   | 2080/4389 [00:01<00:01, 1296.97it/s][A
     50%|██████████████████▋                  | 2211/4389 [00:01<00:01, 1298.72it/s][A
     53%|███████████████████▋                 | 2342/4389 [00:01<00:01, 1300.19it/s][A
     56%|████████████████████▊                | 2473/4389 [00:01<00:01, 1300.59it/s][A
     59%|█████████████████████▉               | 2604/4389 [00:02<00:01, 1255.03it/s][A
     62%|███████████████████████              | 2732/4389 [00:02<00:01, 1261.44it/s][A
     65%|████████████████████████▏            | 2863/4389 [00:02<00:01, 1273.72it/s][A
     68%|█████████████████████████▏           | 2993/4389 [00:02<00:01, 1280.94it/s][A
     71%|██████████████████████████▎          | 3124/4389 [00:02<00:00, 1286.85it/s][A
     74%|███████████████████████████▍         | 3253/4389 [00:02<00:00, 1286.59it/s][A
     77%|████████████████████████████▌        | 3384/4389 [00:02<00:00, 1291.40it/s][A
     80%|█████████████████████████████▋       | 3515/4389 [00:02<00:00, 1296.14it/s][A
     83%|██████████████████████████████▋      | 3645/4389 [00:02<00:00, 1297.16it/s][A
     86%|███████████████████████████████▊     | 3776/4389 [00:02<00:00, 1299.80it/s][A
     89%|████████████████████████████████▉    | 3907/4389 [00:03<00:00, 1299.02it/s][A
     92%|██████████████████████████████████   | 4037/4389 [00:03<00:00, 1299.10it/s][A
     95%|███████████████████████████████████▏ | 4168/4389 [00:03<00:00, 1299.92it/s][A
    100%|█████████████████████████████████████| 4389/4389 [00:03<00:00, 1277.61it/s][A
    Testing DataLoader 0:  65%|█████████████▌       | 20/31 [05:49<03:12,  0.06it/s]Sampling. The number of nodes to sample is 3560.
    Sampling on one graph took 8.603728294372559 seconds.
    
      0%|                                                  | 0/3560 [00:00<?, ?it/s][A
      4%|█▋                                    | 153/3560 [00:00<00:02, 1521.98it/s][A
      9%|███▎                                  | 306/3560 [00:00<00:02, 1523.15it/s][A
     13%|████▉                                 | 460/3560 [00:00<00:02, 1528.01it/s][A
     17%|██████▌                               | 613/3560 [00:00<00:01, 1528.55it/s][A
     22%|████████▏                             | 766/3560 [00:00<00:01, 1528.26it/s][A
     26%|█████████▊                            | 919/3560 [00:00<00:01, 1528.12it/s][A
     30%|███████████▏                         | 1072/3560 [00:00<00:01, 1528.12it/s][A
     34%|████████████▋                        | 1225/3560 [00:00<00:01, 1526.71it/s][A
     39%|██████████████▎                      | 1378/3560 [00:00<00:01, 1526.96it/s][A
     43%|███████████████▉                     | 1531/3560 [00:01<00:01, 1526.77it/s][A
     47%|█████████████████▌                   | 1684/3560 [00:01<00:01, 1459.18it/s][A
     52%|███████████████████                  | 1838/3560 [00:01<00:01, 1480.20it/s][A
     56%|████████████████████▋                | 1992/3560 [00:01<00:01, 1496.90it/s][A
     60%|██████████████████████▎              | 2146/3560 [00:01<00:00, 1508.71it/s][A
     65%|███████████████████████▉             | 2299/3560 [00:01<00:00, 1513.64it/s][A
     69%|█████████████████████████▍           | 2453/3560 [00:01<00:00, 1520.27it/s][A
     73%|███████████████████████████          | 2607/3560 [00:01<00:00, 1525.78it/s][A
     78%|████████████████████████████▋        | 2762/3560 [00:01<00:00, 1530.29it/s][A
     82%|██████████████████████████████▎      | 2916/3560 [00:01<00:00, 1532.67it/s][A
     86%|███████████████████████████████▉     | 3071/3560 [00:02<00:00, 1535.97it/s][A
     91%|█████████████████████████████████▌   | 3226/3560 [00:02<00:00, 1537.78it/s][A
     95%|███████████████████████████████████▏ | 3380/3560 [00:02<00:00, 1537.88it/s][A
    100%|█████████████████████████████████████| 3560/3560 [00:02<00:00, 1516.05it/s][A
    Testing DataLoader 0:  68%|██████████████▏      | 21/31 [06:04<02:53,  0.06it/s]Sampling. The number of nodes to sample is 1706.
    Sampling on one graph took 8.579698085784912 seconds.
    
      0%|                                                  | 0/1706 [00:00<?, ?it/s][A
     15%|█████▋                                | 253/1706 [00:00<00:00, 2528.66it/s][A
     30%|███████████▎                          | 508/1706 [00:00<00:00, 2537.32it/s][A
     45%|████████████████▉                     | 763/1706 [00:00<00:00, 2539.36it/s][A
     60%|██████████████████████               | 1017/1706 [00:00<00:00, 2410.03it/s][A
     74%|███████████████████████████▎         | 1259/1706 [00:00<00:00, 2371.12it/s][A
    100%|█████████████████████████████████████| 1706/1706 [00:00<00:00, 2436.72it/s][A
    Testing DataLoader 0:  71%|██████████████▉      | 22/31 [06:14<02:33,  0.06it/s]Sampling. The number of nodes to sample is 3296.
    Sampling on one graph took 8.61502194404602 seconds.
    
      0%|                                                  | 0/3296 [00:00<?, ?it/s][A
      5%|█▉                                    | 163/3296 [00:00<00:01, 1623.71it/s][A
     10%|███▊                                  | 326/3296 [00:00<00:01, 1621.47it/s][A
     15%|█████▋                                | 490/3296 [00:00<00:01, 1629.68it/s][A
     20%|███████▌                              | 653/3296 [00:00<00:01, 1548.84it/s][A
     25%|█████████▍                            | 818/3296 [00:00<00:01, 1581.31it/s][A
     30%|███████████▎                          | 983/3296 [00:00<00:01, 1601.96it/s][A
     35%|████████████▉                        | 1147/3296 [00:00<00:01, 1613.39it/s][A
     40%|██████████████▋                      | 1312/3296 [00:00<00:01, 1623.02it/s][A
     45%|████████████████▌                    | 1476/3296 [00:00<00:01, 1626.51it/s][A
     50%|██████████████████▍                  | 1641/3296 [00:01<00:01, 1633.21it/s][A
     55%|████████████████████▎                | 1806/3296 [00:01<00:00, 1635.35it/s][A
     60%|██████████████████████▏              | 1971/3296 [00:01<00:00, 1638.00it/s][A
     65%|███████████████████████▉             | 2136/3296 [00:01<00:00, 1640.72it/s][A
     70%|█████████████████████████▊           | 2301/3296 [00:01<00:00, 1640.39it/s][A
     75%|███████████████████████████▋         | 2466/3296 [00:01<00:00, 1640.61it/s][A
     80%|█████████████████████████████▌       | 2631/3296 [00:01<00:00, 1641.93it/s][A
     85%|███████████████████████████████▍     | 2796/3296 [00:01<00:00, 1544.16it/s][A
     90%|█████████████████████████████████▏   | 2952/3296 [00:01<00:00, 1542.09it/s][A
     95%|██████████████████████████████████▉  | 3117/3296 [00:01<00:00, 1570.52it/s][A
    100%|█████████████████████████████████████| 3296/3296 [00:02<00:00, 1606.44it/s][A
    Testing DataLoader 0:  74%|███████████████▌     | 23/31 [06:27<02:14,  0.06it/s]Sampling. The number of nodes to sample is 5155.
    Sampling on one graph took 10.464542865753174 seconds.
    
      0%|                                                  | 0/5155 [00:00<?, ?it/s][A
      2%|▊                                     | 112/5155 [00:00<00:04, 1119.42it/s][A
      4%|█▋                                    | 225/5155 [00:00<00:04, 1124.28it/s][A
      7%|██▍                                   | 338/5155 [00:00<00:04, 1126.48it/s][A
      9%|███▎                                  | 451/5155 [00:00<00:04, 1126.90it/s][A
     11%|████▏                                 | 565/5155 [00:00<00:04, 1129.62it/s][A
     13%|████▉                                 | 678/5155 [00:00<00:04, 1079.16it/s][A
     15%|█████▊                                | 787/5155 [00:00<00:04, 1042.53it/s][A
     17%|██████▋                               | 901/5155 [00:00<00:03, 1069.66it/s][A
     20%|███████▎                             | 1015/5155 [00:00<00:03, 1089.52it/s][A
     22%|████████                             | 1129/5155 [00:01<00:03, 1101.87it/s][A
     24%|████████▉                            | 1243/5155 [00:01<00:03, 1111.24it/s][A
     26%|█████████▋                           | 1357/5155 [00:01<00:03, 1117.41it/s][A
     29%|██████████▌                          | 1470/5155 [00:01<00:03, 1120.52it/s][A
     31%|███████████▎                         | 1584/5155 [00:01<00:03, 1123.46it/s][A
     33%|████████████▏                        | 1697/5155 [00:01<00:03, 1122.62it/s][A
     35%|████████████▉                        | 1811/5155 [00:01<00:02, 1126.06it/s][A
     37%|█████████████▊                       | 1925/5155 [00:01<00:02, 1127.78it/s][A
     40%|██████████████▋                      | 2038/5155 [00:01<00:02, 1128.16it/s][A
     42%|███████████████▍                     | 2151/5155 [00:01<00:02, 1069.24it/s][A
     44%|████████████████▏                    | 2259/5155 [00:02<00:02, 1061.60it/s][A
     46%|█████████████████                    | 2373/5155 [00:02<00:02, 1082.45it/s][A
     48%|█████████████████▊                   | 2487/5155 [00:02<00:02, 1096.49it/s][A
     50%|██████████████████▋                  | 2601/5155 [00:02<00:02, 1107.11it/s][A
     53%|███████████████████▍                 | 2714/5155 [00:02<00:02, 1113.40it/s][A
     55%|████████████████████▎                | 2828/5155 [00:02<00:02, 1118.91it/s][A
     57%|█████████████████████                | 2942/5155 [00:02<00:01, 1123.14it/s][A
     59%|█████████████████████▉               | 3056/5155 [00:02<00:01, 1126.43it/s][A
     61%|██████████████████████▊              | 3170/5155 [00:02<00:01, 1127.82it/s][A
     64%|███████████████████████▌             | 3283/5155 [00:02<00:01, 1128.43it/s][A
     66%|████████████████████████▍            | 3397/5155 [00:03<00:01, 1129.32it/s][A
     68%|█████████████████████████▏           | 3510/5155 [00:03<00:01, 1089.37it/s][A
     70%|█████████████████████████▉           | 3622/5155 [00:03<00:01, 1097.27it/s][A
     72%|██████████████████████████▊          | 3736/5155 [00:03<00:01, 1107.37it/s][A
     75%|███████████████████████████▋         | 3850/5155 [00:03<00:01, 1114.63it/s][A
     77%|████████████████████████████▍        | 3964/5155 [00:03<00:01, 1120.17it/s][A
     79%|█████████████████████████████▎       | 4078/5155 [00:03<00:00, 1123.72it/s][A
     81%|██████████████████████████████       | 4191/5155 [00:03<00:00, 1124.82it/s][A
     83%|██████████████████████████████▉      | 4304/5155 [00:03<00:00, 1126.18it/s][A
     86%|███████████████████████████████▋     | 4418/5155 [00:03<00:00, 1128.07it/s][A
     88%|████████████████████████████████▌    | 4531/5155 [00:04<00:00, 1128.15it/s][A
     90%|█████████████████████████████████▎   | 4645/5155 [00:04<00:00, 1129.06it/s][A
     92%|██████████████████████████████████▏  | 4759/5155 [00:04<00:00, 1129.87it/s][A
     95%|██████████████████████████████████▉  | 4872/5155 [00:04<00:00, 1129.88it/s][A
     97%|███████████████████████████████████▊ | 4985/5155 [00:04<00:00, 1081.71it/s][A
    100%|█████████████████████████████████████| 5155/5155 [00:04<00:00, 1106.64it/s][A
    Testing DataLoader 0:  77%|████████████████▎    | 24/31 [06:49<01:59,  0.06it/s]Sampling. The number of nodes to sample is 3733.
    Sampling on one graph took 8.775777816772461 seconds.
    
      0%|                                                  | 0/3733 [00:00<?, ?it/s][A
      4%|█▌                                    | 148/3733 [00:00<00:02, 1477.01it/s][A
      8%|███                                   | 297/3733 [00:00<00:02, 1484.26it/s][A
     12%|████▌                                 | 446/3733 [00:00<00:02, 1402.58it/s][A
     16%|██████                                | 595/3733 [00:00<00:02, 1433.88it/s][A
     20%|███████▌                              | 745/3733 [00:00<00:02, 1455.76it/s][A
     24%|█████████                             | 895/3733 [00:00<00:01, 1467.56it/s][A
     28%|██████████▎                          | 1045/3733 [00:00<00:01, 1475.24it/s][A
     32%|███████████▊                         | 1195/3733 [00:00<00:01, 1480.58it/s][A
     36%|█████████████▎                       | 1344/3733 [00:00<00:01, 1482.53it/s][A
     40%|██████████████▊                      | 1494/3733 [00:01<00:01, 1485.93it/s][A
     44%|████████████████▎                    | 1644/3733 [00:01<00:01, 1488.26it/s][A
     48%|█████████████████▊                   | 1793/3733 [00:01<00:01, 1487.58it/s][A
     52%|███████████████████▎                 | 1943/3733 [00:01<00:01, 1490.26it/s][A
     56%|████████████████████▋                | 2093/3733 [00:01<00:01, 1490.26it/s][A
     60%|██████████████████████▏              | 2243/3733 [00:01<00:00, 1491.33it/s][A
     64%|███████████████████████▋             | 2393/3733 [00:01<00:00, 1492.62it/s][A
     68%|█████████████████████████▏           | 2543/3733 [00:01<00:00, 1433.17it/s][A
     72%|██████████████████████████▋          | 2693/3733 [00:01<00:00, 1450.70it/s][A
     76%|████████████████████████████▏        | 2843/3733 [00:01<00:00, 1462.38it/s][A
     80%|█████████████████████████████▋       | 2993/3733 [00:02<00:00, 1470.51it/s][A
     84%|███████████████████████████████▏     | 3142/3733 [00:02<00:00, 1475.88it/s][A
     88%|████████████████████████████████▌    | 3291/3733 [00:02<00:00, 1479.75it/s][A
     92%|██████████████████████████████████   | 3441/3733 [00:02<00:00, 1484.70it/s][A
    100%|█████████████████████████████████████| 3733/3733 [00:02<00:00, 1474.92it/s][A
    Testing DataLoader 0:  81%|████████████████▉    | 25/31 [07:04<01:41,  0.06it/s]Sampling. The number of nodes to sample is 3780.
    Sampling on one graph took 8.686441421508789 seconds.
    
      0%|                                                  | 0/3780 [00:00<?, ?it/s][A
      3%|█▏                                    | 123/3780 [00:00<00:02, 1228.96it/s][A
      7%|██▋                                   | 269/3780 [00:00<00:02, 1360.94it/s][A
     11%|████▏                                 | 415/3780 [00:00<00:02, 1405.62it/s][A
     15%|█████▋                                | 561/3780 [00:00<00:02, 1424.43it/s][A
     19%|███████                               | 707/3780 [00:00<00:02, 1435.33it/s][A
     23%|████████▌                             | 852/3780 [00:00<00:02, 1439.88it/s][A
     26%|██████████                            | 998/3780 [00:00<00:01, 1444.94it/s][A
     30%|███████████▏                         | 1144/3780 [00:00<00:01, 1449.32it/s][A
     34%|████████████▋                        | 1290/3780 [00:00<00:01, 1451.12it/s][A
     38%|██████████████                       | 1437/3780 [00:01<00:01, 1454.04it/s][A
     42%|███████████████▍                     | 1583/3780 [00:01<00:01, 1454.84it/s][A
     46%|████████████████▉                    | 1729/3780 [00:01<00:01, 1455.01it/s][A
     50%|██████████████████▎                  | 1875/3780 [00:01<00:01, 1393.68it/s][A
     53%|███████████████████▊                 | 2021/3780 [00:01<00:01, 1412.67it/s][A
     57%|█████████████████████▏               | 2167/3780 [00:01<00:01, 1425.34it/s][A
     61%|██████████████████████▋              | 2313/3780 [00:01<00:01, 1434.75it/s][A
     65%|████████████████████████             | 2458/3780 [00:01<00:00, 1438.30it/s][A
     69%|█████████████████████████▍           | 2603/3780 [00:01<00:00, 1440.82it/s][A
     73%|██████████████████████████▉          | 2749/3780 [00:01<00:00, 1444.00it/s][A
     77%|████████████████████████████▎        | 2895/3780 [00:02<00:00, 1447.32it/s][A
     80%|█████████████████████████████▊       | 3040/3780 [00:02<00:00, 1446.80it/s][A
     84%|███████████████████████████████▏     | 3185/3780 [00:02<00:00, 1444.76it/s][A
     88%|████████████████████████████████▌    | 3330/3780 [00:02<00:00, 1445.96it/s][A
     92%|██████████████████████████████████   | 3476/3780 [00:02<00:00, 1447.67it/s][A
     96%|███████████████████████████████████▍ | 3621/3780 [00:02<00:00, 1448.15it/s][A
    100%|█████████████████████████████████████| 3780/3780 [00:02<00:00, 1428.01it/s][A
    Testing DataLoader 0:  84%|█████████████████▌   | 26/31 [07:19<01:24,  0.06it/s]Sampling. The number of nodes to sample is 2905.
    Sampling on one graph took 8.51256513595581 seconds.
    
      0%|                                                  | 0/2905 [00:00<?, ?it/s][A
      6%|██▎                                   | 177/2905 [00:00<00:01, 1764.01it/s][A
     12%|████▋                                 | 354/2905 [00:00<00:01, 1766.37it/s][A
     18%|██████▉                               | 531/2905 [00:00<00:01, 1765.32it/s][A
     24%|█████████▎                            | 708/2905 [00:00<00:01, 1765.80it/s][A
     30%|███████████▌                          | 885/2905 [00:00<00:01, 1765.57it/s][A
     37%|█████████████▌                       | 1062/2905 [00:00<00:01, 1765.56it/s][A
     43%|███████████████▊                     | 1240/2905 [00:00<00:00, 1768.66it/s][A
     49%|██████████████████                   | 1417/2905 [00:00<00:00, 1696.11it/s][A
     55%|████████████████████▎                | 1595/2905 [00:00<00:00, 1720.03it/s][A
     61%|██████████████████████▌              | 1772/2905 [00:01<00:00, 1734.60it/s][A
     67%|████████████████████████▊            | 1950/2905 [00:01<00:00, 1746.53it/s][A
     73%|███████████████████████████          | 2127/2905 [00:01<00:00, 1752.72it/s][A
     79%|█████████████████████████████▎       | 2305/2905 [00:01<00:00, 1759.26it/s][A
     86%|███████████████████████████████▋     | 2484/2905 [00:01<00:00, 1766.04it/s][A
     92%|█████████████████████████████████▉   | 2662/2905 [00:01<00:00, 1769.67it/s][A
    100%|█████████████████████████████████████| 2905/2905 [00:01<00:00, 1756.11it/s][A
    Testing DataLoader 0:  87%|██████████████████▎  | 27/31 [07:32<01:06,  0.06it/s]Sampling. The number of nodes to sample is 3192.
    Sampling on one graph took 8.436640739440918 seconds.
    
      0%|                                                  | 0/3192 [00:00<?, ?it/s][A
      4%|█▋                                    | 140/3192 [00:00<00:02, 1397.38it/s][A
     10%|███▋                                  | 306/3192 [00:00<00:01, 1548.91it/s][A
     15%|█████▌                                | 471/3192 [00:00<00:01, 1594.01it/s][A
     20%|███████▌                              | 637/3192 [00:00<00:01, 1618.27it/s][A
     25%|█████████▌                            | 803/3192 [00:00<00:01, 1632.91it/s][A
     30%|███████████▌                          | 969/3192 [00:00<00:01, 1639.37it/s][A
     36%|█████████████▏                       | 1135/3192 [00:00<00:01, 1645.00it/s][A
     41%|███████████████                      | 1301/3192 [00:00<00:01, 1647.92it/s][A
     46%|█████████████████                    | 1467/3192 [00:00<00:01, 1651.27it/s][A
     51%|██████████████████▉                  | 1634/3192 [00:01<00:00, 1655.45it/s][A
     56%|████████████████████▊                | 1800/3192 [00:01<00:00, 1652.68it/s][A
     62%|██████████████████████▊              | 1966/3192 [00:01<00:00, 1653.94it/s][A
     67%|████████████████████████▋            | 2132/3192 [00:01<00:00, 1649.54it/s][A
     72%|██████████████████████████▋          | 2297/3192 [00:01<00:00, 1535.39it/s][A
     77%|████████████████████████████▌        | 2462/3192 [00:01<00:00, 1566.33it/s][A
     82%|██████████████████████████████▍      | 2629/3192 [00:01<00:00, 1593.69it/s][A
     88%|████████████████████████████████▍    | 2795/3192 [00:01<00:00, 1612.66it/s][A
     93%|██████████████████████████████████▎  | 2962/3192 [00:01<00:00, 1627.19it/s][A
    100%|█████████████████████████████████████| 3192/3192 [00:01<00:00, 1620.37it/s][A
    Testing DataLoader 0:  90%|██████████████████▉  | 28/31 [07:45<00:49,  0.06it/s]Sampling. The number of nodes to sample is 4057.
    Sampling on one graph took 8.84314751625061 seconds.
    
      0%|                                                  | 0/4057 [00:00<?, ?it/s][A
      3%|█▎                                    | 136/4057 [00:00<00:02, 1350.29it/s][A
      7%|██▌                                   | 272/4057 [00:00<00:02, 1353.14it/s][A
     10%|███▊                                  | 408/4057 [00:00<00:02, 1355.56it/s][A
     13%|█████                                 | 545/4057 [00:00<00:02, 1357.42it/s][A
     17%|██████▍                               | 681/4057 [00:00<00:02, 1357.86it/s][A
     20%|███████▋                              | 817/4057 [00:00<00:02, 1357.69it/s][A
     23%|████████▉                             | 953/4057 [00:00<00:02, 1357.73it/s][A
     27%|█████████▉                           | 1089/4057 [00:00<00:02, 1299.45it/s][A
     30%|███████████▏                         | 1220/4057 [00:00<00:02, 1261.27it/s][A
     33%|████████████▍                        | 1357/4057 [00:01<00:02, 1290.93it/s][A
     37%|█████████████▌                       | 1493/4057 [00:01<00:01, 1309.40it/s][A
     40%|██████████████▊                      | 1629/4057 [00:01<00:01, 1324.22it/s][A
     44%|████████████████                     | 1766/4057 [00:01<00:01, 1335.72it/s][A
     47%|█████████████████▎                   | 1903/4057 [00:01<00:01, 1343.45it/s][A
     50%|██████████████████▌                  | 2040/4057 [00:01<00:01, 1350.87it/s][A
     54%|███████████████████▊                 | 2177/4057 [00:01<00:01, 1355.20it/s][A
     57%|█████████████████████                | 2313/4057 [00:01<00:01, 1356.60it/s][A
     60%|██████████████████████▎              | 2450/4057 [00:01<00:01, 1358.48it/s][A
     64%|███████████████████████▌             | 2587/4057 [00:01<00:01, 1359.01it/s][A
     67%|████████████████████████▊            | 2723/4057 [00:02<00:00, 1358.60it/s][A
     70%|██████████████████████████           | 2859/4057 [00:02<00:00, 1311.70it/s][A
     74%|███████████████████████████▎         | 2995/4057 [00:02<00:00, 1323.77it/s][A
     77%|████████████████████████████▌        | 3132/4057 [00:02<00:00, 1335.41it/s][A
     81%|█████████████████████████████▊       | 3269/4057 [00:02<00:00, 1343.46it/s][A
     84%|███████████████████████████████      | 3406/4057 [00:02<00:00, 1349.02it/s][A
     87%|████████████████████████████████▎    | 3543/4057 [00:02<00:00, 1353.88it/s][A
     91%|█████████████████████████████████▌   | 3680/4057 [00:02<00:00, 1356.32it/s][A
     94%|██████████████████████████████████▊  | 3817/4057 [00:02<00:00, 1358.01it/s][A
    100%|█████████████████████████████████████| 4057/4057 [00:03<00:00, 1341.93it/s][A
    Testing DataLoader 0:  94%|███████████████████▋ | 29/31 [08:01<00:33,  0.06it/s]Sampling. The number of nodes to sample is 1160.
    Sampling on one graph took 8.490202903747559 seconds.
    
      0%|                                                  | 0/1160 [00:00<?, ?it/s][A
     22%|████████▌                             | 260/1160 [00:00<00:00, 2599.87it/s][A
     49%|██████████████████▍                   | 563/1160 [00:00<00:00, 2852.24it/s][A
    100%|█████████████████████████████████████| 1160/1160 [00:00<00:00, 2955.87it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  97%|████████████████████▎| 30/31 [08:10<00:16,  0.06it/s]Sampling. The number of nodes to sample is 3316.
    Sampling on one graph took 8.59573769569397 seconds.
    
      0%|                                                  | 0/3316 [00:00<?, ?it/s][A
      5%|█▊                                    | 160/3316 [00:00<00:01, 1593.29it/s][A
     10%|███▋                                  | 321/3316 [00:00<00:01, 1597.58it/s][A
     15%|█████▌                                | 482/3316 [00:00<00:01, 1602.60it/s][A
     19%|███████▎                              | 643/3316 [00:00<00:01, 1603.68it/s][A
     24%|█████████▏                            | 804/3316 [00:00<00:01, 1596.99it/s][A
     29%|███████████                           | 966/3316 [00:00<00:01, 1602.47it/s][A
     34%|████████████▌                        | 1127/3316 [00:00<00:01, 1537.97it/s][A
     39%|██████████████▍                      | 1289/3316 [00:00<00:01, 1561.37it/s][A
     44%|████████████████▏                    | 1451/3316 [00:00<00:01, 1576.93it/s][A
     49%|█████████████████▉                   | 1613/3316 [00:01<00:01, 1587.38it/s][A
     53%|███████████████████▊                 | 1774/3316 [00:01<00:00, 1591.32it/s][A
     58%|█████████████████████▌               | 1935/3316 [00:01<00:00, 1596.49it/s][A
     63%|███████████████████████▍             | 2096/3316 [00:01<00:00, 1598.99it/s][A
     68%|█████████████████████████▏           | 2258/3316 [00:01<00:00, 1603.09it/s][A
     73%|██████████████████████████▉          | 2419/3316 [00:01<00:00, 1604.22it/s][A
     78%|████████████████████████████▊        | 2581/3316 [00:01<00:00, 1606.30it/s][A
     83%|██████████████████████████████▌      | 2743/3316 [00:01<00:00, 1607.53it/s][A
     88%|████████████████████████████████▍    | 2904/3316 [00:01<00:00, 1607.27it/s][A
     92%|██████████████████████████████████▏  | 3065/3316 [00:01<00:00, 1605.09it/s][A
    100%|█████████████████████████████████████| 3316/3316 [00:02<00:00, 1584.15it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0: 100%|█████████████████████| 31/31 [08:24<00:00,  0.06it/s]
    Testing checkpoint: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=249.ckpt
    Epoch index: 249
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    [rank: 0] Seed set to 0
    Restoring states from the checkpoint path at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=249.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from the checkpoint at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=249.ckpt
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:492: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
      warnings.warn(  # warn only once
    Testing DataLoader 0:   0%|                              | 0/31 [00:00<?, ?it/s]Sampling. The number of nodes to sample is 3851.
    Sampling on one graph took 8.91994023323059 seconds.
    
      0%|                                                  | 0/3851 [00:00<?, ?it/s][A
      2%|▉                                       | 86/3851 [00:00<00:04, 855.67it/s][A
      6%|██▎                                   | 229/3851 [00:00<00:03, 1191.20it/s][A
     10%|███▋                                  | 373/3851 [00:00<00:02, 1302.04it/s][A
     13%|█████                                 | 517/3851 [00:00<00:02, 1354.82it/s][A
     17%|██████▌                               | 662/3851 [00:00<00:02, 1385.68it/s][A
     21%|███████▉                              | 806/3851 [00:00<00:02, 1403.99it/s][A
     25%|█████████▎                            | 950/3851 [00:00<00:02, 1414.72it/s][A
     28%|██████████▍                          | 1092/3851 [00:00<00:02, 1359.67it/s][A
     32%|███████████▉                         | 1236/3851 [00:00<00:01, 1383.45it/s][A
     36%|█████████████▎                       | 1381/3851 [00:01<00:01, 1401.57it/s][A
     40%|██████████████▋                      | 1525/3851 [00:01<00:01, 1413.05it/s][A
     43%|████████████████                     | 1669/3851 [00:01<00:01, 1419.70it/s][A
     47%|█████████████████▍                   | 1813/3851 [00:01<00:01, 1424.54it/s][A
     51%|██████████████████▊                  | 1957/3851 [00:01<00:01, 1427.46it/s][A
     55%|████████████████████▏                | 2100/3851 [00:01<00:01, 1426.65it/s][A
     58%|█████████████████████▌               | 2243/3851 [00:01<00:01, 1427.25it/s][A
     62%|██████████████████████▉              | 2386/3851 [00:01<00:01, 1346.62it/s][A
     65%|████████████████████████▏            | 2522/3851 [00:01<00:00, 1336.59it/s][A
     69%|█████████████████████████▌           | 2660/3851 [00:01<00:00, 1347.33it/s][A
     73%|██████████████████████████▊          | 2796/3851 [00:02<00:00, 1317.77it/s][A
     76%|████████████████████████████▏        | 2940/3851 [00:02<00:00, 1350.88it/s][A
     80%|█████████████████████████████▋       | 3084/3851 [00:02<00:00, 1374.95it/s][A
     84%|███████████████████████████████      | 3227/3851 [00:02<00:00, 1390.71it/s][A
     87%|████████████████████████████████▎    | 3368/3851 [00:02<00:00, 1393.95it/s][A
     91%|█████████████████████████████████▋   | 3508/3851 [00:02<00:00, 1394.80it/s][A
     95%|███████████████████████████████████  | 3649/3851 [00:02<00:00, 1398.81it/s][A
    100%|█████████████████████████████████████| 3851/3851 [00:02<00:00, 1377.96it/s][A
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:15<07:56,  0.06it/s]Sampling. The number of nodes to sample is 670.
    Sampling on one graph took 8.474803924560547 seconds.
    
      0%|                                                   | 0/670 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████| 670/670 [00:00<00:00, 3983.51it/s][A
    Testing DataLoader 0:   6%|█▍                    | 2/31 [00:24<06:01,  0.08it/s]Sampling. The number of nodes to sample is 4360.
    Sampling on one graph took 9.695926904678345 seconds.
    
      0%|                                                  | 0/4360 [00:00<?, ?it/s][A
      3%|█                                     | 129/4360 [00:00<00:03, 1281.40it/s][A
      6%|██▎                                   | 259/4360 [00:00<00:03, 1289.70it/s][A
      9%|███▍                                  | 389/4360 [00:00<00:03, 1292.84it/s][A
     12%|████▌                                 | 519/4360 [00:00<00:02, 1294.47it/s][A
     15%|█████▋                                | 649/4360 [00:00<00:02, 1294.21it/s][A
     18%|██████▊                               | 779/4360 [00:00<00:02, 1295.55it/s][A
     21%|███████▉                              | 909/4360 [00:00<00:02, 1238.57it/s][A
     24%|████████▊                            | 1034/4360 [00:00<00:02, 1205.15it/s][A
     27%|█████████▉                           | 1164/4360 [00:00<00:02, 1232.47it/s][A
     30%|██████████▉                          | 1294/4360 [00:01<00:02, 1252.24it/s][A
     33%|████████████                         | 1424/4360 [00:01<00:02, 1265.19it/s][A
     36%|█████████████▏                       | 1554/4360 [00:01<00:02, 1274.78it/s][A
     39%|██████████████▎                      | 1684/4360 [00:01<00:02, 1281.66it/s][A
     42%|███████████████▍                     | 1814/4360 [00:01<00:01, 1286.30it/s][A
     45%|████████████████▍                    | 1944/4360 [00:01<00:01, 1288.57it/s][A
     48%|█████████████████▌                   | 2074/4360 [00:01<00:01, 1290.26it/s][A
     51%|██████████████████▋                  | 2204/4360 [00:01<00:01, 1291.35it/s][A
     54%|███████████████████▊                 | 2334/4360 [00:01<00:01, 1290.92it/s][A
     57%|████████████████████▉                | 2464/4360 [00:01<00:01, 1292.23it/s][A
     59%|██████████████████████               | 2594/4360 [00:02<00:01, 1247.73it/s][A
     62%|███████████████████████              | 2721/4360 [00:02<00:01, 1251.83it/s][A
     65%|████████████████████████▏            | 2851/4360 [00:02<00:01, 1265.17it/s][A
     68%|█████████████████████████▎           | 2980/4360 [00:02<00:01, 1272.32it/s][A
     71%|██████████████████████████▍          | 3110/4360 [00:02<00:00, 1279.98it/s][A
     74%|███████████████████████████▍         | 3240/4360 [00:02<00:00, 1285.39it/s][A
     77%|████████████████████████████▌        | 3370/4360 [00:02<00:00, 1288.94it/s][A
     80%|█████████████████████████████▋       | 3500/4360 [00:02<00:00, 1290.50it/s][A
     83%|██████████████████████████████▊      | 3630/4360 [00:02<00:00, 1292.62it/s][A
     86%|███████████████████████████████▉     | 3760/4360 [00:02<00:00, 1284.66it/s][A
     89%|█████████████████████████████████    | 3890/4360 [00:03<00:00, 1287.27it/s][A
     92%|██████████████████████████████████   | 4019/4360 [00:03<00:00, 1287.57it/s][A
     95%|███████████████████████████████████▏ | 4149/4360 [00:03<00:00, 1290.04it/s][A
    100%|█████████████████████████████████████| 4360/4360 [00:03<00:00, 1272.44it/s][A
    Testing DataLoader 0:  10%|██▏                   | 3/31 [00:42<06:39,  0.07it/s]Sampling. The number of nodes to sample is 5180.
    Sampling on one graph took 10.456341028213501 seconds.
    
      0%|                                                  | 0/5180 [00:00<?, ?it/s][A
      2%|▊                                     | 111/5180 [00:00<00:04, 1102.75it/s][A
      4%|█▋                                    | 223/5180 [00:00<00:04, 1107.84it/s][A
      6%|██▍                                   | 335/5180 [00:00<00:04, 1110.55it/s][A
      9%|███▎                                  | 447/5180 [00:00<00:04, 1053.17it/s][A
     11%|████                                  | 558/5180 [00:00<00:04, 1070.32it/s][A
     13%|████▉                                 | 670/5180 [00:00<00:04, 1086.13it/s][A
     15%|█████▋                                | 782/5180 [00:00<00:04, 1096.07it/s][A
     17%|██████▌                               | 894/5180 [00:00<00:03, 1102.54it/s][A
     19%|███████▏                             | 1006/5180 [00:00<00:03, 1107.32it/s][A
     22%|███████▉                             | 1117/5180 [00:01<00:03, 1108.03it/s][A
     24%|████████▊                            | 1229/5180 [00:01<00:03, 1111.10it/s][A
     26%|█████████▌                           | 1341/5180 [00:01<00:03, 1111.54it/s][A
     28%|██████████▍                          | 1453/5180 [00:01<00:03, 1112.01it/s][A
     30%|███████████▏                         | 1565/5180 [00:01<00:03, 1112.57it/s][A
     32%|███████████▉                         | 1677/5180 [00:01<00:03, 1113.74it/s][A
     35%|████████████▊                        | 1789/5180 [00:01<00:03, 1113.34it/s][A
     37%|█████████████▌                       | 1901/5180 [00:01<00:02, 1113.24it/s][A
     39%|██████████████▍                      | 2013/5180 [00:01<00:02, 1113.52it/s][A
     41%|███████████████▏                     | 2125/5180 [00:01<00:02, 1113.79it/s][A
     43%|███████████████▉                     | 2237/5180 [00:02<00:02, 1114.46it/s][A
     45%|████████████████▊                    | 2349/5180 [00:02<00:02, 1114.42it/s][A
     48%|█████████████████▌                   | 2461/5180 [00:02<00:02, 1113.97it/s][A
     50%|██████████████████▍                  | 2573/5180 [00:02<00:02, 1114.78it/s][A
     52%|███████████████████▏                 | 2685/5180 [00:02<00:02, 1114.76it/s][A
     54%|███████████████████▉                 | 2797/5180 [00:02<00:02, 1114.44it/s][A
     56%|████████████████████▊                | 2909/5180 [00:02<00:02, 1114.77it/s][A
     58%|█████████████████████▌               | 3021/5180 [00:02<00:01, 1114.51it/s][A
     60%|██████████████████████▍              | 3133/5180 [00:02<00:01, 1114.51it/s][A
     63%|███████████████████████▏             | 3245/5180 [00:02<00:01, 1067.79it/s][A
     65%|███████████████████████▉             | 3357/5180 [00:03<00:01, 1081.37it/s][A
     67%|████████████████████████▊            | 3469/5180 [00:03<00:01, 1090.52it/s][A
     69%|█████████████████████████▌           | 3581/5180 [00:03<00:01, 1097.19it/s][A
     71%|██████████████████████████▎          | 3691/5180 [00:03<00:01, 1096.60it/s][A
     73%|███████████████████████████▏         | 3803/5180 [00:03<00:01, 1102.53it/s][A
     76%|███████████████████████████▉         | 3915/5180 [00:03<00:01, 1105.39it/s][A
     78%|████████████████████████████▊        | 4027/5180 [00:03<00:01, 1109.55it/s][A
     80%|█████████████████████████████▌       | 4139/5180 [00:03<00:00, 1112.46it/s][A
     82%|██████████████████████████████▎      | 4252/5180 [00:03<00:00, 1114.91it/s][A
     84%|███████████████████████████████▏     | 4365/5180 [00:03<00:00, 1117.07it/s][A
     86%|███████████████████████████████▉     | 4478/5180 [00:04<00:00, 1119.01it/s][A
     89%|████████████████████████████████▊    | 4590/5180 [00:04<00:00, 1082.84it/s][A
     91%|█████████████████████████████████▌   | 4699/5180 [00:04<00:00, 1082.51it/s][A
     93%|██████████████████████████████████▎  | 4812/5180 [00:04<00:00, 1094.26it/s][A
     95%|███████████████████████████████████▏ | 4925/5180 [00:04<00:00, 1102.57it/s][A
     97%|███████████████████████████████████▉ | 5036/5180 [00:04<00:00, 1103.82it/s][A
    100%|█████████████████████████████████████| 5180/5180 [00:04<00:00, 1104.50it/s][A
    Testing DataLoader 0:  13%|██▊                   | 4/31 [01:04<07:16,  0.06it/s]Sampling. The number of nodes to sample is 4946.
    Sampling on one graph took 10.243072748184204 seconds.
    
      0%|                                                  | 0/4946 [00:00<?, ?it/s][A
      2%|▉                                     | 114/4946 [00:00<00:04, 1139.22it/s][A
      5%|█▊                                    | 229/4946 [00:00<00:04, 1143.81it/s][A
      7%|██▋                                   | 345/4946 [00:00<00:04, 1147.36it/s][A
      9%|███▌                                  | 461/4946 [00:00<00:03, 1148.91it/s][A
     12%|████▍                                 | 577/4946 [00:00<00:03, 1151.01it/s][A
     14%|█████▎                                | 693/4946 [00:00<00:03, 1152.77it/s][A
     16%|██████▏                               | 809/4946 [00:00<00:03, 1152.58it/s][A
     19%|███████                               | 925/4946 [00:00<00:03, 1149.59it/s][A
     21%|███████▊                             | 1041/4946 [00:00<00:03, 1151.69it/s][A
     23%|████████▋                            | 1157/4946 [00:01<00:03, 1105.27it/s][A
     26%|█████████▍                           | 1268/4946 [00:01<00:03, 1071.50it/s][A
     28%|██████████▎                          | 1384/4946 [00:01<00:03, 1096.25it/s][A
     30%|███████████▏                         | 1500/4946 [00:01<00:03, 1113.55it/s][A
     33%|████████████                         | 1616/4946 [00:01<00:02, 1126.22it/s][A
     35%|████████████▉                        | 1732/4946 [00:01<00:02, 1135.81it/s][A
     37%|█████████████▊                       | 1848/4946 [00:01<00:02, 1142.20it/s][A
     40%|██████████████▋                      | 1964/4946 [00:01<00:02, 1146.01it/s][A
     42%|███████████████▌                     | 2080/4946 [00:01<00:02, 1150.12it/s][A
     44%|████████████████▍                    | 2196/4946 [00:01<00:02, 1151.94it/s][A
     47%|█████████████████▎                   | 2312/4946 [00:02<00:02, 1153.17it/s][A
     49%|██████████████████▏                  | 2428/4946 [00:02<00:02, 1154.04it/s][A
     51%|███████████████████                  | 2544/4946 [00:02<00:02, 1154.84it/s][A
     54%|███████████████████▉                 | 2660/4946 [00:02<00:01, 1154.52it/s][A
     56%|████████████████████▊                | 2776/4946 [00:02<00:01, 1154.66it/s][A
     58%|█████████████████████▋               | 2892/4946 [00:02<00:01, 1107.67it/s][A
     61%|██████████████████████▌              | 3008/4946 [00:02<00:01, 1121.55it/s][A
     63%|███████████████████████▎             | 3124/4946 [00:02<00:01, 1130.78it/s][A
     66%|████████████████████████▏            | 3240/4946 [00:02<00:01, 1138.68it/s][A
     68%|█████████████████████████            | 3356/4946 [00:02<00:01, 1143.89it/s][A
     70%|█████████████████████████▉           | 3472/4946 [00:03<00:01, 1146.72it/s][A
     73%|██████████████████████████▊          | 3588/4946 [00:03<00:01, 1150.21it/s][A
     75%|███████████████████████████▋         | 3704/4946 [00:03<00:01, 1151.10it/s][A
     77%|████████████████████████████▌        | 3820/4946 [00:03<00:00, 1152.22it/s][A
     80%|█████████████████████████████▍       | 3936/4946 [00:03<00:00, 1152.54it/s][A
     82%|██████████████████████████████▎      | 4052/4946 [00:03<00:00, 1153.62it/s][A
     84%|███████████████████████████████▏     | 4168/4946 [00:03<00:00, 1117.89it/s][A
     87%|████████████████████████████████     | 4282/4946 [00:03<00:00, 1123.52it/s][A
     89%|████████████████████████████████▉    | 4398/4946 [00:03<00:00, 1132.99it/s][A
     91%|█████████████████████████████████▊   | 4514/4946 [00:03<00:00, 1139.30it/s][A
     94%|██████████████████████████████████▋  | 4629/4946 [00:04<00:00, 1141.34it/s][A
     96%|███████████████████████████████████▍ | 4745/4946 [00:04<00:00, 1145.43it/s][A
    100%|█████████████████████████████████████| 4946/4946 [00:04<00:00, 1139.44it/s][A
    Testing DataLoader 0:  16%|███▌                  | 5/31 [01:25<07:23,  0.06it/s]Sampling. The number of nodes to sample is 2037.
    Sampling on one graph took 8.753081798553467 seconds.
    
      0%|                                                  | 0/2037 [00:00<?, ?it/s][A
     11%|████▏                                 | 227/2037 [00:00<00:00, 2265.87it/s][A
     22%|████████▌                             | 457/2037 [00:00<00:00, 2285.89it/s][A
     34%|████████████▊                         | 687/2037 [00:00<00:00, 2291.02it/s][A
     45%|█████████████████                     | 917/2037 [00:00<00:00, 2291.00it/s][A
     56%|████████████████████▊                | 1147/2037 [00:00<00:00, 2289.95it/s][A
     68%|█████████████████████████            | 1377/2037 [00:00<00:00, 2290.74it/s][A
     79%|█████████████████████████████▏       | 1607/2037 [00:00<00:00, 2290.04it/s][A
    100%|█████████████████████████████████████| 2037/2037 [00:00<00:00, 2255.50it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  19%|████▎                 | 6/31 [01:36<06:40,  0.06it/s]Sampling. The number of nodes to sample is 5050.
    Sampling on one graph took 10.358797073364258 seconds.
    
      0%|                                                  | 0/5050 [00:00<?, ?it/s][A
      2%|▊                                     | 114/5050 [00:00<00:04, 1137.96it/s][A
      5%|█▋                                    | 229/5050 [00:00<00:04, 1139.81it/s][A
      7%|██▌                                   | 344/5050 [00:00<00:04, 1140.98it/s][A
      9%|███▍                                  | 459/5050 [00:00<00:04, 1141.22it/s][A
     11%|████▎                                 | 574/5050 [00:00<00:03, 1141.85it/s][A
     14%|█████▏                                | 689/5050 [00:00<00:03, 1141.86it/s][A
     16%|██████                                | 804/5050 [00:00<00:03, 1143.23it/s][A
     18%|██████▉                               | 919/5050 [00:00<00:03, 1143.68it/s][A
     20%|███████▌                             | 1034/5050 [00:00<00:03, 1091.89it/s][A
     23%|████████▍                            | 1144/5050 [00:01<00:03, 1060.51it/s][A
     25%|█████████▏                           | 1258/5050 [00:01<00:03, 1081.32it/s][A
     27%|██████████                           | 1372/5050 [00:01<00:03, 1095.90it/s][A
     29%|██████████▉                          | 1486/5050 [00:01<00:03, 1106.84it/s][A
     32%|███████████▋                         | 1600/5050 [00:01<00:03, 1114.43it/s][A
     34%|████████████▌                        | 1715/5050 [00:01<00:02, 1122.11it/s][A
     36%|█████████████▍                       | 1830/5050 [00:01<00:02, 1127.84it/s][A
     39%|██████████████▎                      | 1945/5050 [00:01<00:02, 1132.63it/s][A
     41%|███████████████                      | 2060/5050 [00:01<00:02, 1136.44it/s][A
     43%|███████████████▉                     | 2175/5050 [00:01<00:02, 1138.24it/s][A
     45%|████████████████▊                    | 2289/5050 [00:02<00:02, 1137.22it/s][A
     48%|█████████████████▌                   | 2404/5050 [00:02<00:02, 1138.98it/s][A
     50%|██████████████████▍                  | 2519/5050 [00:02<00:02, 1140.90it/s][A
     52%|███████████████████▎                 | 2634/5050 [00:02<00:02, 1093.52it/s][A
     54%|████████████████████▏                | 2748/5050 [00:02<00:02, 1105.80it/s][A
     57%|████████████████████▉                | 2862/5050 [00:02<00:01, 1114.35it/s][A
     59%|█████████████████████▊               | 2975/5050 [00:02<00:01, 1117.51it/s][A
     61%|██████████████████████▋              | 3090/5050 [00:02<00:01, 1125.07it/s][A
     63%|███████████████████████▍             | 3204/5050 [00:02<00:01, 1127.95it/s][A
     66%|████████████████████████▎            | 3319/5050 [00:02<00:01, 1133.05it/s][A
     68%|█████████████████████████▏           | 3434/5050 [00:03<00:01, 1136.24it/s][A
     70%|█████████████████████████▉           | 3548/5050 [00:03<00:01, 1133.60it/s][A
     73%|██████████████████████████▊          | 3663/5050 [00:03<00:01, 1136.73it/s][A
     75%|███████████████████████████▋         | 3778/5050 [00:03<00:01, 1139.30it/s][A
     77%|████████████████████████████▌        | 3893/5050 [00:03<00:01, 1142.12it/s][A
     79%|█████████████████████████████▎       | 4008/5050 [00:03<00:00, 1095.68it/s][A
     82%|██████████████████████████████▏      | 4123/5050 [00:03<00:00, 1111.03it/s][A
     84%|███████████████████████████████      | 4238/5050 [00:03<00:00, 1121.30it/s][A
     86%|███████████████████████████████▉     | 4353/5050 [00:03<00:00, 1129.12it/s][A
     88%|████████████████████████████████▋    | 4468/5050 [00:03<00:00, 1132.46it/s][A
     91%|█████████████████████████████████▌   | 4582/5050 [00:04<00:00, 1131.13it/s][A
     93%|██████████████████████████████████▍  | 4696/5050 [00:04<00:00, 1131.03it/s][A
     95%|███████████████████████████████████▏ | 4810/5050 [00:04<00:00, 1131.77it/s][A
     98%|████████████████████████████████████ | 4924/5050 [00:04<00:00, 1132.19it/s][A
    100%|█████████████████████████████████████| 5050/5050 [00:04<00:00, 1124.64it/s][A
    Testing DataLoader 0:  23%|████▉                 | 7/31 [01:57<06:42,  0.06it/s]Sampling. The number of nodes to sample is 5047.
    Sampling on one graph took 10.427898645401001 seconds.
    
      0%|                                                  | 0/5047 [00:00<?, ?it/s][A
      2%|▊                                     | 114/5047 [00:00<00:04, 1133.18it/s][A
      5%|█▋                                    | 229/5047 [00:00<00:04, 1142.17it/s][A
      7%|██▌                                   | 344/5047 [00:00<00:04, 1144.55it/s][A
      9%|███▍                                  | 459/5047 [00:00<00:04, 1113.34it/s][A
     11%|████▎                                 | 571/5047 [00:00<00:04, 1100.58it/s][A
     14%|█████▏                                | 687/5047 [00:00<00:03, 1118.21it/s][A
     16%|██████                                | 803/5047 [00:00<00:03, 1128.84it/s][A
     18%|██████▉                               | 919/5047 [00:00<00:03, 1136.22it/s][A
     21%|███████▌                             | 1035/5047 [00:00<00:03, 1141.04it/s][A
     23%|████████▍                            | 1150/5047 [00:01<00:03, 1143.43it/s][A
     25%|█████████▎                           | 1265/5047 [00:01<00:03, 1145.04it/s][A
     27%|██████████                           | 1381/5047 [00:01<00:03, 1147.67it/s][A
     30%|██████████▉                          | 1497/5047 [00:01<00:03, 1150.13it/s][A
     32%|███████████▊                         | 1613/5047 [00:01<00:02, 1152.29it/s][A
     34%|████████████▋                        | 1729/5047 [00:01<00:02, 1152.45it/s][A
     37%|█████████████▌                       | 1845/5047 [00:01<00:02, 1152.58it/s][A
     39%|██████████████▍                      | 1961/5047 [00:01<00:02, 1103.51it/s][A
     41%|███████████████▏                     | 2072/5047 [00:01<00:02, 1069.35it/s][A
     43%|████████████████                     | 2188/5047 [00:01<00:02, 1093.78it/s][A
     46%|████████████████▉                    | 2304/5047 [00:02<00:02, 1110.93it/s][A
     48%|█████████████████▋                   | 2418/5047 [00:02<00:02, 1117.75it/s][A
     50%|██████████████████▌                  | 2534/5047 [00:02<00:02, 1127.69it/s][A
     53%|███████████████████▍                 | 2650/5047 [00:02<00:02, 1134.97it/s][A
     55%|████████████████████▎                | 2766/5047 [00:02<00:01, 1140.59it/s][A
     57%|█████████████████████▏               | 2882/5047 [00:02<00:01, 1143.68it/s][A
     59%|█████████████████████▉               | 2997/5047 [00:02<00:01, 1145.43it/s][A
     62%|██████████████████████▊              | 3112/5047 [00:02<00:01, 1145.76it/s][A
     64%|███████████████████████▋             | 3227/5047 [00:02<00:01, 1144.41it/s][A
     66%|████████████████████████▌            | 3342/5047 [00:02<00:01, 1142.90it/s][A
     68%|█████████████████████████▎           | 3457/5047 [00:03<00:01, 1094.33it/s][A
     71%|██████████████████████████▏          | 3567/5047 [00:03<00:01, 1062.25it/s][A
     73%|██████████████████████████▉          | 3682/5047 [00:03<00:01, 1085.42it/s][A
     75%|███████████████████████████▊         | 3797/5047 [00:03<00:01, 1101.92it/s][A
     78%|████████████████████████████▋        | 3912/5047 [00:03<00:01, 1115.84it/s][A
     80%|█████████████████████████████▌       | 4028/5047 [00:03<00:00, 1127.34it/s][A
     82%|██████████████████████████████▍      | 4144/5047 [00:03<00:00, 1134.48it/s][A
     84%|███████████████████████████████▏     | 4259/5047 [00:03<00:00, 1136.86it/s][A
     87%|████████████████████████████████     | 4374/5047 [00:03<00:00, 1140.42it/s][A
     89%|████████████████████████████████▉    | 4489/5047 [00:03<00:00, 1141.82it/s][A
     91%|█████████████████████████████████▊   | 4604/5047 [00:04<00:00, 1144.08it/s][A
     94%|██████████████████████████████████▌  | 4719/5047 [00:04<00:00, 1142.77it/s][A
     96%|███████████████████████████████████▍ | 4834/5047 [00:04<00:00, 1141.34it/s][A
    100%|█████████████████████████████████████| 5047/5047 [00:04<00:00, 1125.26it/s][A
    Testing DataLoader 0:  26%|█████▋                | 8/31 [02:18<06:39,  0.06it/s]Sampling. The number of nodes to sample is 5235.
    Sampling on one graph took 10.618553876876831 seconds.
    
      0%|                                                  | 0/5235 [00:00<?, ?it/s][A
      2%|▊                                     | 109/5235 [00:00<00:04, 1082.54it/s][A
      4%|█▌                                    | 220/5235 [00:00<00:04, 1093.12it/s][A
      6%|██▍                                   | 330/5235 [00:00<00:04, 1095.95it/s][A
      8%|███▏                                  | 440/5235 [00:00<00:04, 1091.50it/s][A
     11%|███▉                                  | 550/5235 [00:00<00:04, 1093.02it/s][A
     13%|████▊                                 | 660/5235 [00:00<00:04, 1088.11it/s][A
     15%|█████▌                                | 770/5235 [00:00<00:04, 1090.44it/s][A
     17%|██████▍                               | 880/5235 [00:00<00:03, 1091.78it/s][A
     19%|███████▏                              | 990/5235 [00:00<00:03, 1093.73it/s][A
     21%|███████▊                             | 1100/5235 [00:01<00:03, 1048.36it/s][A
     23%|████████▌                            | 1210/5235 [00:01<00:03, 1061.61it/s][A
     25%|█████████▎                           | 1319/5235 [00:01<00:03, 1069.25it/s][A
     27%|██████████                           | 1428/5235 [00:01<00:03, 1074.29it/s][A
     29%|██████████▊                          | 1538/5235 [00:01<00:03, 1079.08it/s][A
     31%|███████████▋                         | 1647/5235 [00:01<00:03, 1082.22it/s][A
     34%|████████████▍                        | 1757/5235 [00:01<00:03, 1086.56it/s][A
     36%|█████████████▏                       | 1868/5235 [00:01<00:03, 1090.93it/s][A
     38%|█████████████▉                       | 1978/5235 [00:01<00:02, 1093.63it/s][A
     40%|██████████████▊                      | 2089/5235 [00:01<00:02, 1096.40it/s][A
     42%|███████████████▌                     | 2199/5235 [00:02<00:02, 1096.89it/s][A
     44%|████████████████▎                    | 2309/5235 [00:02<00:02, 1096.46it/s][A
     46%|█████████████████                    | 2419/5235 [00:02<00:02, 1055.11it/s][A
     48%|█████████████████▊                   | 2527/5235 [00:02<00:02, 1061.43it/s][A
     50%|██████████████████▋                  | 2637/5235 [00:02<00:02, 1072.66it/s][A
     52%|███████████████████▍                 | 2748/5235 [00:02<00:02, 1081.45it/s][A
     55%|████████████████████▏                | 2857/5235 [00:02<00:02, 1083.77it/s][A
     57%|████████████████████▉                | 2968/5235 [00:02<00:02, 1089.66it/s][A
     59%|█████████████████████▊               | 3079/5235 [00:02<00:01, 1092.90it/s][A
     61%|██████████████████████▌              | 3189/5235 [00:02<00:01, 1091.20it/s][A
     63%|███████████████████████▎             | 3299/5235 [00:03<00:01, 1092.51it/s][A
     65%|████████████████████████             | 3409/5235 [00:03<00:01, 1094.04it/s][A
     67%|████████████████████████▊            | 3519/5235 [00:03<00:01, 1089.72it/s][A
     69%|█████████████████████████▋           | 3628/5235 [00:03<00:01, 1083.42it/s][A
     71%|██████████████████████████▍          | 3737/5235 [00:03<00:01, 1084.15it/s][A
     73%|███████████████████████████▏         | 3846/5235 [00:03<00:01, 1044.95it/s][A
     76%|███████████████████████████▉         | 3956/5235 [00:03<00:01, 1059.66it/s][A
     78%|████████████████████████████▋        | 4066/5235 [00:03<00:01, 1070.74it/s][A
     80%|█████████████████████████████▌       | 4177/5235 [00:03<00:00, 1080.37it/s][A
     82%|██████████████████████████████▎      | 4287/5235 [00:03<00:00, 1085.12it/s][A
     84%|███████████████████████████████      | 4396/5235 [00:04<00:00, 1083.95it/s][A
     86%|███████████████████████████████▊     | 4506/5235 [00:04<00:00, 1087.32it/s][A
     88%|████████████████████████████████▋    | 4616/5235 [00:04<00:00, 1090.87it/s][A
     90%|█████████████████████████████████▍   | 4726/5235 [00:04<00:00, 1092.30it/s][A
     92%|██████████████████████████████████▏  | 4836/5235 [00:04<00:00, 1093.81it/s][A
     94%|██████████████████████████████████▉  | 4946/5235 [00:04<00:00, 1094.07it/s][A
     97%|███████████████████████████████████▋ | 5056/5235 [00:04<00:00, 1091.65it/s][A
    100%|█████████████████████████████████████| 5235/5235 [00:04<00:00, 1081.14it/s][A
    Testing DataLoader 0:  29%|██████▍               | 9/31 [02:41<06:33,  0.06it/s]Sampling. The number of nodes to sample is 3870.
    Sampling on one graph took 8.823965072631836 seconds.
    
      0%|                                                  | 0/3870 [00:00<?, ?it/s][A
      4%|█▍                                    | 142/3870 [00:00<00:02, 1405.99it/s][A
      7%|██▊                                   | 284/3870 [00:00<00:02, 1412.62it/s][A
     11%|████▏                                 | 426/3870 [00:00<00:02, 1334.36it/s][A
     15%|█████▌                                | 569/3870 [00:00<00:02, 1370.08it/s][A
     18%|██████▉                               | 712/3870 [00:00<00:02, 1389.73it/s][A
     22%|████████▍                             | 855/3870 [00:00<00:02, 1401.60it/s][A
     26%|█████████▊                            | 996/3870 [00:00<00:02, 1403.49it/s][A
     29%|██████████▉                          | 1138/3870 [00:00<00:01, 1408.06it/s][A
     33%|████████████▏                        | 1280/3870 [00:00<00:01, 1409.15it/s][A
     37%|█████████████▌                       | 1421/3870 [00:01<00:01, 1404.20it/s][A
     40%|██████████████▉                      | 1562/3870 [00:01<00:01, 1399.82it/s][A
     44%|████████████████▎                    | 1704/3870 [00:01<00:01, 1403.83it/s][A
     48%|█████████████████▋                   | 1846/3870 [00:01<00:01, 1408.09it/s][A
     51%|███████████████████                  | 1989/3870 [00:01<00:01, 1412.37it/s][A
     55%|████████████████████▎                | 2131/3870 [00:01<00:01, 1358.44it/s][A
     59%|█████████████████████▋               | 2273/3870 [00:01<00:01, 1374.10it/s][A
     62%|███████████████████████              | 2416/3870 [00:01<00:01, 1388.23it/s][A
     66%|████████████████████████▍            | 2556/3870 [00:01<00:00, 1388.33it/s][A
     70%|█████████████████████████▊           | 2697/3870 [00:01<00:00, 1394.03it/s][A
     73%|███████████████████████████          | 2837/3870 [00:02<00:00, 1395.71it/s][A
     77%|████████████████████████████▍        | 2979/3870 [00:02<00:00, 1400.11it/s][A
     81%|█████████████████████████████▊       | 3121/3870 [00:02<00:00, 1404.24it/s][A
     84%|███████████████████████████████▏     | 3262/3870 [00:02<00:00, 1403.07it/s][A
     88%|████████████████████████████████▌    | 3403/3870 [00:02<00:00, 1402.81it/s][A
     92%|█████████████████████████████████▉   | 3545/3870 [00:02<00:00, 1405.71it/s][A
     95%|███████████████████████████████████▏ | 3686/3870 [00:02<00:00, 1402.90it/s][A
    100%|█████████████████████████████████████| 3870/3870 [00:02<00:00, 1396.95it/s][A
    Testing DataLoader 0:  32%|██████▊              | 10/31 [02:56<06:10,  0.06it/s]Sampling. The number of nodes to sample is 4317.
    Sampling on one graph took 9.653472661972046 seconds.
    
      0%|                                                  | 0/4317 [00:00<?, ?it/s][A
      3%|█▏                                    | 129/4317 [00:00<00:03, 1288.47it/s][A
      6%|██▎                                   | 259/4317 [00:00<00:03, 1293.70it/s][A
      9%|███▍                                  | 389/4317 [00:00<00:03, 1295.66it/s][A
     12%|████▌                                 | 519/4317 [00:00<00:02, 1286.58it/s][A
     15%|█████▋                                | 649/4317 [00:00<00:02, 1289.18it/s][A
     18%|██████▊                               | 778/4317 [00:00<00:02, 1262.57it/s][A
     21%|███████▉                              | 905/4317 [00:00<00:02, 1242.90it/s][A
     24%|████████▊                            | 1035/4317 [00:00<00:02, 1259.73it/s][A
     27%|█████████▉                           | 1165/4317 [00:00<00:02, 1271.49it/s][A
     30%|███████████                          | 1295/4317 [00:01<00:02, 1279.05it/s][A
     33%|████████████▏                        | 1424/4317 [00:01<00:02, 1282.32it/s][A
     36%|█████████████▎                       | 1553/4317 [00:01<00:02, 1283.02it/s][A
     39%|██████████████▍                      | 1682/4317 [00:01<00:02, 1283.40it/s][A
     42%|███████████████▌                     | 1812/4317 [00:01<00:01, 1287.24it/s][A
     45%|████████████████▋                    | 1942/4317 [00:01<00:01, 1290.00it/s][A
     48%|█████████████████▊                   | 2072/4317 [00:01<00:01, 1291.74it/s][A
     51%|██████████████████▉                  | 2203/4317 [00:01<00:01, 1295.56it/s][A
     54%|███████████████████▉                 | 2333/4317 [00:01<00:01, 1296.64it/s][A
     57%|█████████████████████                | 2463/4317 [00:01<00:01, 1245.57it/s][A
     60%|██████████████████████▏              | 2592/4317 [00:02<00:01, 1257.90it/s][A
     63%|███████████████████████▎             | 2721/4317 [00:02<00:01, 1265.51it/s][A
     66%|████████████████████████▍            | 2851/4317 [00:02<00:01, 1273.18it/s][A
     69%|█████████████████████████▌           | 2982/4317 [00:02<00:01, 1281.55it/s][A
     72%|██████████████████████████▋          | 3112/4317 [00:02<00:00, 1286.03it/s][A
     75%|███████████████████████████▊         | 3242/4317 [00:02<00:00, 1289.44it/s][A
     78%|████████████████████████████▉        | 3372/4317 [00:02<00:00, 1291.66it/s][A
     81%|██████████████████████████████       | 3503/4317 [00:02<00:00, 1294.99it/s][A
     84%|███████████████████████████████▏     | 3633/4317 [00:02<00:00, 1296.27it/s][A
     87%|████████████████████████████████▎    | 3763/4317 [00:02<00:00, 1297.31it/s][A
     90%|█████████████████████████████████▎   | 3893/4317 [00:03<00:00, 1297.85it/s][A
     93%|██████████████████████████████████▍  | 4023/4317 [00:03<00:00, 1298.15it/s][A
     96%|███████████████████████████████████▌ | 4153/4317 [00:03<00:00, 1247.36it/s][A
    100%|█████████████████████████████████████| 4317/4317 [00:03<00:00, 1278.85it/s][A
    Testing DataLoader 0:  35%|███████▍             | 11/31 [03:14<05:53,  0.06it/s]Sampling. The number of nodes to sample is 4081.
    Sampling on one graph took 8.883440256118774 seconds.
    
      0%|                                                  | 0/4081 [00:00<?, ?it/s][A
      3%|█▎                                    | 136/4081 [00:00<00:02, 1352.20it/s][A
      7%|██▌                                   | 272/4081 [00:00<00:02, 1355.34it/s][A
     10%|███▊                                  | 408/4081 [00:00<00:02, 1352.89it/s][A
     13%|█████                                 | 545/4081 [00:00<00:02, 1356.11it/s][A
     17%|██████▎                               | 682/4081 [00:00<00:02, 1358.32it/s][A
     20%|███████▌                              | 818/4081 [00:00<00:02, 1356.90it/s][A
     23%|████████▉                             | 955/4081 [00:00<00:02, 1358.32it/s][A
     27%|█████████▉                           | 1092/4081 [00:00<00:02, 1361.08it/s][A
     30%|███████████▏                         | 1229/4081 [00:00<00:02, 1362.26it/s][A
     33%|████████████▍                        | 1366/4081 [00:01<00:02, 1261.58it/s][A
     37%|█████████████▌                       | 1498/4081 [00:01<00:02, 1278.37it/s][A
     40%|██████████████▊                      | 1635/4081 [00:01<00:01, 1303.80it/s][A
     43%|████████████████                     | 1772/4081 [00:01<00:01, 1322.95it/s][A
     47%|█████████████████▎                   | 1909/4081 [00:01<00:01, 1334.99it/s][A
     50%|██████████████████▌                  | 2046/4081 [00:01<00:01, 1344.40it/s][A
     53%|███████████████████▊                 | 2183/4081 [00:01<00:01, 1351.96it/s][A
     57%|█████████████████████                | 2319/4081 [00:01<00:01, 1350.88it/s][A
     60%|██████████████████████▎              | 2455/4081 [00:01<00:01, 1353.09it/s][A
     64%|███████████████████████▌             | 2592/4081 [00:01<00:01, 1357.82it/s][A
     67%|████████████████████████▋            | 2729/4081 [00:02<00:00, 1359.01it/s][A
     70%|█████████████████████████▉           | 2866/4081 [00:02<00:00, 1360.50it/s][A
     74%|███████████████████████████▏         | 3003/4081 [00:02<00:00, 1307.10it/s][A
     77%|████████████████████████████▍        | 3139/4081 [00:02<00:00, 1319.63it/s][A
     80%|█████████████████████████████▋       | 3275/4081 [00:02<00:00, 1328.79it/s][A
     84%|██████████████████████████████▉      | 3411/4081 [00:02<00:00, 1336.05it/s][A
     87%|████████████████████████████████▏    | 3548/4081 [00:02<00:00, 1344.28it/s][A
     90%|█████████████████████████████████▍   | 3684/4081 [00:02<00:00, 1346.77it/s][A
     94%|██████████████████████████████████▋  | 3820/4081 [00:02<00:00, 1349.60it/s][A
    100%|█████████████████████████████████████| 4081/4081 [00:03<00:00, 1340.80it/s][A
    Testing DataLoader 0:  39%|████████▏            | 12/31 [03:30<05:33,  0.06it/s]Sampling. The number of nodes to sample is 4052.
    Sampling on one graph took 8.878602266311646 seconds.
    
      0%|                                                  | 0/4052 [00:00<?, ?it/s][A
      3%|█▎                                    | 135/4052 [00:00<00:02, 1346.47it/s][A
      7%|██▌                                   | 272/4052 [00:00<00:02, 1356.86it/s][A
     10%|███▊                                  | 409/4052 [00:00<00:02, 1360.57it/s][A
     13%|█████                                 | 546/4052 [00:00<00:02, 1362.64it/s][A
     17%|██████▍                               | 683/4052 [00:00<00:02, 1358.41it/s][A
     20%|███████▋                              | 819/4052 [00:00<00:02, 1358.56it/s][A
     24%|████████▉                             | 955/4052 [00:00<00:02, 1299.74it/s][A
     27%|█████████▉                           | 1086/4052 [00:00<00:02, 1266.45it/s][A
     30%|███████████▏                         | 1223/4052 [00:00<00:02, 1295.94it/s][A
     34%|████████████▍                        | 1358/4052 [00:01<00:02, 1311.97it/s][A
     37%|█████████████▋                       | 1493/4052 [00:01<00:01, 1320.76it/s][A
     40%|██████████████▊                      | 1628/4052 [00:01<00:01, 1327.40it/s][A
     44%|████████████████                     | 1764/4052 [00:01<00:01, 1334.67it/s][A
     47%|█████████████████▎                   | 1900/4052 [00:01<00:01, 1340.64it/s][A
     50%|██████████████████▌                  | 2037/4052 [00:01<00:01, 1347.92it/s][A
     54%|███████████████████▊                 | 2172/4052 [00:01<00:01, 1342.55it/s][A
     57%|█████████████████████                | 2309/4052 [00:01<00:01, 1349.25it/s][A
     60%|██████████████████████▎              | 2446/4052 [00:01<00:01, 1352.99it/s][A
     64%|███████████████████████▌             | 2582/4052 [00:01<00:01, 1303.77it/s][A
     67%|████████████████████████▊            | 2717/4052 [00:02<00:01, 1315.21it/s][A
     70%|██████████████████████████           | 2852/4052 [00:02<00:00, 1322.61it/s][A
     74%|███████████████████████████▎         | 2988/4052 [00:02<00:00, 1332.88it/s][A
     77%|████████████████████████████▌        | 3124/4052 [00:02<00:00, 1340.78it/s][A
     80%|█████████████████████████████▊       | 3261/4052 [00:02<00:00, 1347.17it/s][A
     84%|███████████████████████████████      | 3398/4052 [00:02<00:00, 1352.28it/s][A
     87%|████████████████████████████████▎    | 3535/4052 [00:02<00:00, 1356.93it/s][A
     91%|█████████████████████████████████▌   | 3672/4052 [00:02<00:00, 1360.67it/s][A
     94%|██████████████████████████████████▊  | 3809/4052 [00:02<00:00, 1358.18it/s][A
    100%|█████████████████████████████████████| 4052/4052 [00:03<00:00, 1338.70it/s][A
    Testing DataLoader 0:  42%|████████▊            | 13/31 [03:46<05:14,  0.06it/s]Sampling. The number of nodes to sample is 3564.
    Sampling on one graph took 8.627844095230103 seconds.
    
      0%|                                                  | 0/3564 [00:00<?, ?it/s][A
      4%|█▌                                    | 152/3564 [00:00<00:02, 1511.03it/s][A
      9%|███▎                                  | 305/3564 [00:00<00:02, 1518.23it/s][A
     13%|████▊                                 | 457/3564 [00:00<00:02, 1514.57it/s][A
     17%|██████▍                               | 609/3564 [00:00<00:01, 1511.66it/s][A
     21%|████████                              | 761/3564 [00:00<00:01, 1512.19it/s][A
     26%|█████████▋                            | 913/3564 [00:00<00:01, 1445.62it/s][A
     30%|███████████                          | 1065/3564 [00:00<00:01, 1468.06it/s][A
     34%|████████████▌                        | 1216/3564 [00:00<00:01, 1480.83it/s][A
     38%|██████████████▏                      | 1369/3564 [00:00<00:01, 1493.01it/s][A
     43%|███████████████▊                     | 1521/3564 [00:01<00:01, 1500.63it/s][A
     47%|█████████████████▍                   | 1674/3564 [00:01<00:01, 1507.60it/s][A
     51%|██████████████████▉                  | 1827/3564 [00:01<00:01, 1511.36it/s][A
     56%|████████████████████▌                | 1979/3564 [00:01<00:01, 1513.52it/s][A
     60%|██████████████████████               | 2131/3564 [00:01<00:00, 1514.04it/s][A
     64%|███████████████████████▋             | 2283/3564 [00:01<00:00, 1512.21it/s][A
     68%|█████████████████████████▎           | 2435/3564 [00:01<00:00, 1513.14it/s][A
     73%|██████████████████████████▊          | 2587/3564 [00:01<00:00, 1513.92it/s][A
     77%|████████████████████████████▍        | 2740/3564 [00:01<00:00, 1516.37it/s][A
     81%|██████████████████████████████       | 2892/3564 [00:01<00:00, 1455.57it/s][A
     85%|███████████████████████████████▌     | 3045/3564 [00:02<00:00, 1475.47it/s][A
     90%|█████████████████████████████████▏   | 3197/3564 [00:02<00:00, 1487.39it/s][A
     94%|██████████████████████████████████▋  | 3347/3564 [00:02<00:00, 1490.77it/s][A
    100%|█████████████████████████████████████| 3564/3564 [00:02<00:00, 1495.68it/s][A
    Testing DataLoader 0:  45%|█████████▍           | 14/31 [04:01<04:53,  0.06it/s]Sampling. The number of nodes to sample is 4186.
    Sampling on one graph took 10.582089185714722 seconds.
    
      0%|                                                  | 0/4186 [00:00<?, ?it/s][A
      3%|█▏                                    | 131/4186 [00:00<00:03, 1303.64it/s][A
      6%|██▍                                   | 263/4186 [00:00<00:02, 1308.63it/s][A
      9%|███▌                                  | 394/4186 [00:00<00:02, 1276.38it/s][A
     13%|████▊                                 | 526/4186 [00:00<00:02, 1290.88it/s][A
     16%|█████▉                                | 658/4186 [00:00<00:02, 1300.02it/s][A
     19%|███████▏                              | 789/4186 [00:00<00:02, 1275.23it/s][A
     22%|████████▎                             | 917/4186 [00:00<00:02, 1192.62it/s][A
     25%|█████████▏                           | 1038/4186 [00:00<00:02, 1159.58it/s][A
     28%|██████████▏                          | 1155/4186 [00:00<00:02, 1144.00it/s][A
     30%|███████████▏                         | 1270/4186 [00:01<00:02, 1103.17it/s][A
     33%|████████████▏                        | 1381/4186 [00:01<00:02, 1021.12it/s][A
     35%|█████████████▍                        | 1485/4186 [00:01<00:02, 997.85it/s][A
     38%|██████████████                       | 1588/4186 [00:01<00:02, 1004.19it/s][A
     40%|██████████████▉                      | 1690/4186 [00:01<00:02, 1006.61it/s][A
     43%|███████████████▊                     | 1793/4186 [00:01<00:02, 1012.99it/s][A
     45%|█████████████████▏                    | 1895/4186 [00:01<00:02, 979.64it/s][A
     48%|██████████████████                    | 1994/4186 [00:01<00:02, 956.29it/s][A
     50%|██████████████████▉                   | 2090/4186 [00:01<00:02, 940.53it/s][A
     52%|███████████████████▊                  | 2185/4186 [00:02<00:02, 934.62it/s][A
     54%|████████████████████▋                 | 2280/4186 [00:02<00:02, 936.84it/s][A
     57%|█████████████████████▌                | 2376/4186 [00:02<00:01, 942.53it/s][A
     59%|██████████████████████▍               | 2471/4186 [00:02<00:01, 944.46it/s][A
     61%|███████████████████████▎              | 2566/4186 [00:02<00:01, 905.99it/s][A
     64%|████████████████████████▏             | 2661/4186 [00:02<00:01, 917.69it/s][A
     66%|█████████████████████████             | 2754/4186 [00:02<00:01, 910.98it/s][A
     69%|██████████████████████████            | 2871/4186 [00:02<00:01, 985.22it/s][A
     72%|██████████████████████████▌          | 3003/4186 [00:02<00:01, 1082.15it/s][A
     75%|███████████████████████████▋         | 3135/4186 [00:02<00:00, 1151.68it/s][A
     78%|████████████████████████████▉        | 3267/4186 [00:03<00:00, 1200.56it/s][A
     81%|██████████████████████████████       | 3399/4186 [00:03<00:00, 1234.76it/s][A
     84%|███████████████████████████████▏     | 3531/4186 [00:03<00:00, 1257.72it/s][A
     88%|████████████████████████████████▍    | 3663/4186 [00:03<00:00, 1274.16it/s][A
     91%|█████████████████████████████████▌   | 3793/4186 [00:03<00:00, 1281.39it/s][A
     94%|██████████████████████████████████▋  | 3925/4186 [00:03<00:00, 1291.18it/s][A
    100%|█████████████████████████████████████| 4186/4186 [00:03<00:00, 1108.77it/s][A
    Testing DataLoader 0:  48%|██████████▏          | 15/31 [04:20<04:37,  0.06it/s]Sampling. The number of nodes to sample is 2800.
    Sampling on one graph took 8.831885814666748 seconds.
    
      0%|                                                  | 0/2800 [00:00<?, ?it/s][A
      6%|██▍                                   | 179/2800 [00:00<00:01, 1783.89it/s][A
     13%|████▊                                 | 359/2800 [00:00<00:01, 1789.74it/s][A
     19%|███████▎                              | 539/2800 [00:00<00:01, 1794.03it/s][A
     26%|█████████▊                            | 720/2800 [00:00<00:01, 1799.09it/s][A
     32%|████████████▏                         | 900/2800 [00:00<00:01, 1782.83it/s][A
     39%|██████████████▎                      | 1082/2800 [00:00<00:00, 1792.56it/s][A
     45%|████████████████▋                    | 1264/2800 [00:00<00:00, 1799.91it/s][A
     52%|███████████████████                  | 1446/2800 [00:00<00:00, 1805.78it/s][A
     58%|█████████████████████▍               | 1627/2800 [00:00<00:00, 1805.30it/s][A
     65%|███████████████████████▉             | 1808/2800 [00:01<00:00, 1806.60it/s][A
     71%|██████████████████████████▎          | 1989/2800 [00:01<00:00, 1733.14it/s][A
     78%|████████████████████████████▋        | 2170/2800 [00:01<00:00, 1755.32it/s][A
     84%|███████████████████████████████      | 2352/2800 [00:01<00:00, 1773.14it/s][A
     90%|█████████████████████████████████▍   | 2534/2800 [00:01<00:00, 1786.72it/s][A
    100%|█████████████████████████████████████| 2800/2800 [00:01<00:00, 1787.45it/s][A
    Testing DataLoader 0:  52%|██████████▊          | 16/31 [04:33<04:16,  0.06it/s]Sampling. The number of nodes to sample is 4529.
    Sampling on one graph took 10.117744445800781 seconds.
    
      0%|                                                  | 0/4529 [00:00<?, ?it/s][A
      3%|█                                     | 125/4529 [00:00<00:03, 1243.33it/s][A
      6%|██                                    | 250/4529 [00:00<00:03, 1245.06it/s][A
      8%|███▏                                  | 375/4529 [00:00<00:03, 1245.14it/s][A
     11%|████▏                                 | 500/4529 [00:00<00:03, 1245.34it/s][A
     14%|█████▏                                | 625/4529 [00:00<00:03, 1246.49it/s][A
     17%|██████▎                               | 750/4529 [00:00<00:03, 1247.18it/s][A
     19%|███████▎                              | 876/4529 [00:00<00:02, 1248.52it/s][A
     22%|████████▏                            | 1002/4529 [00:00<00:02, 1249.32it/s][A
     25%|█████████▏                           | 1127/4529 [00:00<00:02, 1249.01it/s][A
     28%|██████████▏                          | 1253/4529 [00:01<00:02, 1249.41it/s][A
     30%|███████████▎                         | 1379/4529 [00:01<00:02, 1250.00it/s][A
     33%|████████████▎                        | 1505/4529 [00:01<00:02, 1250.78it/s][A
     36%|█████████████▎                       | 1631/4529 [00:01<00:02, 1251.11it/s][A
     39%|██████████████▎                      | 1757/4529 [00:01<00:02, 1251.16it/s][A
     42%|███████████████▍                     | 1883/4529 [00:01<00:02, 1251.94it/s][A
     44%|████████████████▍                    | 2009/4529 [00:01<00:02, 1251.04it/s][A
     47%|█████████████████▍                   | 2135/4529 [00:01<00:01, 1204.10it/s][A
     50%|██████████████████▍                  | 2260/4529 [00:01<00:01, 1216.55it/s][A
     53%|███████████████████▍                 | 2385/4529 [00:01<00:01, 1225.12it/s][A
     55%|████████████████████▌                | 2510/4529 [00:02<00:01, 1232.32it/s][A
     58%|█████████████████████▌               | 2635/4529 [00:02<00:01, 1237.45it/s][A
     61%|██████████████████████▌              | 2760/4529 [00:02<00:01, 1238.35it/s][A
     64%|███████████████████████▌             | 2885/4529 [00:02<00:01, 1241.55it/s][A
     66%|████████████████████████▌            | 3010/4529 [00:02<00:01, 1243.57it/s][A
     69%|█████████████████████████▌           | 3135/4529 [00:02<00:01, 1244.35it/s][A
     72%|██████████████████████████▋          | 3260/4529 [00:02<00:01, 1244.90it/s][A
     75%|███████████████████████████▋         | 3385/4529 [00:02<00:00, 1245.65it/s][A
     78%|████████████████████████████▋        | 3510/4529 [00:02<00:00, 1245.94it/s][A
     80%|█████████████████████████████▋       | 3635/4529 [00:02<00:00, 1156.34it/s][A
     83%|██████████████████████████████▋      | 3755/4529 [00:03<00:00, 1168.10it/s][A
     86%|███████████████████████████████▋     | 3880/4529 [00:03<00:00, 1191.33it/s][A
     88%|████████████████████████████████▋    | 4005/4529 [00:03<00:00, 1208.21it/s][A
     91%|█████████████████████████████████▋   | 4130/4529 [00:03<00:00, 1220.38it/s][A
     94%|██████████████████████████████████▊  | 4256/4529 [00:03<00:00, 1229.37it/s][A
     97%|███████████████████████████████████▊ | 4380/4529 [00:03<00:00, 1231.12it/s][A
    100%|█████████████████████████████████████| 4529/4529 [00:03<00:00, 1233.29it/s][A
    Testing DataLoader 0:  55%|███████████▌         | 17/31 [04:52<04:00,  0.06it/s]Sampling. The number of nodes to sample is 5021.
    Sampling on one graph took 10.339067459106445 seconds.
    
      0%|                                                  | 0/5021 [00:00<?, ?it/s][A
      2%|▊                                     | 114/5021 [00:00<00:04, 1130.77it/s][A
      5%|█▋                                    | 228/5021 [00:00<00:04, 1134.09it/s][A
      7%|██▌                                   | 342/5021 [00:00<00:04, 1135.79it/s][A
      9%|███▍                                  | 456/5021 [00:00<00:04, 1136.23it/s][A
     11%|████▎                                 | 571/5021 [00:00<00:03, 1137.81it/s][A
     14%|█████▏                                | 685/5021 [00:00<00:03, 1138.27it/s][A
     16%|██████                                | 800/5021 [00:00<00:03, 1138.97it/s][A
     18%|██████▉                               | 914/5021 [00:00<00:03, 1092.16it/s][A
     20%|███████▌                             | 1029/5021 [00:00<00:03, 1107.56it/s][A
     23%|████████▍                            | 1144/5021 [00:01<00:03, 1118.16it/s][A
     25%|█████████▎                           | 1259/5021 [00:01<00:03, 1124.92it/s][A
     27%|██████████                           | 1373/5021 [00:01<00:03, 1127.33it/s][A
     30%|██████████▉                          | 1488/5021 [00:01<00:03, 1131.48it/s][A
     32%|███████████▊                         | 1603/5021 [00:01<00:03, 1134.59it/s][A
     34%|████████████▋                        | 1718/5021 [00:01<00:02, 1137.33it/s][A
     36%|█████████████▌                       | 1832/5021 [00:01<00:02, 1134.16it/s][A
     39%|██████████████▎                      | 1947/5021 [00:01<00:02, 1136.58it/s][A
     41%|███████████████▏                     | 2062/5021 [00:01<00:02, 1137.88it/s][A
     43%|████████████████                     | 2177/5021 [00:01<00:02, 1139.47it/s][A
     46%|████████████████▉                    | 2292/5021 [00:02<00:02, 1140.12it/s][A
     48%|█████████████████▋                   | 2407/5021 [00:02<00:02, 1140.64it/s][A
     50%|██████████████████▌                  | 2522/5021 [00:02<00:02, 1138.35it/s][A
     52%|███████████████████▍                 | 2636/5021 [00:02<00:02, 1136.60it/s][A
     55%|████████████████████▎                | 2750/5021 [00:02<00:01, 1136.41it/s][A
     57%|█████████████████████                | 2864/5021 [00:02<00:01, 1135.58it/s][A
     59%|█████████████████████▉               | 2978/5021 [00:02<00:01, 1134.27it/s][A
     62%|██████████████████████▊              | 3092/5021 [00:02<00:01, 1132.69it/s][A
     64%|███████████████████████▋             | 3206/5021 [00:02<00:01, 1131.73it/s][A
     66%|████████████████████████▍            | 3320/5021 [00:02<00:01, 1129.07it/s][A
     68%|█████████████████████████▎           | 3434/5021 [00:03<00:01, 1129.97it/s][A
     71%|██████████████████████████▏          | 3548/5021 [00:03<00:01, 1131.01it/s][A
     73%|██████████████████████████▉          | 3662/5021 [00:03<00:01, 1123.60it/s][A
     75%|███████████████████████████▊         | 3776/5021 [00:03<00:01, 1127.65it/s][A
     77%|████████████████████████████▋        | 3889/5021 [00:03<00:01, 1083.94it/s][A
     80%|█████████████████████████████▍       | 4003/5021 [00:03<00:00, 1099.69it/s][A
     82%|██████████████████████████████▎      | 4118/5021 [00:03<00:00, 1111.82it/s][A
     84%|███████████████████████████████▏     | 4232/5021 [00:03<00:00, 1119.86it/s][A
     87%|████████████████████████████████     | 4346/5021 [00:03<00:00, 1125.49it/s][A
     89%|████████████████████████████████▊    | 4460/5021 [00:03<00:00, 1129.71it/s][A
     91%|█████████████████████████████████▋   | 4574/5021 [00:04<00:00, 1132.44it/s][A
     93%|██████████████████████████████████▌  | 4689/5021 [00:04<00:00, 1135.08it/s][A
     96%|███████████████████████████████████▍ | 4804/5021 [00:04<00:00, 1138.95it/s][A
    100%|█████████████████████████████████████| 5021/5021 [00:04<00:00, 1129.97it/s][A
    Testing DataLoader 0:  58%|████████████▏        | 18/31 [05:13<03:46,  0.06it/s]Sampling. The number of nodes to sample is 4991.
    Sampling on one graph took 10.325889348983765 seconds.
    
      0%|                                                  | 0/4991 [00:00<?, ?it/s][A
      2%|▊                                     | 113/4991 [00:00<00:04, 1125.44it/s][A
      5%|█▋                                    | 227/4991 [00:00<00:04, 1132.69it/s][A
      7%|██▌                                   | 341/4991 [00:00<00:04, 1134.64it/s][A
      9%|███▍                                  | 455/4991 [00:00<00:03, 1134.83it/s][A
     11%|████▎                                 | 569/4991 [00:00<00:03, 1136.46it/s][A
     14%|█████▏                                | 683/4991 [00:00<00:03, 1134.36it/s][A
     16%|██████                                | 797/4991 [00:00<00:03, 1134.92it/s][A
     18%|██████▉                               | 911/4991 [00:00<00:03, 1135.75it/s][A
     21%|███████▌                             | 1025/4991 [00:00<00:03, 1136.13it/s][A
     23%|████████▍                            | 1139/4991 [00:01<00:03, 1135.84it/s][A
     25%|█████████▎                           | 1253/4991 [00:01<00:03, 1136.27it/s][A
     27%|██████████▏                          | 1367/4991 [00:01<00:03, 1135.56it/s][A
     30%|██████████▉                          | 1481/4991 [00:01<00:03, 1135.99it/s][A
     32%|███████████▊                         | 1595/4991 [00:01<00:02, 1136.59it/s][A
     34%|████████████▋                        | 1709/4991 [00:01<00:02, 1137.20it/s][A
     37%|█████████████▌                       | 1823/4991 [00:01<00:02, 1136.84it/s][A
     39%|██████████████▎                      | 1937/4991 [00:01<00:02, 1136.90it/s][A
     41%|███████████████▏                     | 2052/4991 [00:01<00:02, 1138.12it/s][A
     43%|████████████████                     | 2167/4991 [00:01<00:02, 1139.90it/s][A
     46%|████████████████▉                    | 2281/4991 [00:02<00:02, 1085.60it/s][A
     48%|█████████████████▊                   | 2396/4991 [00:02<00:02, 1102.63it/s][A
     50%|██████████████████▌                  | 2510/4991 [00:02<00:02, 1112.75it/s][A
     53%|███████████████████▍                 | 2624/4991 [00:02<00:02, 1119.90it/s][A
     55%|████████████████████▎                | 2739/4991 [00:02<00:01, 1126.16it/s][A
     57%|█████████████████████▏               | 2854/4991 [00:02<00:01, 1130.67it/s][A
     59%|██████████████████████               | 2968/4991 [00:02<00:01, 1129.95it/s][A
     62%|██████████████████████▊              | 3082/4991 [00:02<00:01, 1132.93it/s][A
     64%|███████████████████████▋             | 3197/4991 [00:02<00:01, 1135.77it/s][A
     66%|████████████████████████▌            | 3312/4991 [00:02<00:01, 1137.68it/s][A
     69%|█████████████████████████▍           | 3426/4991 [00:03<00:01, 1138.34it/s][A
     71%|██████████████████████████▏          | 3540/4991 [00:03<00:01, 1138.64it/s][A
     73%|███████████████████████████          | 3655/4991 [00:03<00:01, 1139.31it/s][A
     76%|███████████████████████████▉         | 3770/4991 [00:03<00:01, 1139.61it/s][A
     78%|████████████████████████████▊        | 3884/4991 [00:03<00:00, 1139.27it/s][A
     80%|█████████████████████████████▋       | 3999/4991 [00:03<00:00, 1139.84it/s][A
     82%|██████████████████████████████▍      | 4114/4991 [00:03<00:00, 1141.37it/s][A
     85%|███████████████████████████████▎     | 4229/4991 [00:03<00:00, 1140.72it/s][A
     87%|████████████████████████████████▏    | 4344/4991 [00:03<00:00, 1140.52it/s][A
     89%|█████████████████████████████████    | 4459/4991 [00:03<00:00, 1141.32it/s][A
     92%|█████████████████████████████████▉   | 4574/4991 [00:04<00:00, 1141.68it/s][A
     94%|██████████████████████████████████▊  | 4689/4991 [00:04<00:00, 1142.33it/s][A
     96%|███████████████████████████████████▌ | 4804/4991 [00:04<00:00, 1144.21it/s][A
    100%|█████████████████████████████████████| 4991/4991 [00:04<00:00, 1134.74it/s][A
    Testing DataLoader 0:  61%|████████████▊        | 19/31 [05:34<03:31,  0.06it/s]Sampling. The number of nodes to sample is 4389.
    Sampling on one graph took 9.701767444610596 seconds.
    
      0%|                                                  | 0/4389 [00:00<?, ?it/s][A
      3%|█                                     | 127/4389 [00:00<00:03, 1269.24it/s][A
      6%|██▏                                   | 254/4389 [00:00<00:03, 1180.91it/s][A
      9%|███▎                                  | 382/4389 [00:00<00:03, 1223.67it/s][A
     12%|████▎                                 | 505/4389 [00:00<00:03, 1173.62it/s][A
     14%|█████▍                                | 633/4389 [00:00<00:03, 1208.80it/s][A
     17%|██████▌                               | 762/4389 [00:00<00:02, 1232.48it/s][A
     20%|███████▋                              | 890/4389 [00:00<00:02, 1246.08it/s][A
     23%|████████▌                            | 1019/4389 [00:00<00:02, 1256.95it/s][A
     26%|█████████▋                           | 1147/4389 [00:00<00:02, 1263.20it/s][A
     29%|██████████▋                          | 1275/4389 [00:01<00:02, 1267.84it/s][A
     32%|███████████▊                         | 1403/4389 [00:01<00:02, 1270.92it/s][A
     35%|████████████▉                        | 1531/4389 [00:01<00:02, 1272.41it/s][A
     38%|█████████████▉                       | 1659/4389 [00:01<00:02, 1274.60it/s][A
     41%|███████████████                      | 1787/4389 [00:01<00:02, 1275.79it/s][A
     44%|████████████████▏                    | 1915/4389 [00:01<00:01, 1276.38it/s][A
     47%|█████████████████▏                   | 2043/4389 [00:01<00:01, 1231.84it/s][A
     49%|██████████████████▎                  | 2169/4389 [00:01<00:01, 1239.95it/s][A
     52%|███████████████████▎                 | 2298/4389 [00:01<00:01, 1251.98it/s][A
     55%|████████████████████▍                | 2424/4389 [00:01<00:01, 1253.89it/s][A
     58%|█████████████████████▌               | 2552/4389 [00:02<00:01, 1260.71it/s][A
     61%|██████████████████████▌              | 2680/4389 [00:02<00:01, 1266.40it/s][A
     64%|███████████████████████▋             | 2808/4389 [00:02<00:01, 1269.37it/s][A
     67%|████████████████████████▊            | 2937/4389 [00:02<00:01, 1272.79it/s][A
     70%|█████████████████████████▊           | 3066/4389 [00:02<00:01, 1275.28it/s][A
     73%|██████████████████████████▉          | 3194/4389 [00:02<00:00, 1276.08it/s][A
     76%|████████████████████████████         | 3322/4389 [00:02<00:00, 1276.84it/s][A
     79%|█████████████████████████████        | 3451/4389 [00:02<00:00, 1278.18it/s][A
     82%|██████████████████████████████▏      | 3580/4389 [00:02<00:00, 1279.96it/s][A
     85%|███████████████████████████████▎     | 3709/4389 [00:02<00:00, 1224.91it/s][A
     87%|████████████████████████████████▎    | 3833/4389 [00:03<00:00, 1190.70it/s][A
     90%|█████████████████████████████████▍   | 3961/4389 [00:03<00:00, 1215.81it/s][A
     93%|██████████████████████████████████▍  | 4089/4389 [00:03<00:00, 1234.16it/s][A
     96%|███████████████████████████████████▌ | 4218/4389 [00:03<00:00, 1248.05it/s][A
    100%|█████████████████████████████████████| 4389/4389 [00:03<00:00, 1251.70it/s][A
    Testing DataLoader 0:  65%|█████████████▌       | 20/31 [05:52<03:13,  0.06it/s]Sampling. The number of nodes to sample is 3560.
    Sampling on one graph took 8.646739482879639 seconds.
    
      0%|                                                  | 0/3560 [00:00<?, ?it/s][A
      4%|█▌                                    | 149/3560 [00:00<00:02, 1489.10it/s][A
      8%|███▏                                  | 300/3560 [00:00<00:02, 1496.42it/s][A
     13%|████▊                                 | 450/3560 [00:00<00:02, 1497.21it/s][A
     17%|██████▍                               | 601/3560 [00:00<00:01, 1499.41it/s][A
     21%|████████                              | 753/3560 [00:00<00:01, 1503.32it/s][A
     25%|█████████▋                            | 904/3560 [00:00<00:01, 1437.85it/s][A
     29%|██████████▉                          | 1049/3560 [00:00<00:01, 1396.69it/s][A
     34%|████████████▍                        | 1200/3560 [00:00<00:01, 1429.75it/s][A
     38%|██████████████                       | 1351/3560 [00:00<00:01, 1452.58it/s][A
     42%|███████████████▌                     | 1502/3560 [00:01<00:01, 1469.05it/s][A
     46%|█████████████████▏                   | 1653/3560 [00:01<00:01, 1480.71it/s][A
     51%|██████████████████▋                  | 1803/3560 [00:01<00:01, 1485.18it/s][A
     55%|████████████████████▎                | 1955/3560 [00:01<00:01, 1493.59it/s][A
     59%|█████████████████████▉               | 2105/3560 [00:01<00:00, 1494.91it/s][A
     63%|███████████████████████▍             | 2256/3560 [00:01<00:00, 1497.65it/s][A
     68%|█████████████████████████            | 2407/3560 [00:01<00:00, 1500.98it/s][A
     72%|██████████████████████████▌          | 2558/3560 [00:01<00:00, 1502.83it/s][A
     76%|████████████████████████████▏        | 2709/3560 [00:01<00:00, 1503.47it/s][A
     80%|█████████████████████████████▋       | 2860/3560 [00:01<00:00, 1442.39it/s][A
     84%|███████████████████████████████▏     | 3005/3560 [00:02<00:00, 1402.22it/s][A
     89%|████████████████████████████████▊    | 3156/3560 [00:02<00:00, 1432.69it/s][A
     93%|██████████████████████████████████▎  | 3307/3560 [00:02<00:00, 1453.72it/s][A
    100%|█████████████████████████████████████| 3560/3560 [00:02<00:00, 1468.99it/s][A
    Testing DataLoader 0:  68%|██████████████▏      | 21/31 [06:06<02:54,  0.06it/s]Sampling. The number of nodes to sample is 1706.
    Sampling on one graph took 8.615693092346191 seconds.
    
      0%|                                                  | 0/1706 [00:00<?, ?it/s][A
     15%|█████▋                                | 255/1706 [00:00<00:00, 2540.27it/s][A
     30%|███████████▎                          | 510/1706 [00:00<00:00, 2544.39it/s][A
     45%|█████████████████                     | 765/1706 [00:00<00:00, 2529.91it/s][A
     60%|██████████████████████               | 1019/1706 [00:00<00:00, 2532.98it/s][A
     75%|███████████████████████████▋         | 1274/1706 [00:00<00:00, 2538.96it/s][A
    100%|█████████████████████████████████████| 1706/1706 [00:00<00:00, 2539.74it/s][A
    Testing DataLoader 0:  71%|██████████████▉      | 22/31 [06:16<02:34,  0.06it/s]Sampling. The number of nodes to sample is 3296.
    Sampling on one graph took 8.583760023117065 seconds.
    
      0%|                                                  | 0/3296 [00:00<?, ?it/s][A
      4%|█▌                                    | 140/3296 [00:00<00:02, 1393.72it/s][A
      9%|███▍                                  | 303/3296 [00:00<00:01, 1528.58it/s][A
     14%|█████▎                                | 466/3296 [00:00<00:01, 1573.54it/s][A
     19%|███████▎                              | 629/3296 [00:00<00:01, 1593.03it/s][A
     24%|█████████                             | 791/3296 [00:00<00:01, 1599.82it/s][A
     29%|██████████▉                           | 952/3296 [00:00<00:01, 1602.57it/s][A
     34%|████████████▌                        | 1114/3296 [00:00<00:01, 1606.58it/s][A
     39%|██████████████▎                      | 1277/3296 [00:00<00:01, 1611.73it/s][A
     44%|████████████████▏                    | 1440/3296 [00:00<00:01, 1615.22it/s][A
     49%|█████████████████▉                   | 1603/3296 [00:01<00:01, 1619.40it/s][A
     54%|███████████████████▊                 | 1766/3296 [00:01<00:00, 1620.56it/s][A
     59%|█████████████████████▋               | 1929/3296 [00:01<00:00, 1619.91it/s][A
     63%|███████████████████████▍             | 2091/3296 [00:01<00:00, 1550.75it/s][A
     68%|█████████████████████████▎           | 2254/3296 [00:01<00:00, 1572.97it/s][A
     73%|███████████████████████████          | 2412/3296 [00:01<00:00, 1512.83it/s][A
     78%|████████████████████████████▉        | 2575/3296 [00:01<00:00, 1544.56it/s][A
     83%|██████████████████████████████▋      | 2738/3296 [00:01<00:00, 1567.03it/s][A
     88%|████████████████████████████████▌    | 2901/3296 [00:01<00:00, 1583.22it/s][A
     93%|██████████████████████████████████▍  | 3064/3296 [00:01<00:00, 1594.97it/s][A
    100%|█████████████████████████████████████| 3296/3296 [00:02<00:00, 1585.44it/s][A
    Testing DataLoader 0:  74%|███████████████▌     | 23/31 [06:30<02:15,  0.06it/s]Sampling. The number of nodes to sample is 5155.
    Sampling on one graph took 10.520159006118774 seconds.
    
      0%|                                                  | 0/5155 [00:00<?, ?it/s][A
      2%|▊                                     | 110/5155 [00:00<00:04, 1095.67it/s][A
      4%|█▌                                    | 220/5155 [00:00<00:04, 1010.46it/s][A
      6%|██▍                                   | 331/5155 [00:00<00:04, 1053.72it/s][A
      9%|███▎                                  | 442/5155 [00:00<00:04, 1072.50it/s][A
     11%|████                                  | 553/5155 [00:00<00:04, 1084.24it/s][A
     13%|████▉                                 | 664/5155 [00:00<00:04, 1091.67it/s][A
     15%|█████▋                                | 775/5155 [00:00<00:03, 1096.28it/s][A
     17%|██████▌                               | 887/5155 [00:00<00:03, 1100.91it/s][A
     19%|███████▎                              | 998/5155 [00:00<00:03, 1102.48it/s][A
     22%|███████▉                             | 1109/5155 [00:01<00:03, 1104.32it/s][A
     24%|████████▊                            | 1220/5155 [00:01<00:03, 1104.66it/s][A
     26%|█████████▌                           | 1331/5155 [00:01<00:03, 1105.65it/s][A
     28%|██████████▎                          | 1442/5155 [00:01<00:03, 1106.60it/s][A
     30%|███████████▏                         | 1553/5155 [00:01<00:03, 1062.90it/s][A
     32%|███████████▉                         | 1663/5155 [00:01<00:03, 1073.54it/s][A
     34%|████████████▋                        | 1774/5155 [00:01<00:03, 1082.42it/s][A
     37%|█████████████▌                       | 1885/5155 [00:01<00:02, 1090.20it/s][A
     39%|██████████████▎                      | 1997/5155 [00:01<00:02, 1097.35it/s][A
     41%|███████████████▏                     | 2109/5155 [00:01<00:02, 1101.79it/s][A
     43%|███████████████▉                     | 2220/5155 [00:02<00:02, 1099.27it/s][A
     45%|████████████████▋                    | 2330/5155 [00:02<00:02, 1094.98it/s][A
     47%|█████████████████▌                   | 2441/5155 [00:02<00:02, 1098.37it/s][A
     50%|██████████████████▎                  | 2552/5155 [00:02<00:02, 1101.32it/s][A
     52%|███████████████████                  | 2664/5155 [00:02<00:02, 1104.40it/s][A
     54%|███████████████████▉                 | 2776/5155 [00:02<00:02, 1106.71it/s][A
     56%|████████████████████▋                | 2888/5155 [00:02<00:02, 1107.92it/s][A
     58%|█████████████████████▌               | 2999/5155 [00:02<00:02, 1061.91it/s][A
     60%|██████████████████████▎              | 3111/5155 [00:02<00:01, 1077.54it/s][A
     63%|███████████████████████▏             | 3223/5155 [00:02<00:01, 1087.43it/s][A
     65%|███████████████████████▉             | 3335/5155 [00:03<00:01, 1094.48it/s][A
     67%|████████████████████████▋            | 3447/5155 [00:03<00:01, 1100.12it/s][A
     69%|█████████████████████████▌           | 3559/5155 [00:03<00:01, 1103.87it/s][A
     71%|██████████████████████████▎          | 3670/5155 [00:03<00:01, 1105.35it/s][A
     73%|███████████████████████████▏         | 3782/5155 [00:03<00:01, 1106.86it/s][A
     76%|███████████████████████████▉         | 3894/5155 [00:03<00:01, 1108.73it/s][A
     78%|████████████████████████████▊        | 4006/5155 [00:03<00:01, 1109.53it/s][A
     80%|█████████████████████████████▌       | 4118/5155 [00:03<00:00, 1110.33it/s][A
     82%|██████████████████████████████▎      | 4230/5155 [00:03<00:00, 1111.19it/s][A
     84%|███████████████████████████████▏     | 4342/5155 [00:03<00:00, 1110.73it/s][A
     86%|███████████████████████████████▉     | 4454/5155 [00:04<00:00, 1068.66it/s][A
     89%|████████████████████████████████▊    | 4564/5155 [00:04<00:00, 1075.70it/s][A
     91%|█████████████████████████████████▌   | 4672/5155 [00:04<00:00, 1036.72it/s][A
     93%|██████████████████████████████████▎  | 4784/5155 [00:04<00:00, 1058.83it/s][A
     95%|███████████████████████████████████▏ | 4896/5155 [00:04<00:00, 1075.15it/s][A
     97%|███████████████████████████████████▉ | 5008/5155 [00:04<00:00, 1086.98it/s][A
    100%|█████████████████████████████████████| 5155/5155 [00:04<00:00, 1090.99it/s][A
    Testing DataLoader 0:  77%|████████████████▎    | 24/31 [06:52<02:00,  0.06it/s]Sampling. The number of nodes to sample is 3733.
    Sampling on one graph took 8.78532862663269 seconds.
    
      0%|                                                  | 0/3733 [00:00<?, ?it/s][A
      4%|█▍                                    | 147/3733 [00:00<00:02, 1461.82it/s][A
      8%|███                                   | 295/3733 [00:00<00:02, 1468.39it/s][A
     12%|████▌                                 | 443/3733 [00:00<00:02, 1471.65it/s][A
     16%|██████                                | 591/3733 [00:00<00:02, 1472.09it/s][A
     20%|███████▌                              | 739/3733 [00:00<00:02, 1471.73it/s][A
     24%|█████████                             | 887/3733 [00:00<00:01, 1473.02it/s][A
     28%|██████████▎                          | 1035/3733 [00:00<00:01, 1473.10it/s][A
     32%|███████████▋                         | 1183/3733 [00:00<00:01, 1473.20it/s][A
     36%|█████████████▏                       | 1331/3733 [00:00<00:01, 1473.23it/s][A
     40%|██████████████▋                      | 1479/3733 [00:01<00:01, 1472.02it/s][A
     44%|████████████████▏                    | 1627/3733 [00:01<00:01, 1471.36it/s][A
     48%|█████████████████▌                   | 1775/3733 [00:01<00:01, 1407.60it/s][A
     52%|███████████████████                  | 1923/3733 [00:01<00:01, 1426.74it/s][A
     55%|████████████████████▌                | 2070/3733 [00:01<00:01, 1437.95it/s][A
     59%|█████████████████████▉               | 2218/3733 [00:01<00:01, 1449.65it/s][A
     63%|███████████████████████▍             | 2364/3733 [00:01<00:00, 1451.70it/s][A
     67%|████████████████████████▉            | 2512/3733 [00:01<00:00, 1459.70it/s][A
     71%|██████████████████████████▎          | 2661/3733 [00:01<00:00, 1465.92it/s][A
     75%|███████████████████████████▊         | 2809/3733 [00:01<00:00, 1469.78it/s][A
     79%|█████████████████████████████▎       | 2957/3733 [00:02<00:00, 1472.04it/s][A
     83%|██████████████████████████████▊      | 3105/3733 [00:02<00:00, 1474.16it/s][A
     87%|████████████████████████████████▏    | 3253/3733 [00:02<00:00, 1475.15it/s][A
     91%|█████████████████████████████████▋   | 3401/3733 [00:02<00:00, 1476.32it/s][A
     95%|███████████████████████████████████▏ | 3549/3733 [00:02<00:00, 1476.24it/s][A
    100%|█████████████████████████████████████| 3733/3733 [00:02<00:00, 1447.86it/s][A
    Testing DataLoader 0:  81%|████████████████▉    | 25/31 [07:07<01:42,  0.06it/s]Sampling. The number of nodes to sample is 3780.
    Sampling on one graph took 8.823106527328491 seconds.
    
      0%|                                                  | 0/3780 [00:00<?, ?it/s][A
      4%|█▍                                    | 145/3780 [00:00<00:02, 1444.44it/s][A
      8%|██▉                                   | 291/3780 [00:00<00:02, 1449.29it/s][A
     12%|████▍                                 | 437/3780 [00:00<00:02, 1452.03it/s][A
     15%|█████▊                                | 583/3780 [00:00<00:02, 1453.28it/s][A
     19%|███████▎                              | 729/3780 [00:00<00:02, 1390.68it/s][A
     23%|████████▊                             | 871/3780 [00:00<00:02, 1399.51it/s][A
     27%|█████████▉                           | 1017/3780 [00:00<00:01, 1417.88it/s][A
     31%|███████████▍                         | 1163/3780 [00:00<00:01, 1430.10it/s][A
     35%|████████████▊                        | 1309/3780 [00:00<00:01, 1436.60it/s][A
     39%|██████████████▎                      | 1456/3780 [00:01<00:01, 1444.08it/s][A
     42%|███████████████▋                     | 1601/3780 [00:01<00:01, 1445.60it/s][A
     46%|█████████████████                    | 1747/3780 [00:01<00:01, 1449.49it/s][A
     50%|██████████████████▌                  | 1893/3780 [00:01<00:01, 1451.26it/s][A
     54%|███████████████████▉                 | 2039/3780 [00:01<00:01, 1452.64it/s][A
     58%|█████████████████████▍               | 2185/3780 [00:01<00:01, 1454.34it/s][A
     62%|██████████████████████▊              | 2331/3780 [00:01<00:00, 1454.41it/s][A
     66%|████████████████████████▏            | 2477/3780 [00:01<00:00, 1452.94it/s][A
     69%|█████████████████████████▋           | 2623/3780 [00:01<00:00, 1394.11it/s][A
     73%|███████████████████████████          | 2769/3780 [00:01<00:00, 1411.59it/s][A
     77%|████████████████████████████▌        | 2915/3780 [00:02<00:00, 1424.59it/s][A
     81%|█████████████████████████████▉       | 3061/3780 [00:02<00:00, 1433.16it/s][A
     85%|███████████████████████████████▍     | 3207/3780 [00:02<00:00, 1439.55it/s][A
     89%|████████████████████████████████▊    | 3352/3780 [00:02<00:00, 1441.01it/s][A
     93%|██████████████████████████████████▏  | 3497/3780 [00:02<00:00, 1443.62it/s][A
    100%|█████████████████████████████████████| 3780/3780 [00:02<00:00, 1437.31it/s][A
    Testing DataLoader 0:  84%|█████████████████▌   | 26/31 [07:22<01:25,  0.06it/s]Sampling. The number of nodes to sample is 2905.
    Sampling on one graph took 8.595379114151001 seconds.
    
      0%|                                                  | 0/2905 [00:00<?, ?it/s][A
      6%|██▎                                   | 176/2905 [00:00<00:01, 1750.21it/s][A
     12%|████▌                                 | 352/2905 [00:00<00:01, 1600.22it/s][A
     18%|██████▋                               | 513/2905 [00:00<00:01, 1568.98it/s][A
     24%|█████████                             | 690/2905 [00:00<00:01, 1643.61it/s][A
     30%|███████████▎                          | 867/2905 [00:00<00:01, 1686.89it/s][A
     36%|█████████████▎                       | 1044/2905 [00:00<00:01, 1714.19it/s][A
     42%|███████████████▌                     | 1222/2905 [00:00<00:00, 1732.75it/s][A
     48%|█████████████████▊                   | 1399/2905 [00:00<00:00, 1744.18it/s][A
     54%|████████████████████                 | 1577/2905 [00:00<00:00, 1752.63it/s][A
     60%|██████████████████████▎              | 1755/2905 [00:01<00:00, 1758.58it/s][A
     67%|████████████████████████▌            | 1933/2905 [00:01<00:00, 1762.02it/s][A
     73%|██████████████████████████▊          | 2110/2905 [00:01<00:00, 1763.77it/s][A
     79%|█████████████████████████████▏       | 2287/2905 [00:01<00:00, 1764.60it/s][A
     85%|███████████████████████████████▍     | 2464/2905 [00:01<00:00, 1766.14it/s][A
     91%|█████████████████████████████████▋   | 2641/2905 [00:01<00:00, 1689.86it/s][A
    100%|█████████████████████████████████████| 2905/2905 [00:01<00:00, 1700.86it/s][A
    Testing DataLoader 0:  87%|██████████████████▎  | 27/31 [07:35<01:07,  0.06it/s]Sampling. The number of nodes to sample is 3192.
    Sampling on one graph took 8.570141077041626 seconds.
    
      0%|                                                  | 0/3192 [00:00<?, ?it/s][A
      5%|█▉                                    | 165/3192 [00:00<00:01, 1647.73it/s][A
     10%|███▉                                  | 331/3192 [00:00<00:01, 1652.01it/s][A
     16%|█████▉                                | 498/3192 [00:00<00:01, 1655.80it/s][A
     21%|███████▉                              | 665/3192 [00:00<00:01, 1658.76it/s][A
     26%|█████████▉                            | 831/3192 [00:00<00:01, 1574.55it/s][A
     31%|███████████▊                          | 997/3192 [00:00<00:01, 1602.54it/s][A
     36%|█████████████▍                       | 1164/3192 [00:00<00:01, 1622.43it/s][A
     42%|███████████████▍                     | 1330/3192 [00:00<00:01, 1633.87it/s][A
     47%|█████████████████▎                   | 1496/3192 [00:00<00:01, 1639.32it/s][A
     52%|███████████████████▎                 | 1661/3192 [00:01<00:00, 1634.73it/s][A
     57%|█████████████████████▏               | 1826/3192 [00:01<00:00, 1638.44it/s][A
     62%|███████████████████████              | 1992/3192 [00:01<00:00, 1642.81it/s][A
     68%|█████████████████████████            | 2158/3192 [00:01<00:00, 1645.83it/s][A
     73%|██████████████████████████▉          | 2324/3192 [00:01<00:00, 1648.31it/s][A
     78%|████████████████████████████▊        | 2490/3192 [00:01<00:00, 1649.82it/s][A
     83%|██████████████████████████████▊      | 2656/3192 [00:01<00:00, 1649.86it/s][A
     88%|████████████████████████████████▋    | 2822/3192 [00:01<00:00, 1652.18it/s][A
     94%|██████████████████████████████████▋  | 2988/3192 [00:01<00:00, 1586.05it/s][A
    100%|█████████████████████████████████████| 3192/3192 [00:01<00:00, 1616.47it/s][A
    Testing DataLoader 0:  90%|██████████████████▉  | 28/31 [07:48<00:50,  0.06it/s]Sampling. The number of nodes to sample is 4057.
    Sampling on one graph took 8.896071672439575 seconds.
    
      0%|                                                  | 0/4057 [00:00<?, ?it/s][A
      3%|█▏                                    | 121/4057 [00:00<00:03, 1202.56it/s][A
      6%|██▍                                   | 259/4057 [00:00<00:02, 1303.07it/s][A
     10%|███▋                                  | 397/4057 [00:00<00:02, 1335.60it/s][A
     13%|█████                                 | 535/4057 [00:00<00:02, 1351.24it/s][A
     17%|██████▎                               | 673/4057 [00:00<00:02, 1358.92it/s][A
     20%|███████▌                              | 811/4057 [00:00<00:02, 1363.60it/s][A
     23%|████████▉                             | 949/4057 [00:00<00:02, 1368.42it/s][A
     27%|█████████▉                           | 1087/4057 [00:00<00:02, 1370.49it/s][A
     30%|███████████▏                         | 1225/4057 [00:00<00:02, 1371.16it/s][A
     34%|████████████▍                        | 1363/4057 [00:01<00:01, 1371.66it/s][A
     37%|█████████████▋                       | 1501/4057 [00:01<00:01, 1372.79it/s][A
     40%|██████████████▉                      | 1639/4057 [00:01<00:01, 1373.23it/s][A
     44%|████████████████▏                    | 1777/4057 [00:01<00:01, 1326.49it/s][A
     47%|█████████████████▍                   | 1911/4057 [00:01<00:01, 1329.67it/s][A
     51%|██████████████████▋                  | 2049/4057 [00:01<00:01, 1342.81it/s][A
     54%|███████████████████▉                 | 2187/4057 [00:01<00:01, 1353.21it/s][A
     57%|█████████████████████▏               | 2325/4057 [00:01<00:01, 1360.31it/s][A
     61%|██████████████████████▍              | 2463/4057 [00:01<00:01, 1365.22it/s][A
     64%|███████████████████████▋             | 2601/4057 [00:01<00:01, 1368.97it/s][A
     68%|████████████████████████▉            | 2739/4057 [00:02<00:00, 1371.67it/s][A
     71%|██████████████████████████▏          | 2877/4057 [00:02<00:00, 1372.53it/s][A
     74%|███████████████████████████▍         | 3015/4057 [00:02<00:00, 1374.70it/s][A
     78%|████████████████████████████▊        | 3153/4057 [00:02<00:00, 1375.09it/s][A
     81%|██████████████████████████████       | 3291/4057 [00:02<00:00, 1375.18it/s][A
     85%|███████████████████████████████▎     | 3429/4057 [00:02<00:00, 1374.46it/s][A
     88%|████████████████████████████████▌    | 3567/4057 [00:02<00:00, 1324.91it/s][A
     91%|█████████████████████████████████▊   | 3706/4057 [00:02<00:00, 1341.06it/s][A
     95%|███████████████████████████████████  | 3841/4057 [00:02<00:00, 1343.09it/s][A
    100%|█████████████████████████████████████| 4057/4057 [00:02<00:00, 1356.28it/s][A
    Testing DataLoader 0:  94%|███████████████████▋ | 29/31 [08:04<00:33,  0.06it/s]Sampling. The number of nodes to sample is 1160.
    Sampling on one graph took 8.504653930664062 seconds.
    
      0%|                                                  | 0/1160 [00:00<?, ?it/s][A
     27%|██████████                            | 308/1160 [00:00<00:00, 3074.46it/s][A
     53%|████████████████████▎                 | 619/1160 [00:00<00:00, 3090.82it/s][A
    100%|█████████████████████████████████████| 1160/1160 [00:00<00:00, 3091.26it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  97%|████████████████████▎| 30/31 [08:14<00:16,  0.06it/s]Sampling. The number of nodes to sample is 3316.
    Sampling on one graph took 8.574345827102661 seconds.
    
      0%|                                                  | 0/3316 [00:00<?, ?it/s][A
      5%|█▊                                    | 159/3316 [00:00<00:01, 1585.66it/s][A
     10%|███▋                                  | 320/3316 [00:00<00:01, 1594.38it/s][A
     15%|█████▌                                | 481/3316 [00:00<00:01, 1599.24it/s][A
     19%|███████▎                              | 641/3316 [00:00<00:01, 1599.33it/s][A
     24%|█████████▏                            | 801/3316 [00:00<00:01, 1598.89it/s][A
     29%|███████████                           | 961/3316 [00:00<00:01, 1571.54it/s][A
     34%|████████████▍                        | 1119/3316 [00:00<00:01, 1538.29it/s][A
     39%|██████████████▎                      | 1280/3316 [00:00<00:01, 1558.07it/s][A
     43%|████████████████                     | 1441/3316 [00:00<00:01, 1572.08it/s][A
     48%|█████████████████▊                   | 1601/3316 [00:01<00:01, 1580.48it/s][A
     53%|███████████████████▋                 | 1761/3316 [00:01<00:00, 1585.64it/s][A
     58%|█████████████████████▍               | 1922/3316 [00:01<00:00, 1590.36it/s][A
     63%|███████████████████████▏             | 2083/3316 [00:01<00:00, 1594.46it/s][A
     68%|█████████████████████████            | 2244/3316 [00:01<00:00, 1597.26it/s][A
     72%|██████████████████████████▊          | 2404/3316 [00:01<00:00, 1597.13it/s][A
     77%|████████████████████████████▌        | 2565/3316 [00:01<00:00, 1598.18it/s][A
     82%|██████████████████████████████▍      | 2726/3316 [00:01<00:00, 1600.08it/s][A
     87%|████████████████████████████████▏    | 2887/3316 [00:01<00:00, 1601.08it/s][A
     92%|██████████████████████████████████   | 3048/3316 [00:01<00:00, 1538.93it/s][A
    100%|█████████████████████████████████████| 3316/3316 [00:02<00:00, 1579.32it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0: 100%|█████████████████████| 31/31 [08:27<00:00,  0.06it/s]
    Testing checkpoint: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    Epoch index: 999
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    [rank: 0] Seed set to 0
    Restoring states from the checkpoint path at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from the checkpoint at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=999.ckpt
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:492: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
      warnings.warn(  # warn only once
    Testing DataLoader 0:   0%|                              | 0/31 [00:00<?, ?it/s]Sampling. The number of nodes to sample is 3851.
    Sampling on one graph took 8.865850448608398 seconds.
    
      0%|                                                  | 0/3851 [00:00<?, ?it/s][A
      4%|█▍                                    | 142/3851 [00:00<00:02, 1412.35it/s][A
      7%|██▊                                   | 284/3851 [00:00<00:02, 1415.82it/s][A
     11%|████▏                                 | 426/3851 [00:00<00:02, 1416.26it/s][A
     15%|█████▌                                | 568/3851 [00:00<00:02, 1415.25it/s][A
     18%|███████                               | 710/3851 [00:00<00:02, 1412.84it/s][A
     22%|████████▍                             | 852/3851 [00:00<00:02, 1413.98it/s][A
     26%|█████████▊                            | 994/3851 [00:00<00:02, 1413.11it/s][A
     30%|██████████▉                          | 1137/3851 [00:00<00:01, 1415.82it/s][A
     33%|████████████▎                        | 1279/3851 [00:00<00:01, 1416.43it/s][A
     37%|█████████████▋                       | 1421/3851 [00:01<00:01, 1416.65it/s][A
     41%|███████████████                      | 1563/3851 [00:01<00:01, 1417.33it/s][A
     44%|████████████████▍                    | 1705/3851 [00:01<00:01, 1417.18it/s][A
     48%|█████████████████▋                   | 1847/3851 [00:01<00:01, 1417.00it/s][A
     52%|███████████████████                  | 1989/3851 [00:01<00:01, 1417.72it/s][A
     55%|████████████████████▍                | 2131/3851 [00:01<00:01, 1416.51it/s][A
     59%|█████████████████████▊               | 2273/3851 [00:01<00:01, 1415.64it/s][A
     63%|███████████████████████▏             | 2415/3851 [00:01<00:01, 1414.00it/s][A
     66%|████████████████████████▌            | 2557/3851 [00:01<00:00, 1414.84it/s][A
     70%|█████████████████████████▉           | 2699/3851 [00:01<00:00, 1414.01it/s][A
     74%|███████████████████████████▎         | 2841/3851 [00:02<00:00, 1414.67it/s][A
     77%|████████████████████████████▋        | 2983/3851 [00:02<00:00, 1416.13it/s][A
     81%|██████████████████████████████       | 3125/3851 [00:02<00:00, 1367.50it/s][A
     85%|███████████████████████████████▎     | 3264/3851 [00:02<00:00, 1372.32it/s][A
     88%|████████████████████████████████▋    | 3406/3851 [00:02<00:00, 1386.31it/s][A
     92%|██████████████████████████████████   | 3549/3851 [00:02<00:00, 1396.70it/s][A
     96%|███████████████████████████████████▍ | 3691/3851 [00:02<00:00, 1402.67it/s][A
    100%|█████████████████████████████████████| 3851/3851 [00:02<00:00, 1408.65it/s][A
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:15<07:43,  0.06it/s]Sampling. The number of nodes to sample is 670.
    Sampling on one graph took 8.499013662338257 seconds.
    
      0%|                                                   | 0/670 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████| 670/670 [00:00<00:00, 3966.68it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:   6%|█▍                    | 2/31 [00:24<05:56,  0.08it/s]Sampling. The number of nodes to sample is 4360.
    Sampling on one graph took 9.594253778457642 seconds.
    
      0%|                                                  | 0/4360 [00:00<?, ?it/s][A
      3%|█                                     | 129/4360 [00:00<00:03, 1287.49it/s][A
      6%|██▏                                   | 258/4360 [00:00<00:03, 1287.06it/s][A
      9%|███▎                                  | 387/4360 [00:00<00:03, 1281.46it/s][A
     12%|████▍                                 | 516/4360 [00:00<00:02, 1283.56it/s][A
     15%|█████▌                                | 645/4360 [00:00<00:02, 1284.20it/s][A
     18%|██████▊                               | 775/4360 [00:00<00:02, 1286.63it/s][A
     21%|███████▉                              | 904/4360 [00:00<00:02, 1229.07it/s][A
     24%|████████▊                            | 1034/4360 [00:00<00:02, 1249.01it/s][A
     27%|█████████▉                           | 1164/4360 [00:00<00:02, 1262.12it/s][A
     30%|██████████▉                          | 1292/4360 [00:01<00:02, 1266.49it/s][A
     33%|████████████                         | 1422/4360 [00:01<00:02, 1275.33it/s][A
     36%|█████████████▏                       | 1551/4360 [00:01<00:02, 1278.34it/s][A
     39%|██████████████▎                      | 1680/4360 [00:01<00:02, 1281.40it/s][A
     41%|███████████████▎                     | 1809/4360 [00:01<00:01, 1283.11it/s][A
     44%|████████████████▍                    | 1938/4360 [00:01<00:01, 1284.91it/s][A
     47%|█████████████████▌                   | 2068/4360 [00:01<00:01, 1287.03it/s][A
     50%|██████████████████▋                  | 2198/4360 [00:01<00:01, 1288.84it/s][A
     53%|███████████████████▊                 | 2328/4360 [00:01<00:01, 1289.37it/s][A
     56%|████████████████████▊                | 2458/4360 [00:01<00:01, 1287.69it/s][A
     59%|█████████████████████▉               | 2587/4360 [00:02<00:01, 1242.91it/s][A
     62%|███████████████████████              | 2717/4360 [00:02<00:01, 1257.44it/s][A
     65%|████████████████████████▏            | 2846/4360 [00:02<00:01, 1266.89it/s][A
     68%|█████████████████████████▎           | 2976/4360 [00:02<00:01, 1273.82it/s][A
     71%|██████████████████████████▎          | 3106/4360 [00:02<00:00, 1279.83it/s][A
     74%|███████████████████████████▍         | 3235/4360 [00:02<00:00, 1282.57it/s][A
     77%|████████████████████████████▌        | 3365/4360 [00:02<00:00, 1285.97it/s][A
     80%|█████████████████████████████▋       | 3495/4360 [00:02<00:00, 1289.05it/s][A
     83%|██████████████████████████████▊      | 3625/4360 [00:02<00:00, 1290.56it/s][A
     86%|███████████████████████████████▊     | 3755/4360 [00:02<00:00, 1291.49it/s][A
     89%|████████████████████████████████▉    | 3885/4360 [00:03<00:00, 1291.06it/s][A
     92%|██████████████████████████████████   | 4015/4360 [00:03<00:00, 1291.55it/s][A
     95%|███████████████████████████████████▏ | 4145/4360 [00:03<00:00, 1239.41it/s][A
    100%|█████████████████████████████████████| 4360/4360 [00:03<00:00, 1273.80it/s][A
    Testing DataLoader 0:  10%|██▏                   | 3/31 [00:42<06:35,  0.07it/s]Sampling. The number of nodes to sample is 5180.
    Sampling on one graph took 10.509170532226562 seconds.
    
      0%|                                                  | 0/5180 [00:00<?, ?it/s][A
      2%|▊                                     | 111/5180 [00:00<00:04, 1104.58it/s][A
      4%|█▋                                    | 223/5180 [00:00<00:04, 1111.43it/s][A
      6%|██▍                                   | 335/5180 [00:00<00:04, 1111.97it/s][A
      9%|███▎                                  | 447/5180 [00:00<00:04, 1113.98it/s][A
     11%|████                                  | 559/5180 [00:00<00:04, 1114.45it/s][A
     13%|████▉                                 | 672/5180 [00:00<00:04, 1116.96it/s][A
     15%|█████▊                                | 784/5180 [00:00<00:03, 1117.23it/s][A
     17%|██████▌                               | 896/5180 [00:00<00:03, 1117.25it/s][A
     19%|███████▏                             | 1008/5180 [00:00<00:03, 1116.72it/s][A
     22%|████████                             | 1120/5180 [00:01<00:03, 1116.73it/s][A
     24%|████████▊                            | 1232/5180 [00:01<00:03, 1116.40it/s][A
     26%|█████████▌                           | 1344/5180 [00:01<00:03, 1115.72it/s][A
     28%|██████████▍                          | 1456/5180 [00:01<00:03, 1115.89it/s][A
     30%|███████████▏                         | 1568/5180 [00:01<00:03, 1115.82it/s][A
     32%|████████████                         | 1680/5180 [00:01<00:03, 1112.78it/s][A
     35%|████████████▊                        | 1792/5180 [00:01<00:03, 1112.27it/s][A
     37%|█████████████▌                       | 1904/5180 [00:01<00:02, 1111.53it/s][A
     39%|██████████████▍                      | 2016/5180 [00:01<00:02, 1111.69it/s][A
     41%|███████████████▏                     | 2128/5180 [00:01<00:02, 1112.97it/s][A
     43%|████████████████                     | 2240/5180 [00:02<00:02, 1113.76it/s][A
     45%|████████████████▊                    | 2352/5180 [00:02<00:02, 1115.59it/s][A
     48%|█████████████████▌                   | 2464/5180 [00:02<00:02, 1116.41it/s][A
     50%|██████████████████▍                  | 2576/5180 [00:02<00:02, 1109.59it/s][A
     52%|███████████████████▏                 | 2688/5180 [00:02<00:02, 1111.77it/s][A
     54%|████████████████████                 | 2800/5180 [00:02<00:02, 1113.80it/s][A
     56%|████████████████████▊                | 2912/5180 [00:02<00:02, 1114.34it/s][A
     58%|█████████████████████▌               | 3024/5180 [00:02<00:01, 1115.98it/s][A
     61%|██████████████████████▍              | 3136/5180 [00:02<00:01, 1117.12it/s][A
     63%|███████████████████████▏             | 3249/5180 [00:02<00:01, 1118.04it/s][A
     65%|████████████████████████             | 3362/5180 [00:03<00:01, 1119.84it/s][A
     67%|████████████████████████▊            | 3474/5180 [00:03<00:01, 1119.71it/s][A
     69%|█████████████████████████▌           | 3587/5180 [00:03<00:01, 1119.87it/s][A
     71%|██████████████████████████▍          | 3699/5180 [00:03<00:01, 1119.49it/s][A
     74%|███████████████████████████▏         | 3811/5180 [00:03<00:01, 1119.27it/s][A
     76%|████████████████████████████         | 3923/5180 [00:03<00:01, 1118.99it/s][A
     78%|████████████████████████████▊        | 4036/5180 [00:03<00:01, 1120.30it/s][A
     80%|█████████████████████████████▋       | 4149/5180 [00:03<00:00, 1119.25it/s][A
     82%|██████████████████████████████▍      | 4261/5180 [00:03<00:00, 1118.95it/s][A
     84%|███████████████████████████████▏     | 4373/5180 [00:03<00:00, 1112.38it/s][A
     87%|████████████████████████████████     | 4485/5180 [00:04<00:00, 1113.56it/s][A
     89%|████████████████████████████████▊    | 4597/5180 [00:04<00:00, 1113.93it/s][A
     91%|█████████████████████████████████▋   | 4709/5180 [00:04<00:00, 1114.55it/s][A
     93%|██████████████████████████████████▍  | 4821/5180 [00:04<00:00, 1071.26it/s][A
     95%|███████████████████████████████████▏ | 4933/5180 [00:04<00:00, 1084.91it/s][A
     97%|████████████████████████████████████ | 5045/5180 [00:04<00:00, 1093.60it/s][A
    100%|█████████████████████████████████████| 5180/5180 [00:04<00:00, 1112.25it/s][A
    Testing DataLoader 0:  13%|██▊                   | 4/31 [01:04<07:12,  0.06it/s]Sampling. The number of nodes to sample is 4946.
    Sampling on one graph took 10.291993856430054 seconds.
    
      0%|                                                  | 0/4946 [00:00<?, ?it/s][A
      2%|▊                                     | 113/4946 [00:00<00:04, 1122.59it/s][A
      5%|█▊                                    | 228/4946 [00:00<00:04, 1137.61it/s][A
      7%|██▋                                   | 343/4946 [00:00<00:04, 1142.20it/s][A
      9%|███▌                                  | 458/4946 [00:00<00:03, 1144.30it/s][A
     12%|████▍                                 | 573/4946 [00:00<00:03, 1144.85it/s][A
     14%|█████▎                                | 688/4946 [00:00<00:03, 1144.91it/s][A
     16%|██████▏                               | 803/4946 [00:00<00:03, 1145.59it/s][A
     19%|███████                               | 918/4946 [00:00<00:03, 1145.79it/s][A
     21%|███████▋                             | 1033/4946 [00:00<00:03, 1145.17it/s][A
     23%|████████▌                            | 1148/4946 [00:01<00:03, 1143.98it/s][A
     26%|█████████▍                           | 1263/4946 [00:01<00:03, 1145.11it/s][A
     28%|██████████▎                          | 1379/4946 [00:01<00:03, 1146.87it/s][A
     30%|███████████▏                         | 1495/4946 [00:01<00:03, 1148.27it/s][A
     33%|████████████                         | 1610/4946 [00:01<00:02, 1147.56it/s][A
     35%|████████████▉                        | 1726/4946 [00:01<00:02, 1148.65it/s][A
     37%|█████████████▊                       | 1841/4946 [00:01<00:02, 1147.63it/s][A
     40%|██████████████▋                      | 1957/4946 [00:01<00:02, 1149.55it/s][A
     42%|███████████████▌                     | 2072/4946 [00:01<00:02, 1149.61it/s][A
     44%|████████████████▎                    | 2187/4946 [00:01<00:02, 1149.13it/s][A
     47%|█████████████████▏                   | 2302/4946 [00:02<00:02, 1149.15it/s][A
     49%|██████████████████                   | 2417/4946 [00:02<00:02, 1148.55it/s][A
     51%|██████████████████▉                  | 2532/4946 [00:02<00:02, 1148.63it/s][A
     54%|███████████████████▊                 | 2648/4946 [00:02<00:01, 1149.82it/s][A
     56%|████████████████████▋                | 2764/4946 [00:02<00:01, 1150.12it/s][A
     58%|█████████████████████▌               | 2880/4946 [00:02<00:01, 1150.20it/s][A
     61%|██████████████████████▍              | 2996/4946 [00:02<00:01, 1151.16it/s][A
     63%|███████████████████████▎             | 3112/4946 [00:02<00:01, 1150.99it/s][A
     65%|████████████████████████▏            | 3228/4946 [00:02<00:01, 1151.31it/s][A
     68%|█████████████████████████            | 3344/4946 [00:02<00:01, 1151.90it/s][A
     70%|█████████████████████████▉           | 3460/4946 [00:03<00:01, 1151.44it/s][A
     72%|██████████████████████████▊          | 3576/4946 [00:03<00:01, 1150.97it/s][A
     75%|███████████████████████████▌         | 3692/4946 [00:03<00:01, 1150.52it/s][A
     77%|████████████████████████████▍        | 3808/4946 [00:03<00:00, 1151.58it/s][A
     79%|█████████████████████████████▎       | 3924/4946 [00:03<00:00, 1151.85it/s][A
     82%|██████████████████████████████▏      | 4040/4946 [00:03<00:00, 1150.89it/s][A
     84%|███████████████████████████████      | 4156/4946 [00:03<00:00, 1150.55it/s][A
     86%|███████████████████████████████▉     | 4272/4946 [00:03<00:00, 1151.55it/s][A
     89%|████████████████████████████████▊    | 4388/4946 [00:03<00:00, 1150.29it/s][A
     91%|█████████████████████████████████▋   | 4504/4946 [00:03<00:00, 1105.55it/s][A
     93%|██████████████████████████████████▌  | 4615/4946 [00:04<00:00, 1076.45it/s][A
     96%|███████████████████████████████████▍ | 4730/4946 [00:04<00:00, 1096.04it/s][A
    100%|█████████████████████████████████████| 4946/4946 [00:04<00:00, 1140.99it/s][A
    Testing DataLoader 0:  16%|███▌                  | 5/31 [01:24<07:20,  0.06it/s]Sampling. The number of nodes to sample is 2037.
    Sampling on one graph took 8.491875648498535 seconds.
    
      0%|                                                  | 0/2037 [00:00<?, ?it/s][A
     11%|████▏                                 | 224/2037 [00:00<00:00, 2236.02it/s][A
     22%|████████▍                             | 449/2037 [00:00<00:00, 2240.41it/s][A
     33%|████████████▌                         | 674/2037 [00:00<00:00, 2242.58it/s][A
     44%|████████████████▊                     | 900/2037 [00:00<00:00, 2245.72it/s][A
     55%|████████████████████▍                | 1125/2037 [00:00<00:00, 2243.53it/s][A
     66%|████████████████████████▌            | 1350/2037 [00:00<00:00, 2242.35it/s][A
     77%|████████████████████████████▌        | 1575/2037 [00:00<00:00, 2242.75it/s][A
     88%|████████████████████████████████▋    | 1800/2037 [00:00<00:00, 2242.91it/s][A
    100%|█████████████████████████████████████| 2037/2037 [00:00<00:00, 2242.69it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  19%|████▎                 | 6/31 [01:35<06:37,  0.06it/s]Sampling. The number of nodes to sample is 5050.
    Sampling on one graph took 10.378899812698364 seconds.
    
      0%|                                                  | 0/5050 [00:00<?, ?it/s][A
      2%|▊                                     | 114/5050 [00:00<00:04, 1134.64it/s][A
      5%|█▋                                    | 228/5050 [00:00<00:04, 1135.96it/s][A
      7%|██▌                                   | 343/5050 [00:00<00:04, 1138.00it/s][A
      9%|███▍                                  | 457/5050 [00:00<00:04, 1137.84it/s][A
     11%|████▎                                 | 571/5050 [00:00<00:03, 1136.55it/s][A
     14%|█████▏                                | 686/5050 [00:00<00:03, 1138.31it/s][A
     16%|██████                                | 801/5050 [00:00<00:03, 1139.33it/s][A
     18%|██████▉                               | 915/5050 [00:00<00:03, 1138.42it/s][A
     20%|███████▌                             | 1029/5050 [00:00<00:03, 1136.32it/s][A
     23%|████████▎                            | 1143/5050 [00:01<00:03, 1137.33it/s][A
     25%|█████████▏                           | 1257/5050 [00:01<00:03, 1137.46it/s][A
     27%|██████████                           | 1371/5050 [00:01<00:03, 1137.22it/s][A
     29%|██████████▉                          | 1485/5050 [00:01<00:03, 1137.72it/s][A
     32%|███████████▋                         | 1599/5050 [00:01<00:03, 1086.55it/s][A
     34%|████████████▌                        | 1713/5050 [00:01<00:03, 1101.22it/s][A
     36%|█████████████▍                       | 1827/5050 [00:01<00:02, 1111.61it/s][A
     38%|██████████████▏                      | 1941/5050 [00:01<00:02, 1119.77it/s][A
     41%|███████████████                      | 2055/5050 [00:01<00:02, 1124.28it/s][A
     43%|███████████████▉                     | 2169/5050 [00:01<00:02, 1128.30it/s][A
     45%|████████████████▋                    | 2283/5050 [00:02<00:02, 1130.93it/s][A
     47%|█████████████████▌                   | 2397/5050 [00:02<00:02, 1132.95it/s][A
     50%|██████████████████▍                  | 2511/5050 [00:02<00:02, 1134.15it/s][A
     52%|███████████████████▏                 | 2625/5050 [00:02<00:02, 1134.98it/s][A
     54%|████████████████████                 | 2739/5050 [00:02<00:02, 1136.04it/s][A
     56%|████████████████████▉                | 2853/5050 [00:02<00:01, 1131.21it/s][A
     59%|█████████████████████▋               | 2967/5050 [00:02<00:01, 1132.94it/s][A
     61%|██████████████████████▌              | 3081/5050 [00:02<00:01, 1134.08it/s][A
     63%|███████████████████████▍             | 3195/5050 [00:02<00:01, 1135.06it/s][A
     66%|████████████████████████▎            | 3310/5050 [00:02<00:01, 1136.63it/s][A
     68%|█████████████████████████            | 3424/5050 [00:03<00:01, 1136.71it/s][A
     70%|█████████████████████████▉           | 3538/5050 [00:03<00:01, 1136.55it/s][A
     72%|██████████████████████████▊          | 3652/5050 [00:03<00:01, 1137.03it/s][A
     75%|███████████████████████████▌         | 3766/5050 [00:03<00:01, 1137.70it/s][A
     77%|████████████████████████████▍        | 3880/5050 [00:03<00:01, 1138.24it/s][A
     79%|█████████████████████████████▎       | 3994/5050 [00:03<00:00, 1138.04it/s][A
     81%|██████████████████████████████       | 4108/5050 [00:03<00:00, 1133.66it/s][A
     84%|██████████████████████████████▉      | 4222/5050 [00:03<00:00, 1131.62it/s][A
     86%|███████████████████████████████▊     | 4336/5050 [00:03<00:00, 1131.43it/s][A
     88%|████████████████████████████████▌    | 4450/5050 [00:03<00:00, 1131.57it/s][A
     90%|█████████████████████████████████▍   | 4564/5050 [00:04<00:00, 1086.73it/s][A
     93%|██████████████████████████████████▎  | 4678/5050 [00:04<00:00, 1101.61it/s][A
     95%|███████████████████████████████████  | 4792/5050 [00:04<00:00, 1112.04it/s][A
     97%|███████████████████████████████████▉ | 4906/5050 [00:04<00:00, 1119.92it/s][A
    100%|█████████████████████████████████████| 5050/5050 [00:04<00:00, 1128.78it/s][A
    Testing DataLoader 0:  23%|████▉                 | 7/31 [01:56<06:39,  0.06it/s]Sampling. The number of nodes to sample is 5047.
    Sampling on one graph took 10.414512395858765 seconds.
    
      0%|                                                  | 0/5047 [00:00<?, ?it/s][A
      2%|▊                                       | 96/5047 [00:00<00:05, 954.20it/s][A
      4%|█▌                                    | 210/5047 [00:00<00:04, 1058.35it/s][A
      6%|██▍                                   | 323/5047 [00:00<00:04, 1087.65it/s][A
      9%|███▎                                  | 436/5047 [00:00<00:04, 1103.38it/s][A
     11%|████▏                                 | 550/5047 [00:00<00:04, 1113.50it/s][A
     13%|████▉                                 | 663/5047 [00:00<00:03, 1117.08it/s][A
     15%|█████▊                                | 776/5047 [00:00<00:03, 1120.63it/s][A
     18%|██████▋                               | 890/5047 [00:00<00:03, 1124.26it/s][A
     20%|███████▎                             | 1003/5047 [00:00<00:03, 1125.61it/s][A
     22%|████████▏                            | 1117/5047 [00:01<00:03, 1127.46it/s][A
     24%|█████████                            | 1230/5047 [00:01<00:03, 1128.20it/s][A
     27%|█████████▊                           | 1343/5047 [00:01<00:03, 1128.49it/s][A
     29%|██████████▋                          | 1457/5047 [00:01<00:03, 1129.14it/s][A
     31%|███████████▌                         | 1570/5047 [00:01<00:03, 1084.36it/s][A
     33%|████████████▎                        | 1679/5047 [00:01<00:03, 1051.04it/s][A
     35%|█████████████▏                       | 1791/5047 [00:01<00:03, 1070.41it/s][A
     38%|█████████████▉                       | 1904/5047 [00:01<00:02, 1087.53it/s][A
     40%|██████████████▊                      | 2018/5047 [00:01<00:02, 1100.18it/s][A
     42%|███████████████▌                     | 2131/5047 [00:01<00:02, 1108.02it/s][A
     44%|████████████████▍                    | 2245/5047 [00:02<00:02, 1115.24it/s][A
     47%|█████████████████▎                   | 2359/5047 [00:02<00:02, 1120.81it/s][A
     49%|██████████████████                   | 2472/5047 [00:02<00:02, 1122.80it/s][A
     51%|██████████████████▉                  | 2585/5047 [00:02<00:02, 1124.77it/s][A
     53%|███████████████████▊                 | 2698/5047 [00:02<00:02, 1125.95it/s][A
     56%|████████████████████▌                | 2811/5047 [00:02<00:01, 1126.92it/s][A
     58%|█████████████████████▍               | 2924/5047 [00:02<00:01, 1126.99it/s][A
     60%|██████████████████████▎              | 3037/5047 [00:02<00:01, 1126.78it/s][A
     62%|███████████████████████              | 3150/5047 [00:02<00:01, 1081.81it/s][A
     65%|███████████████████████▉             | 3263/5047 [00:02<00:01, 1095.14it/s][A
     67%|████████████████████████▋            | 3376/5047 [00:03<00:01, 1105.15it/s][A
     69%|█████████████████████████▌           | 3490/5047 [00:03<00:01, 1113.45it/s][A
     71%|██████████████████████████▍          | 3604/5047 [00:03<00:01, 1118.70it/s][A
     74%|███████████████████████████▏         | 3717/5047 [00:03<00:01, 1121.77it/s][A
     76%|████████████████████████████         | 3830/5047 [00:03<00:01, 1123.91it/s][A
     78%|████████████████████████████▉        | 3943/5047 [00:03<00:00, 1125.47it/s][A
     80%|█████████████████████████████▋       | 4056/5047 [00:03<00:00, 1126.01it/s][A
     83%|██████████████████████████████▌      | 4169/5047 [00:03<00:00, 1125.95it/s][A
     85%|███████████████████████████████▍     | 4283/5047 [00:03<00:00, 1127.30it/s][A
     87%|████████████████████████████████▏    | 4396/5047 [00:03<00:00, 1126.98it/s][A
     89%|█████████████████████████████████    | 4509/5047 [00:04<00:00, 1080.28it/s][A
     92%|█████████████████████████████████▉   | 4622/5047 [00:04<00:00, 1092.80it/s][A
     94%|██████████████████████████████████▋  | 4736/5047 [00:04<00:00, 1104.02it/s][A
     96%|███████████████████████████████████▌ | 4847/5047 [00:04<00:00, 1105.43it/s][A
    100%|█████████████████████████████████████| 5047/5047 [00:04<00:00, 1110.14it/s][A
    Testing DataLoader 0:  26%|█████▋                | 8/31 [02:17<06:36,  0.06it/s]Sampling. The number of nodes to sample is 5235.
    Sampling on one graph took 10.60219955444336 seconds.
    
      0%|                                                  | 0/5235 [00:00<?, ?it/s][A
      2%|▊                                     | 110/5235 [00:00<00:04, 1094.60it/s][A
      4%|█▌                                    | 221/5235 [00:00<00:04, 1098.10it/s][A
      6%|██▍                                   | 332/5235 [00:00<00:04, 1099.11it/s][A
      8%|███▏                                  | 442/5235 [00:00<00:04, 1099.43it/s][A
     11%|████                                  | 552/5235 [00:00<00:04, 1099.36it/s][A
     13%|████▊                                 | 662/5235 [00:00<00:04, 1093.03it/s][A
     15%|█████▌                                | 772/5235 [00:00<00:04, 1093.60it/s][A
     17%|██████▍                               | 883/5235 [00:00<00:03, 1096.64it/s][A
     19%|███████▏                              | 994/5235 [00:00<00:03, 1099.45it/s][A
     21%|███████▊                             | 1105/5235 [00:01<00:03, 1102.40it/s][A
     23%|████████▌                            | 1216/5235 [00:01<00:03, 1104.53it/s][A
     25%|█████████▍                           | 1327/5235 [00:01<00:03, 1103.86it/s][A
     27%|██████████▏                          | 1438/5235 [00:01<00:03, 1105.27it/s][A
     30%|██████████▉                          | 1549/5235 [00:01<00:03, 1105.59it/s][A
     32%|███████████▋                         | 1661/5235 [00:01<00:03, 1107.69it/s][A
     34%|████████████▌                        | 1772/5235 [00:01<00:03, 1108.13it/s][A
     36%|█████████████▎                       | 1883/5235 [00:01<00:03, 1108.23it/s][A
     38%|██████████████                       | 1995/5235 [00:01<00:02, 1108.79it/s][A
     40%|██████████████▉                      | 2106/5235 [00:01<00:02, 1107.48it/s][A
     42%|███████████████▋                     | 2217/5235 [00:02<00:02, 1106.37it/s][A
     44%|████████████████▍                    | 2328/5235 [00:02<00:02, 1106.42it/s][A
     47%|█████████████████▏                   | 2439/5235 [00:02<00:02, 1107.33it/s][A
     49%|██████████████████                   | 2550/5235 [00:02<00:02, 1061.45it/s][A
     51%|██████████████████▊                  | 2662/5235 [00:02<00:02, 1076.49it/s][A
     53%|███████████████████▌                 | 2774/5235 [00:02<00:02, 1087.10it/s][A
     55%|████████████████████▍                | 2886/5235 [00:02<00:02, 1094.12it/s][A
     57%|█████████████████████▏               | 2998/5235 [00:02<00:02, 1099.78it/s][A
     59%|█████████████████████▉               | 3110/5235 [00:02<00:01, 1103.72it/s][A
     62%|██████████████████████▊              | 3222/5235 [00:02<00:01, 1105.85it/s][A
     64%|███████████████████████▌             | 3333/5235 [00:03<00:01, 1105.63it/s][A
     66%|████████████████████████▎            | 3445/5235 [00:03<00:01, 1107.43it/s][A
     68%|█████████████████████████▏           | 3557/5235 [00:03<00:01, 1108.25it/s][A
     70%|█████████████████████████▉           | 3669/5235 [00:03<00:01, 1108.84it/s][A
     72%|██████████████████████████▋          | 3780/5235 [00:03<00:01, 1108.19it/s][A
     74%|███████████████████████████▌         | 3891/5235 [00:03<00:01, 1108.37it/s][A
     76%|████████████████████████████▎        | 4002/5235 [00:03<00:01, 1060.17it/s][A
     79%|█████████████████████████████        | 4113/5235 [00:03<00:01, 1073.29it/s][A
     81%|█████████████████████████████▊       | 4224/5235 [00:03<00:00, 1082.44it/s][A
     83%|██████████████████████████████▋      | 4335/5235 [00:03<00:00, 1089.67it/s][A
     85%|███████████████████████████████▍     | 4446/5235 [00:04<00:00, 1094.39it/s][A
     87%|████████████████████████████████▏    | 4557/5235 [00:04<00:00, 1098.48it/s][A
     89%|████████████████████████████████▉    | 4668/5235 [00:04<00:00, 1100.98it/s][A
     91%|█████████████████████████████████▊   | 4779/5235 [00:04<00:00, 1103.56it/s][A
     93%|██████████████████████████████████▌  | 4890/5235 [00:04<00:00, 1105.34it/s][A
     96%|███████████████████████████████████▎ | 5001/5235 [00:04<00:00, 1105.50it/s][A
     98%|████████████████████████████████████▏| 5112/5235 [00:04<00:00, 1106.06it/s][A
    100%|█████████████████████████████████████| 5235/5235 [00:04<00:00, 1099.40it/s][A
    Testing DataLoader 0:  29%|██████▍               | 9/31 [02:39<06:30,  0.06it/s]Sampling. The number of nodes to sample is 3870.
    Sampling on one graph took 8.816938877105713 seconds.
    
      0%|                                                  | 0/3870 [00:00<?, ?it/s][A
      4%|█▍                                    | 143/3870 [00:00<00:02, 1421.18it/s][A
      7%|██▊                                   | 286/3870 [00:00<00:02, 1423.39it/s][A
     11%|████▏                                 | 429/3870 [00:00<00:02, 1424.36it/s][A
     15%|█████▌                                | 572/3870 [00:00<00:02, 1425.91it/s][A
     18%|███████                               | 715/3870 [00:00<00:02, 1426.52it/s][A
     22%|████████▍                             | 858/3870 [00:00<00:02, 1369.96it/s][A
     26%|█████████▊                            | 998/3870 [00:00<00:02, 1379.15it/s][A
     29%|██████████▉                          | 1141/3870 [00:00<00:01, 1394.56it/s][A
     33%|████████████▎                        | 1284/3870 [00:00<00:01, 1404.60it/s][A
     37%|█████████████▋                       | 1427/3870 [00:01<00:01, 1411.72it/s][A
     41%|███████████████                      | 1569/3870 [00:01<00:01, 1412.01it/s][A
     44%|████████████████▎                    | 1711/3870 [00:01<00:01, 1411.19it/s][A
     48%|█████████████████▋                   | 1853/3870 [00:01<00:01, 1413.71it/s][A
     52%|███████████████████                  | 1995/3870 [00:01<00:01, 1414.72it/s][A
     55%|████████████████████▍                | 2138/3870 [00:01<00:01, 1416.81it/s][A
     59%|█████████████████████▊               | 2281/3870 [00:01<00:01, 1420.54it/s][A
     63%|███████████████████████▏             | 2424/3870 [00:01<00:01, 1423.36it/s][A
     66%|████████████████████████▌            | 2567/3870 [00:01<00:00, 1423.17it/s][A
     70%|█████████████████████████▉           | 2710/3870 [00:01<00:00, 1355.85it/s][A
     74%|███████████████████████████▎         | 2851/3870 [00:02<00:00, 1369.36it/s][A
     77%|████████████████████████████▌        | 2993/3870 [00:02<00:00, 1383.86it/s][A
     81%|█████████████████████████████▉       | 3136/3870 [00:02<00:00, 1395.11it/s][A
     85%|███████████████████████████████▎     | 3279/3870 [00:02<00:00, 1402.83it/s][A
     88%|████████████████████████████████▋    | 3420/3870 [00:02<00:00, 1400.41it/s][A
     92%|██████████████████████████████████   | 3563/3870 [00:02<00:00, 1407.97it/s][A
     96%|███████████████████████████████████▍ | 3706/3870 [00:02<00:00, 1412.98it/s][A
    100%|█████████████████████████████████████| 3870/3870 [00:02<00:00, 1405.58it/s][A
    Testing DataLoader 0:  32%|██████▊              | 10/31 [02:55<06:07,  0.06it/s]Sampling. The number of nodes to sample is 4317.
    Sampling on one graph took 9.697660684585571 seconds.
    
      0%|                                                  | 0/4317 [00:00<?, ?it/s][A
      3%|█▏                                    | 131/4317 [00:00<00:03, 1304.11it/s][A
      6%|██▎                                   | 262/4317 [00:00<00:03, 1305.70it/s][A
      9%|███▍                                  | 394/4317 [00:00<00:02, 1309.16it/s][A
     12%|████▋                                 | 526/4317 [00:00<00:02, 1310.48it/s][A
     15%|█████▊                                | 658/4317 [00:00<00:02, 1310.96it/s][A
     18%|██████▉                               | 790/4317 [00:00<00:02, 1312.73it/s][A
     21%|████████                              | 922/4317 [00:00<00:02, 1313.81it/s][A
     24%|█████████                            | 1054/4317 [00:00<00:02, 1314.71it/s][A
     27%|██████████▏                          | 1186/4317 [00:00<00:02, 1314.65it/s][A
     31%|███████████▎                         | 1318/4317 [00:01<00:02, 1314.07it/s][A
     34%|████████████▍                        | 1450/4317 [00:01<00:02, 1295.04it/s][A
     37%|█████████████▌                       | 1580/4317 [00:01<00:02, 1262.01it/s][A
     40%|██████████████▋                      | 1712/4317 [00:01<00:02, 1277.17it/s][A
     43%|███████████████▊                     | 1844/4317 [00:01<00:01, 1288.26it/s][A
     46%|████████████████▉                    | 1976/4317 [00:01<00:01, 1295.57it/s][A
     49%|██████████████████                   | 2108/4317 [00:01<00:01, 1301.23it/s][A
     52%|███████████████████▏                 | 2239/4317 [00:01<00:01, 1301.99it/s][A
     55%|████████████████████▎                | 2370/4317 [00:01<00:01, 1298.15it/s][A
     58%|█████████████████████▍               | 2502/4317 [00:01<00:01, 1302.65it/s][A
     61%|██████████████████████▌              | 2634/4317 [00:02<00:01, 1305.23it/s][A
     64%|███████████████████████▋             | 2766/4317 [00:02<00:01, 1308.34it/s][A
     67%|████████████████████████▊            | 2898/4317 [00:02<00:01, 1310.23it/s][A
     70%|█████████████████████████▉           | 3030/4317 [00:02<00:00, 1309.95it/s][A
     73%|███████████████████████████          | 3162/4317 [00:02<00:00, 1257.06it/s][A
     76%|████████████████████████████▏        | 3293/4317 [00:02<00:00, 1271.72it/s][A
     79%|█████████████████████████████▎       | 3425/4317 [00:02<00:00, 1283.55it/s][A
     82%|██████████████████████████████▍      | 3557/4317 [00:02<00:00, 1292.71it/s][A
     85%|███████████████████████████████▌     | 3689/4317 [00:02<00:00, 1298.55it/s][A
     89%|████████████████████████████████▋    | 3821/4317 [00:02<00:00, 1303.43it/s][A
     92%|█████████████████████████████████▉   | 3953/4317 [00:03<00:00, 1306.97it/s][A
     95%|███████████████████████████████████  | 4085/4317 [00:03<00:00, 1308.97it/s][A
    100%|█████████████████████████████████████| 4317/4317 [00:03<00:00, 1300.19it/s][A
    Testing DataLoader 0:  35%|███████▍             | 11/31 [03:12<05:50,  0.06it/s]Sampling. The number of nodes to sample is 4081.
    Sampling on one graph took 8.980005741119385 seconds.
    
      0%|                                                  | 0/4081 [00:00<?, ?it/s][A
      3%|█▎                                    | 137/4081 [00:00<00:02, 1365.21it/s][A
      7%|██▌                                   | 274/4081 [00:00<00:02, 1366.62it/s][A
     10%|███▊                                  | 412/4081 [00:00<00:02, 1369.83it/s][A
     13%|█████                                 | 550/4081 [00:00<00:02, 1371.06it/s][A
     17%|██████▍                               | 688/4081 [00:00<00:02, 1372.15it/s][A
     20%|███████▋                              | 826/4081 [00:00<00:02, 1374.00it/s][A
     24%|████████▉                             | 964/4081 [00:00<00:02, 1373.90it/s][A
     27%|█████████▉                           | 1102/4081 [00:00<00:02, 1374.70it/s][A
     30%|███████████▏                         | 1240/4081 [00:00<00:02, 1372.46it/s][A
     34%|████████████▍                        | 1378/4081 [00:01<00:01, 1373.82it/s][A
     37%|█████████████▋                       | 1516/4081 [00:01<00:01, 1374.92it/s][A
     41%|██████████████▉                      | 1654/4081 [00:01<00:01, 1320.66it/s][A
     44%|████████████████▏                    | 1792/4081 [00:01<00:01, 1337.27it/s][A
     47%|█████████████████▍                   | 1930/4081 [00:01<00:01, 1348.84it/s][A
     51%|██████████████████▋                  | 2068/4081 [00:01<00:01, 1357.26it/s][A
     54%|████████████████████                 | 2206/4081 [00:01<00:01, 1363.23it/s][A
     57%|█████████████████████▎               | 2344/4081 [00:01<00:01, 1366.73it/s][A
     61%|██████████████████████▌              | 2482/4081 [00:01<00:01, 1369.20it/s][A
     64%|███████████████████████▊             | 2620/4081 [00:01<00:01, 1372.22it/s][A
     68%|█████████████████████████            | 2758/4081 [00:02<00:00, 1373.59it/s][A
     71%|██████████████████████████▎          | 2896/4081 [00:02<00:00, 1373.32it/s][A
     74%|███████████████████████████▌         | 3034/4081 [00:02<00:00, 1372.89it/s][A
     78%|████████████████████████████▊        | 3172/4081 [00:02<00:00, 1373.09it/s][A
     81%|██████████████████████████████       | 3310/4081 [00:02<00:00, 1373.30it/s][A
     84%|███████████████████████████████▎     | 3448/4081 [00:02<00:00, 1321.22it/s][A
     88%|████████████████████████████████▌    | 3587/4081 [00:02<00:00, 1338.88it/s][A
     91%|█████████████████████████████████▊   | 3725/4081 [00:02<00:00, 1349.92it/s][A
     95%|███████████████████████████████████  | 3863/4081 [00:02<00:00, 1357.49it/s][A
    100%|█████████████████████████████████████| 4081/4081 [00:02<00:00, 1362.30it/s][A
    Testing DataLoader 0:  39%|████████▏            | 12/31 [03:29<05:31,  0.06it/s]Sampling. The number of nodes to sample is 4052.
    Sampling on one graph took 8.891419410705566 seconds.
    
      0%|                                                  | 0/4052 [00:00<?, ?it/s][A
      3%|█▎                                    | 138/4052 [00:00<00:02, 1373.83it/s][A
      7%|██▌                                   | 277/4052 [00:00<00:02, 1379.30it/s][A
     10%|███▉                                  | 416/4052 [00:00<00:02, 1381.05it/s][A
     14%|█████▏                                | 555/4052 [00:00<00:02, 1380.26it/s][A
     17%|██████▌                               | 694/4052 [00:00<00:02, 1381.60it/s][A
     21%|███████▊                              | 833/4052 [00:00<00:02, 1381.30it/s][A
     24%|█████████                             | 972/4052 [00:00<00:02, 1382.31it/s][A
     27%|██████████▏                          | 1111/4052 [00:00<00:02, 1321.53it/s][A
     31%|███████████▍                         | 1247/4052 [00:00<00:02, 1332.58it/s][A
     34%|████████████▋                        | 1386/4052 [00:01<00:01, 1347.63it/s][A
     38%|█████████████▉                       | 1525/4052 [00:01<00:01, 1358.70it/s][A
     41%|███████████████▏                     | 1664/4052 [00:01<00:01, 1366.50it/s][A
     44%|████████████████▍                    | 1803/4052 [00:01<00:01, 1372.16it/s][A
     48%|█████████████████▋                   | 1941/4052 [00:01<00:01, 1374.40it/s][A
     51%|██████████████████▉                  | 2080/4052 [00:01<00:01, 1378.33it/s][A
     55%|████████████████████▎                | 2219/4052 [00:01<00:01, 1379.83it/s][A
     58%|█████████████████████▌               | 2358/4052 [00:01<00:01, 1380.93it/s][A
     62%|██████████████████████▊              | 2497/4052 [00:01<00:01, 1381.90it/s][A
     65%|████████████████████████             | 2636/4052 [00:01<00:01, 1381.27it/s][A
     68%|█████████████████████████▎           | 2775/4052 [00:02<00:00, 1378.77it/s][A
     72%|██████████████████████████▌          | 2913/4052 [00:02<00:00, 1323.32it/s][A
     75%|███████████████████████████▊         | 3051/4052 [00:02<00:00, 1338.88it/s][A
     79%|█████████████████████████████        | 3188/4052 [00:02<00:00, 1346.39it/s][A
     82%|██████████████████████████████▎      | 3326/4052 [00:02<00:00, 1353.85it/s][A
     85%|███████████████████████████████▋     | 3464/4052 [00:02<00:00, 1359.23it/s][A
     89%|████████████████████████████████▉    | 3602/4052 [00:02<00:00, 1364.59it/s][A
     92%|██████████████████████████████████▏  | 3740/4052 [00:02<00:00, 1367.19it/s][A
     96%|███████████████████████████████████▍ | 3878/4052 [00:02<00:00, 1370.33it/s][A
    100%|█████████████████████████████████████| 4052/4052 [00:02<00:00, 1365.54it/s][A
    Testing DataLoader 0:  42%|████████▊            | 13/31 [03:45<05:11,  0.06it/s]Sampling. The number of nodes to sample is 3564.
    Sampling on one graph took 8.6067955493927 seconds.
    
      0%|                                                  | 0/3564 [00:00<?, ?it/s][A
      4%|█▌                                    | 151/3564 [00:00<00:02, 1506.77it/s][A
      9%|███▎                                  | 305/3564 [00:00<00:02, 1524.85it/s][A
     13%|████▉                                 | 459/3564 [00:00<00:02, 1529.50it/s][A
     17%|██████▌                               | 613/3564 [00:00<00:01, 1531.60it/s][A
     22%|████████▏                             | 767/3564 [00:00<00:01, 1529.92it/s][A
     26%|█████████▊                            | 921/3564 [00:00<00:01, 1530.51it/s][A
     30%|███████████▏                         | 1075/3564 [00:00<00:01, 1530.73it/s][A
     34%|████████████▊                        | 1229/3564 [00:00<00:01, 1531.67it/s][A
     39%|██████████████▎                      | 1383/3564 [00:00<00:01, 1529.60it/s][A
     43%|███████████████▉                     | 1537/3564 [00:01<00:01, 1530.67it/s][A
     47%|█████████████████▌                   | 1691/3564 [00:01<00:01, 1530.98it/s][A
     52%|███████████████████▏                 | 1845/3564 [00:01<00:01, 1467.15it/s][A
     56%|████████████████████▋                | 1998/3564 [00:01<00:01, 1482.96it/s][A
     60%|██████████████████████▎              | 2152/3564 [00:01<00:00, 1498.21it/s][A
     65%|███████████████████████▉             | 2306/3564 [00:01<00:00, 1507.83it/s][A
     69%|█████████████████████████▌           | 2460/3564 [00:01<00:00, 1514.77it/s][A
     73%|███████████████████████████▏         | 2614/3564 [00:01<00:00, 1521.07it/s][A
     78%|████████████████████████████▋        | 2767/3564 [00:01<00:00, 1517.72it/s][A
     82%|██████████████████████████████▎      | 2920/3564 [00:01<00:00, 1518.97it/s][A
     86%|███████████████████████████████▉     | 3073/3564 [00:02<00:00, 1520.26it/s][A
     91%|█████████████████████████████████▍   | 3226/3564 [00:02<00:00, 1520.38it/s][A
     95%|███████████████████████████████████  | 3380/3564 [00:02<00:00, 1523.33it/s][A
    100%|█████████████████████████████████████| 3564/3564 [00:02<00:00, 1519.11it/s][A
    Testing DataLoader 0:  45%|█████████▍           | 14/31 [03:59<04:50,  0.06it/s]Sampling. The number of nodes to sample is 4186.
    Sampling on one graph took 9.027856588363647 seconds.
    
      0%|                                                  | 0/4186 [00:00<?, ?it/s][A
      3%|█▏                                    | 132/4186 [00:00<00:03, 1315.39it/s][A
      6%|██▍                                   | 265/4186 [00:00<00:02, 1321.83it/s][A
     10%|███▌                                  | 398/4186 [00:00<00:02, 1324.87it/s][A
     13%|████▊                                 | 531/4186 [00:00<00:02, 1322.91it/s][A
     16%|██████                                | 664/4186 [00:00<00:02, 1265.30it/s][A
     19%|███████▏                              | 797/4186 [00:00<00:02, 1283.78it/s][A
     22%|████████▍                             | 930/4186 [00:00<00:02, 1297.10it/s][A
     25%|█████████▍                           | 1062/4186 [00:00<00:02, 1304.11it/s][A
     29%|██████████▌                          | 1195/4186 [00:00<00:02, 1309.74it/s][A
     32%|███████████▋                         | 1328/4186 [00:01<00:02, 1314.86it/s][A
     35%|████████████▉                        | 1461/4186 [00:01<00:02, 1317.75it/s][A
     38%|██████████████                       | 1594/4186 [00:01<00:01, 1320.31it/s][A
     41%|███████████████▎                     | 1727/4186 [00:01<00:01, 1321.89it/s][A
     44%|████████████████▍                    | 1860/4186 [00:01<00:01, 1321.47it/s][A
     48%|█████████████████▌                   | 1993/4186 [00:01<00:01, 1322.83it/s][A
     51%|██████████████████▊                  | 2126/4186 [00:01<00:01, 1322.68it/s][A
     54%|███████████████████▉                 | 2259/4186 [00:01<00:01, 1318.77it/s][A
     57%|█████████████████████▏               | 2391/4186 [00:01<00:01, 1262.05it/s][A
     60%|██████████████████████▎              | 2524/4186 [00:01<00:01, 1279.82it/s][A
     63%|███████████████████████▍             | 2657/4186 [00:02<00:01, 1292.35it/s][A
     67%|████████████████████████▋            | 2790/4186 [00:02<00:01, 1303.43it/s][A
     70%|█████████████████████████▊           | 2921/4186 [00:02<00:00, 1305.23it/s][A
     73%|███████████████████████████          | 3055/4186 [00:02<00:00, 1314.11it/s][A
     76%|████████████████████████████▏        | 3188/4186 [00:02<00:00, 1318.35it/s][A
     79%|█████████████████████████████▎       | 3322/4186 [00:02<00:00, 1322.52it/s][A
     83%|██████████████████████████████▌      | 3456/4186 [00:02<00:00, 1324.83it/s][A
     86%|███████████████████████████████▋     | 3590/4186 [00:02<00:00, 1327.35it/s][A
     89%|████████████████████████████████▉    | 3723/4186 [00:02<00:00, 1325.79it/s][A
     92%|██████████████████████████████████   | 3856/4186 [00:02<00:00, 1325.74it/s][A
     95%|███████████████████████████████████▎ | 3990/4186 [00:03<00:00, 1329.18it/s][A
    100%|█████████████████████████████████████| 4186/4186 [00:03<00:00, 1307.77it/s][A
    Testing DataLoader 0:  48%|██████████▏          | 15/31 [04:16<04:33,  0.06it/s]Sampling. The number of nodes to sample is 2800.
    Sampling on one graph took 8.5395987033844 seconds.
    
      0%|                                                  | 0/2800 [00:00<?, ?it/s][A
      6%|██▍                                   | 181/2800 [00:00<00:01, 1806.74it/s][A
     13%|████▉                                 | 363/2800 [00:00<00:01, 1813.56it/s][A
     19%|███████▍                              | 545/2800 [00:00<00:01, 1704.61it/s][A
     26%|█████████▊                            | 727/2800 [00:00<00:01, 1747.00it/s][A
     32%|████████████▎                         | 909/2800 [00:00<00:01, 1770.26it/s][A
     39%|██████████████▍                      | 1090/2800 [00:00<00:00, 1782.67it/s][A
     45%|████████████████▊                    | 1271/2800 [00:00<00:00, 1789.97it/s][A
     52%|███████████████████▏                 | 1452/2800 [00:00<00:00, 1794.79it/s][A
     58%|█████████████████████▌               | 1632/2800 [00:00<00:00, 1794.04it/s][A
     65%|███████████████████████▉             | 1813/2800 [00:01<00:00, 1798.59it/s][A
     71%|██████████████████████████▎          | 1994/2800 [00:01<00:00, 1801.84it/s][A
     78%|████████████████████████████▊        | 2176/2800 [00:01<00:00, 1804.70it/s][A
     84%|███████████████████████████████▏     | 2358/2800 [00:01<00:00, 1807.82it/s][A
     91%|█████████████████████████████████▌   | 2540/2800 [00:01<00:00, 1809.76it/s][A
    100%|█████████████████████████████████████| 2800/2800 [00:01<00:00, 1776.94it/s][A
    Testing DataLoader 0:  52%|██████████▊          | 16/31 [04:28<04:11,  0.06it/s]Sampling. The number of nodes to sample is 4529.
    Sampling on one graph took 9.824167966842651 seconds.
    
      0%|                                                  | 0/4529 [00:00<?, ?it/s][A
      3%|█                                     | 124/4529 [00:00<00:03, 1230.89it/s][A
      5%|██                                    | 248/4529 [00:00<00:03, 1235.58it/s][A
      8%|███                                   | 372/4529 [00:00<00:03, 1235.06it/s][A
     11%|████▏                                 | 496/4529 [00:00<00:03, 1235.62it/s][A
     14%|█████▏                                | 621/4529 [00:00<00:03, 1237.41it/s][A
     16%|██████▎                               | 745/4529 [00:00<00:03, 1236.04it/s][A
     19%|███████▎                              | 869/4529 [00:00<00:02, 1235.64it/s][A
     22%|████████▎                             | 993/4529 [00:00<00:02, 1187.72it/s][A
     25%|█████████                            | 1116/4529 [00:00<00:02, 1199.51it/s][A
     27%|██████████▏                          | 1240/4529 [00:01<00:02, 1211.34it/s][A
     30%|███████████▏                         | 1364/4529 [00:01<00:02, 1219.40it/s][A
     33%|████████████▏                        | 1488/4529 [00:01<00:02, 1224.64it/s][A
     36%|█████████████▏                       | 1612/4529 [00:01<00:02, 1227.57it/s][A
     38%|██████████████▏                      | 1736/4529 [00:01<00:02, 1229.91it/s][A
     41%|███████████████▏                     | 1860/4529 [00:01<00:02, 1230.96it/s][A
     44%|████████████████▏                    | 1984/4529 [00:01<00:02, 1232.94it/s][A
     47%|█████████████████▏                   | 2108/4529 [00:01<00:01, 1227.01it/s][A
     49%|██████████████████▏                  | 2232/4529 [00:01<00:01, 1228.51it/s][A
     52%|███████████████████▏                 | 2356/4529 [00:01<00:01, 1230.93it/s][A
     55%|████████████████████▎                | 2480/4529 [00:02<00:01, 1232.18it/s][A
     57%|█████████████████████▎               | 2604/4529 [00:02<00:01, 1179.28it/s][A
     60%|██████████████████████▏              | 2723/4529 [00:02<00:01, 1145.74it/s][A
     63%|███████████████████████▎             | 2848/4529 [00:02<00:01, 1173.69it/s][A
     66%|████████████████████████▎            | 2972/4529 [00:02<00:01, 1192.57it/s][A
     68%|█████████████████████████▎           | 3096/4529 [00:02<00:01, 1205.56it/s][A
     71%|██████████████████████████▎          | 3220/4529 [00:02<00:01, 1215.22it/s][A
     74%|███████████████████████████▎         | 3344/4529 [00:02<00:00, 1221.36it/s][A
     77%|████████████████████████████▎        | 3468/4529 [00:02<00:00, 1225.91it/s][A
     79%|█████████████████████████████▎       | 3592/4529 [00:02<00:00, 1229.82it/s][A
     82%|██████████████████████████████▎      | 3716/4529 [00:03<00:00, 1231.69it/s][A
     85%|███████████████████████████████▎     | 3840/4529 [00:03<00:00, 1229.67it/s][A
     88%|████████████████████████████████▍    | 3964/4529 [00:03<00:00, 1231.79it/s][A
     90%|█████████████████████████████████▍   | 4088/4529 [00:03<00:00, 1233.82it/s][A
     93%|██████████████████████████████████▍  | 4212/4529 [00:03<00:00, 1185.32it/s][A
     96%|███████████████████████████████████▍ | 4337/4529 [00:03<00:00, 1201.43it/s][A
    100%|█████████████████████████████████████| 4529/4529 [00:03<00:00, 1216.23it/s][A
    Testing DataLoader 0:  55%|███████████▌         | 17/31 [04:47<03:56,  0.06it/s]Sampling. The number of nodes to sample is 5021.
    Sampling on one graph took 10.478262424468994 seconds.
    
      0%|                                                  | 0/5021 [00:00<?, ?it/s][A
      2%|▊                                     | 112/5021 [00:00<00:04, 1116.65it/s][A
      4%|█▋                                    | 225/5021 [00:00<00:04, 1123.80it/s][A
      7%|██▌                                   | 338/5021 [00:00<00:04, 1125.44it/s][A
      9%|███▍                                  | 451/5021 [00:00<00:04, 1126.47it/s][A
     11%|████▎                                 | 565/5021 [00:00<00:03, 1127.57it/s][A
     14%|█████▏                                | 678/5021 [00:00<00:03, 1127.99it/s][A
     16%|█████▉                                | 791/5021 [00:00<00:03, 1127.78it/s][A
     18%|██████▊                               | 904/5021 [00:00<00:03, 1127.41it/s][A
     20%|███████▍                             | 1017/5021 [00:00<00:03, 1127.26it/s][A
     23%|████████▎                            | 1130/5021 [00:01<00:03, 1126.94it/s][A
     25%|█████████▏                           | 1243/5021 [00:01<00:03, 1126.99it/s][A
     27%|█████████▉                           | 1356/5021 [00:01<00:03, 1079.13it/s][A
     29%|██████████▊                          | 1465/5021 [00:01<00:03, 1048.78it/s][A
     31%|███████████▋                         | 1578/5021 [00:01<00:03, 1071.41it/s][A
     34%|████████████▍                        | 1691/5021 [00:01<00:03, 1088.49it/s][A
     36%|█████████████▎                       | 1804/5021 [00:01<00:02, 1099.75it/s][A
     38%|██████████████▏                      | 1917/5021 [00:01<00:02, 1107.72it/s][A
     40%|██████████████▉                      | 2030/5021 [00:01<00:02, 1113.85it/s][A
     43%|███████████████▊                     | 2143/5021 [00:01<00:02, 1117.30it/s][A
     45%|████████████████▌                    | 2256/5021 [00:02<00:02, 1119.92it/s][A
     47%|█████████████████▍                   | 2369/5021 [00:02<00:02, 1122.11it/s][A
     49%|██████████████████▎                  | 2482/5021 [00:02<00:02, 1123.63it/s][A
     52%|███████████████████                  | 2595/5021 [00:02<00:02, 1124.99it/s][A
     54%|███████████████████▉                 | 2708/5021 [00:02<00:02, 1125.08it/s][A
     56%|████████████████████▊                | 2821/5021 [00:02<00:02, 1078.52it/s][A
     58%|█████████████████████▌               | 2933/5021 [00:02<00:01, 1090.23it/s][A
     61%|██████████████████████▍              | 3045/5021 [00:02<00:01, 1098.78it/s][A
     63%|███████████████████████▎             | 3158/5021 [00:02<00:01, 1107.32it/s][A
     65%|████████████████████████             | 3271/5021 [00:02<00:01, 1113.26it/s][A
     67%|████████████████████████▉            | 3384/5021 [00:03<00:01, 1117.09it/s][A
     70%|█████████████████████████▊           | 3497/5021 [00:03<00:01, 1120.60it/s][A
     72%|██████████████████████████▌          | 3610/5021 [00:03<00:01, 1122.76it/s][A
     74%|███████████████████████████▍         | 3723/5021 [00:03<00:01, 1123.47it/s][A
     76%|████████████████████████████▎        | 3836/5021 [00:03<00:01, 1120.35it/s][A
     79%|█████████████████████████████        | 3949/5021 [00:03<00:00, 1122.40it/s][A
     81%|█████████████████████████████▉       | 4062/5021 [00:03<00:00, 1123.72it/s][A
     83%|██████████████████████████████▊      | 4175/5021 [00:03<00:00, 1075.08it/s][A
     85%|███████████████████████████████▌     | 4288/5021 [00:03<00:00, 1089.01it/s][A
     88%|████████████████████████████████▍    | 4401/5021 [00:03<00:00, 1100.59it/s][A
     90%|█████████████████████████████████▎   | 4514/5021 [00:04<00:00, 1107.99it/s][A
     92%|██████████████████████████████████   | 4627/5021 [00:04<00:00, 1113.99it/s][A
     94%|██████████████████████████████████▉  | 4740/5021 [00:04<00:00, 1117.65it/s][A
     97%|███████████████████████████████████▊ | 4853/5021 [00:04<00:00, 1120.23it/s][A
    100%|█████████████████████████████████████| 5021/5021 [00:04<00:00, 1111.45it/s][A
    Testing DataLoader 0:  58%|████████████▏        | 18/31 [05:08<03:42,  0.06it/s]Sampling. The number of nodes to sample is 4991.
    Sampling on one graph took 10.419379949569702 seconds.
    
      0%|                                                  | 0/4991 [00:00<?, ?it/s][A
      2%|▊                                     | 113/4991 [00:00<00:04, 1121.63it/s][A
      5%|█▋                                    | 226/4991 [00:00<00:04, 1125.31it/s][A
      7%|██▌                                   | 339/4991 [00:00<00:04, 1125.59it/s][A
      9%|███▍                                  | 452/4991 [00:00<00:04, 1126.05it/s][A
     11%|████▎                                 | 565/4991 [00:00<00:03, 1124.10it/s][A
     14%|█████▏                                | 679/4991 [00:00<00:03, 1127.50it/s][A
     16%|██████                                | 793/4991 [00:00<00:03, 1129.47it/s][A
     18%|██████▉                               | 906/4991 [00:00<00:03, 1079.09it/s][A
     20%|███████▌                             | 1015/4991 [00:00<00:03, 1048.53it/s][A
     23%|████████▎                            | 1129/4991 [00:01<00:03, 1072.74it/s][A
     25%|█████████▏                           | 1243/4991 [00:01<00:03, 1090.13it/s][A
     27%|██████████                           | 1356/4991 [00:01<00:03, 1100.95it/s][A
     29%|██████████▉                          | 1470/4991 [00:01<00:03, 1109.87it/s][A
     32%|███████████▋                         | 1583/4991 [00:01<00:03, 1115.85it/s][A
     34%|████████████▌                        | 1697/4991 [00:01<00:02, 1120.43it/s][A
     36%|█████████████▍                       | 1811/4991 [00:01<00:02, 1123.86it/s][A
     39%|██████████████▎                      | 1925/4991 [00:01<00:02, 1125.86it/s][A
     41%|███████████████                      | 2039/4991 [00:01<00:02, 1128.47it/s][A
     43%|███████████████▉                     | 2153/4991 [00:01<00:02, 1130.34it/s][A
     45%|████████████████▊                    | 2267/4991 [00:02<00:02, 1131.26it/s][A
     48%|█████████████████▋                   | 2381/4991 [00:02<00:02, 1087.62it/s][A
     50%|██████████████████▍                  | 2495/4991 [00:02<00:02, 1100.98it/s][A
     52%|███████████████████▎                 | 2609/4991 [00:02<00:02, 1110.09it/s][A
     55%|████████████████████▏                | 2723/4991 [00:02<00:02, 1117.28it/s][A
     57%|█████████████████████                | 2836/4991 [00:02<00:01, 1118.82it/s][A
     59%|█████████████████████▊               | 2950/4991 [00:02<00:01, 1124.37it/s][A
     61%|██████████████████████▋              | 3064/4991 [00:02<00:01, 1127.23it/s][A
     64%|███████████████████████▌             | 3178/4991 [00:02<00:01, 1129.36it/s][A
     66%|████████████████████████▍            | 3292/4991 [00:02<00:01, 1130.58it/s][A
     68%|█████████████████████████▏           | 3406/4991 [00:03<00:01, 1131.89it/s][A
     71%|██████████████████████████           | 3520/4991 [00:03<00:01, 1131.87it/s][A
     73%|██████████████████████████▉          | 3634/4991 [00:03<00:01, 1125.94it/s][A
     75%|███████████████████████████▊         | 3747/4991 [00:03<00:01, 1120.51it/s][A
     77%|████████████████████████████▌        | 3860/4991 [00:03<00:01, 1041.51it/s][A
     80%|█████████████████████████████▍       | 3972/4991 [00:03<00:00, 1062.12it/s][A
     82%|██████████████████████████████▎      | 4086/4991 [00:03<00:00, 1083.56it/s][A
     84%|███████████████████████████████▏     | 4200/4991 [00:03<00:00, 1098.07it/s][A
     86%|███████████████████████████████▉     | 4313/4991 [00:03<00:00, 1106.88it/s][A
     89%|████████████████████████████████▊    | 4427/4991 [00:03<00:00, 1115.21it/s][A
     91%|█████████████████████████████████▋   | 4540/4991 [00:04<00:00, 1118.94it/s][A
     93%|██████████████████████████████████▍  | 4653/4991 [00:04<00:00, 1121.93it/s][A
     96%|███████████████████████████████████▎ | 4767/4991 [00:04<00:00, 1125.42it/s][A
    100%|█████████████████████████████████████| 4991/4991 [00:04<00:00, 1112.38it/s][A
    Testing DataLoader 0:  61%|████████████▊        | 19/31 [05:29<03:28,  0.06it/s]Sampling. The number of nodes to sample is 4389.
    Sampling on one graph took 9.768028020858765 seconds.
    
      0%|                                                  | 0/4389 [00:00<?, ?it/s][A
      3%|█                                     | 127/4389 [00:00<00:03, 1264.53it/s][A
      6%|██▏                                   | 254/4389 [00:00<00:03, 1210.51it/s][A
      9%|███▎                                  | 376/4389 [00:00<00:03, 1138.96it/s][A
     11%|████▎                                 | 501/4389 [00:00<00:03, 1179.76it/s][A
     14%|█████▍                                | 628/4389 [00:00<00:03, 1211.13it/s][A
     17%|██████▌                               | 755/4389 [00:00<00:02, 1229.82it/s][A
     20%|███████▋                              | 882/4389 [00:00<00:02, 1239.96it/s][A
     23%|████████▌                            | 1009/4389 [00:00<00:02, 1248.25it/s][A
     26%|█████████▌                           | 1136/4389 [00:00<00:02, 1252.68it/s][A
     29%|██████████▋                          | 1263/4389 [00:01<00:02, 1256.93it/s][A
     32%|███████████▋                         | 1390/4389 [00:01<00:02, 1259.74it/s][A
     35%|████████████▊                        | 1518/4389 [00:01<00:02, 1262.90it/s][A
     37%|█████████████▊                       | 1645/4389 [00:01<00:02, 1264.31it/s][A
     40%|██████████████▉                      | 1772/4389 [00:01<00:02, 1265.22it/s][A
     43%|████████████████                     | 1899/4389 [00:01<00:02, 1211.70it/s][A
     46%|█████████████████                    | 2021/4389 [00:01<00:02, 1180.33it/s][A
     49%|██████████████████                   | 2148/4389 [00:01<00:01, 1205.87it/s][A
     52%|███████████████████▏                 | 2274/4389 [00:01<00:01, 1220.47it/s][A
     55%|████████████████████▏                | 2400/4389 [00:01<00:01, 1230.62it/s][A
     58%|█████████████████████▎               | 2527/4389 [00:02<00:01, 1242.02it/s][A
     60%|██████████████████████▎              | 2654/4389 [00:02<00:01, 1250.11it/s][A
     63%|███████████████████████▍             | 2781/4389 [00:02<00:01, 1255.83it/s][A
     66%|████████████████████████▌            | 2909/4389 [00:02<00:01, 1260.52it/s][A
     69%|█████████████████████████▌           | 3036/4389 [00:02<00:01, 1263.21it/s][A
     72%|██████████████████████████▋          | 3163/4389 [00:02<00:00, 1260.70it/s][A
     75%|███████████████████████████▋         | 3290/4389 [00:02<00:00, 1262.66it/s][A
     78%|████████████████████████████▊        | 3417/4389 [00:02<00:00, 1263.64it/s][A
     81%|█████████████████████████████▉       | 3545/4389 [00:02<00:00, 1265.60it/s][A
     84%|██████████████████████████████▉      | 3672/4389 [00:02<00:00, 1209.16it/s][A
     86%|███████████████████████████████▉     | 3794/4389 [00:03<00:00, 1180.79it/s][A
     89%|█████████████████████████████████    | 3922/4389 [00:03<00:00, 1207.53it/s][A
     92%|██████████████████████████████████▏  | 4050/4389 [00:03<00:00, 1228.15it/s][A
     95%|███████████████████████████████████▏ | 4178/4389 [00:03<00:00, 1241.18it/s][A
    100%|█████████████████████████████████████| 4389/4389 [00:03<00:00, 1236.46it/s][A
    Testing DataLoader 0:  65%|█████████████▌       | 20/31 [05:48<03:11,  0.06it/s]Sampling. The number of nodes to sample is 3560.
    Sampling on one graph took 8.668197393417358 seconds.
    
      0%|                                                  | 0/3560 [00:00<?, ?it/s][A
      4%|█▌                                    | 150/3560 [00:00<00:02, 1497.32it/s][A
      8%|███▏                                  | 301/3560 [00:00<00:02, 1503.10it/s][A
     13%|████▊                                 | 452/3560 [00:00<00:02, 1504.28it/s][A
     17%|██████▍                               | 603/3560 [00:00<00:01, 1506.02it/s][A
     21%|████████                              | 754/3560 [00:00<00:01, 1505.24it/s][A
     25%|█████████▋                            | 905/3560 [00:00<00:01, 1504.79it/s][A
     30%|██████████▉                          | 1056/3560 [00:00<00:01, 1503.94it/s][A
     34%|████████████▌                        | 1207/3560 [00:00<00:01, 1504.02it/s][A
     38%|██████████████                       | 1358/3560 [00:00<00:01, 1503.75it/s][A
     42%|███████████████▋                     | 1509/3560 [00:01<00:01, 1502.61it/s][A
     47%|█████████████████▎                   | 1660/3560 [00:01<00:01, 1504.11it/s][A
     51%|██████████████████▊                  | 1811/3560 [00:01<00:01, 1505.04it/s][A
     55%|████████████████████▍                | 1962/3560 [00:01<00:01, 1504.75it/s][A
     59%|█████████████████████▉               | 2113/3560 [00:01<00:00, 1505.35it/s][A
     64%|███████████████████████▌             | 2264/3560 [00:01<00:00, 1503.86it/s][A
     68%|█████████████████████████            | 2415/3560 [00:01<00:00, 1504.72it/s][A
     72%|██████████████████████████▋          | 2566/3560 [00:01<00:00, 1445.65it/s][A
     76%|████████████████████████████▏        | 2717/3560 [00:01<00:00, 1462.41it/s][A
     81%|█████████████████████████████▊       | 2868/3560 [00:01<00:00, 1473.88it/s][A
     85%|███████████████████████████████▎     | 3018/3560 [00:02<00:00, 1479.40it/s][A
     89%|████████████████████████████████▉    | 3169/3560 [00:02<00:00, 1486.76it/s][A
     93%|██████████████████████████████████▌  | 3320/3560 [00:02<00:00, 1490.92it/s][A
    100%|█████████████████████████████████████| 3560/3560 [00:02<00:00, 1494.64it/s][A
    Testing DataLoader 0:  68%|██████████████▏      | 21/31 [06:02<02:52,  0.06it/s]Sampling. The number of nodes to sample is 1706.
    Sampling on one graph took 8.593971252441406 seconds.
    
      0%|                                                  | 0/1706 [00:00<?, ?it/s][A
     15%|█████▌                                | 250/1706 [00:00<00:00, 2494.42it/s][A
     29%|███████████▏                          | 500/1706 [00:00<00:00, 2489.59it/s][A
     44%|████████████████▊                     | 754/1706 [00:00<00:00, 2509.48it/s][A
     59%|█████████████████████▊               | 1007/1706 [00:00<00:00, 2514.91it/s][A
     74%|███████████████████████████▎         | 1260/1706 [00:00<00:00, 2518.42it/s][A
    100%|█████████████████████████████████████| 1706/1706 [00:00<00:00, 2514.92it/s][A
    Testing DataLoader 0:  71%|██████████████▉      | 22/31 [06:12<02:32,  0.06it/s]Sampling. The number of nodes to sample is 3296.
    Sampling on one graph took 8.5642249584198 seconds.
    
      0%|                                                  | 0/3296 [00:00<?, ?it/s][A
      5%|█▊                                    | 159/3296 [00:00<00:01, 1585.91it/s][A
     10%|███▋                                  | 320/3296 [00:00<00:01, 1596.10it/s][A
     15%|█████▌                                | 481/3296 [00:00<00:01, 1600.67it/s][A
     19%|███████▍                              | 642/3296 [00:00<00:01, 1597.35it/s][A
     24%|█████████▎                            | 803/3296 [00:00<00:01, 1599.77it/s][A
     29%|███████████                           | 964/3296 [00:00<00:01, 1601.85it/s][A
     34%|████████████▋                        | 1125/3296 [00:00<00:01, 1603.36it/s][A
     39%|██████████████▍                      | 1286/3296 [00:00<00:01, 1529.65it/s][A
     44%|████████████████▏                    | 1447/3296 [00:00<00:01, 1553.26it/s][A
     49%|██████████████████                   | 1608/3296 [00:01<00:01, 1569.98it/s][A
     54%|███████████████████▊                 | 1768/3296 [00:01<00:00, 1577.26it/s][A
     59%|█████████████████████▋               | 1929/3296 [00:01<00:00, 1586.02it/s][A
     63%|███████████████████████▍             | 2090/3296 [00:01<00:00, 1592.63it/s][A
     68%|█████████████████████████▎           | 2251/3296 [00:01<00:00, 1596.79it/s][A
     73%|███████████████████████████          | 2412/3296 [00:01<00:00, 1600.53it/s][A
     78%|████████████████████████████▉        | 2573/3296 [00:01<00:00, 1603.09it/s][A
     83%|██████████████████████████████▋      | 2735/3296 [00:01<00:00, 1605.49it/s][A
     88%|████████████████████████████████▌    | 2896/3296 [00:01<00:00, 1604.86it/s][A
     93%|██████████████████████████████████▎  | 3058/3296 [00:01<00:00, 1606.94it/s][A
    100%|█████████████████████████████████████| 3296/3296 [00:02<00:00, 1581.23it/s][A
    Testing DataLoader 0:  74%|███████████████▌     | 23/31 [06:26<02:14,  0.06it/s]Sampling. The number of nodes to sample is 5155.
    Sampling on one graph took 10.487604141235352 seconds.
    
      0%|                                                  | 0/5155 [00:00<?, ?it/s][A
      2%|▊                                     | 111/5155 [00:00<00:04, 1100.58it/s][A
      4%|█▋                                    | 222/5155 [00:00<00:04, 1104.65it/s][A
      6%|██▍                                   | 334/5155 [00:00<00:04, 1107.74it/s][A
      9%|███▎                                  | 446/5155 [00:00<00:04, 1108.71it/s][A
     11%|████                                  | 558/5155 [00:00<00:04, 1109.41it/s][A
     13%|████▉                                 | 669/5155 [00:00<00:04, 1106.13it/s][A
     15%|█████▋                                | 780/5155 [00:00<00:04, 1055.95it/s][A
     17%|██████▌                               | 892/5155 [00:00<00:03, 1073.78it/s][A
     19%|███████▏                             | 1004/5155 [00:00<00:03, 1085.47it/s][A
     22%|████████                             | 1115/5155 [00:01<00:03, 1092.86it/s][A
     24%|████████▊                            | 1227/5155 [00:01<00:03, 1098.25it/s][A
     26%|█████████▌                           | 1339/5155 [00:01<00:03, 1102.30it/s][A
     28%|██████████▍                          | 1450/5155 [00:01<00:03, 1100.33it/s][A
     30%|███████████▏                         | 1561/5155 [00:01<00:03, 1102.84it/s][A
     32%|████████████                         | 1673/5155 [00:01<00:03, 1105.55it/s][A
     35%|████████████▊                        | 1784/5155 [00:01<00:03, 1105.61it/s][A
     37%|█████████████▌                       | 1895/5155 [00:01<00:02, 1106.22it/s][A
     39%|██████████████▍                      | 2006/5155 [00:01<00:02, 1107.01it/s][A
     41%|███████████████▏                     | 2117/5155 [00:01<00:02, 1079.39it/s][A
     43%|███████████████▉                     | 2226/5155 [00:02<00:02, 1071.18it/s][A
     45%|████████████████▊                    | 2338/5155 [00:02<00:02, 1083.19it/s][A
     48%|█████████████████▌                   | 2449/5155 [00:02<00:02, 1089.47it/s][A
     50%|██████████████████▍                  | 2561/5155 [00:02<00:02, 1095.81it/s][A
     52%|███████████████████▏                 | 2672/5155 [00:02<00:02, 1099.97it/s][A
     54%|███████████████████▉                 | 2785/5155 [00:02<00:02, 1106.09it/s][A
     56%|████████████████████▊                | 2897/5155 [00:02<00:02, 1109.95it/s][A
     58%|█████████████████████▌               | 3009/5155 [00:02<00:01, 1106.70it/s][A
     61%|██████████████████████▍              | 3121/5155 [00:02<00:01, 1109.93it/s][A
     63%|███████████████████████▏             | 3233/5155 [00:02<00:01, 1111.68it/s][A
     65%|████████████████████████             | 3345/5155 [00:03<00:01, 1112.71it/s][A
     67%|████████████████████████▊            | 3457/5155 [00:03<00:01, 1114.62it/s][A
     69%|█████████████████████████▌           | 3569/5155 [00:03<00:01, 1071.82it/s][A
     71%|██████████████████████████▍          | 3681/5155 [00:03<00:01, 1084.48it/s][A
     74%|███████████████████████████▏         | 3793/5155 [00:03<00:01, 1094.08it/s][A
     76%|████████████████████████████         | 3905/5155 [00:03<00:01, 1099.69it/s][A
     78%|████████████████████████████▊        | 4017/5155 [00:03<00:01, 1104.97it/s][A
     80%|█████████████████████████████▋       | 4129/5155 [00:03<00:00, 1106.42it/s][A
     82%|██████████████████████████████▍      | 4241/5155 [00:03<00:00, 1109.21it/s][A
     84%|███████████████████████████████▎     | 4354/5155 [00:03<00:00, 1112.56it/s][A
     87%|████████████████████████████████     | 4466/5155 [00:04<00:00, 1111.40it/s][A
     89%|████████████████████████████████▊    | 4578/5155 [00:04<00:00, 1113.55it/s][A
     91%|█████████████████████████████████▋   | 4691/5155 [00:04<00:00, 1115.60it/s][A
     93%|██████████████████████████████████▍  | 4804/5155 [00:04<00:00, 1118.29it/s][A
     95%|███████████████████████████████████▎ | 4917/5155 [00:04<00:00, 1119.68it/s][A
     98%|████████████████████████████████████ | 5029/5155 [00:04<00:00, 1074.13it/s][A
    100%|█████████████████████████████████████| 5155/5155 [00:04<00:00, 1095.82it/s][A
    Testing DataLoader 0:  77%|████████████████▎    | 24/31 [06:48<01:59,  0.06it/s]Sampling. The number of nodes to sample is 3733.
    Sampling on one graph took 8.769573211669922 seconds.
    
      0%|                                                  | 0/3733 [00:00<?, ?it/s][A
      4%|█▍                                    | 146/3733 [00:00<00:02, 1459.17it/s][A
      8%|██▉                                   | 293/3733 [00:00<00:02, 1462.91it/s][A
     12%|████▍                                 | 440/3733 [00:00<00:02, 1464.40it/s][A
     16%|█████▉                                | 587/3733 [00:00<00:02, 1388.11it/s][A
     20%|███████▍                              | 735/3733 [00:00<00:02, 1420.03it/s][A
     24%|████████▉                             | 883/3733 [00:00<00:01, 1438.61it/s][A
     28%|██████████▏                          | 1031/3733 [00:00<00:01, 1451.18it/s][A
     32%|███████████▋                         | 1179/3733 [00:00<00:01, 1459.79it/s][A
     36%|█████████████▏                       | 1327/3733 [00:00<00:01, 1463.92it/s][A
     40%|██████████████▋                      | 1476/3733 [00:01<00:01, 1469.38it/s][A
     44%|████████████████                     | 1624/3733 [00:01<00:01, 1471.14it/s][A
     47%|█████████████████▌                   | 1772/3733 [00:01<00:01, 1468.50it/s][A
     51%|███████████████████                  | 1920/3733 [00:01<00:01, 1469.67it/s][A
     55%|████████████████████▍                | 2068/3733 [00:01<00:01, 1472.69it/s][A
     59%|█████████████████████▉               | 2216/3733 [00:01<00:01, 1474.24it/s][A
     63%|███████████████████████▍             | 2364/3733 [00:01<00:00, 1475.46it/s][A
     67%|████████████████████████▉            | 2512/3733 [00:01<00:00, 1419.07it/s][A
     71%|██████████████████████████▎          | 2660/3733 [00:01<00:00, 1435.71it/s][A
     75%|███████████████████████████▊         | 2808/3733 [00:01<00:00, 1447.12it/s][A
     79%|█████████████████████████████▎       | 2956/3733 [00:02<00:00, 1455.41it/s][A
     83%|██████████████████████████████▊      | 3104/3733 [00:02<00:00, 1461.30it/s][A
     87%|████████████████████████████████▏    | 3252/3733 [00:02<00:00, 1464.48it/s][A
     91%|█████████████████████████████████▋   | 3399/3733 [00:02<00:00, 1458.40it/s][A
     95%|███████████████████████████████████▏ | 3547/3733 [00:02<00:00, 1462.58it/s][A
    100%|█████████████████████████████████████| 3733/3733 [00:02<00:00, 1456.80it/s][A
    Testing DataLoader 0:  81%|████████████████▉    | 25/31 [07:03<01:41,  0.06it/s]Sampling. The number of nodes to sample is 3780.
    Sampling on one graph took 8.795906066894531 seconds.
    
      0%|                                                  | 0/3780 [00:00<?, ?it/s][A
      4%|█▍                                    | 144/3780 [00:00<00:02, 1437.22it/s][A
      8%|██▉                                   | 289/3780 [00:00<00:02, 1443.06it/s][A
     12%|████▎                                 | 435/3780 [00:00<00:02, 1447.53it/s][A
     15%|█████▊                                | 580/3780 [00:00<00:02, 1448.14it/s][A
     19%|███████▎                              | 725/3780 [00:00<00:02, 1448.06it/s][A
     23%|████████▋                             | 870/3780 [00:00<00:02, 1441.84it/s][A
     27%|█████████▉                           | 1017/3780 [00:00<00:01, 1448.23it/s][A
     31%|███████████▍                         | 1163/3780 [00:00<00:01, 1451.87it/s][A
     35%|████████████▊                        | 1310/3780 [00:00<00:01, 1455.47it/s][A
     39%|██████████████▎                      | 1457/3780 [00:01<00:01, 1458.20it/s][A
     42%|███████████████▋                     | 1604/3780 [00:01<00:01, 1459.13it/s][A
     46%|█████████████████▏                   | 1750/3780 [00:01<00:01, 1394.22it/s][A
     50%|██████████████████▌                  | 1891/3780 [00:01<00:01, 1357.12it/s][A
     54%|███████████████████▉                 | 2038/3780 [00:01<00:01, 1387.22it/s][A
     58%|█████████████████████▍               | 2184/3780 [00:01<00:01, 1406.69it/s][A
     62%|██████████████████████▊              | 2330/3780 [00:01<00:01, 1420.75it/s][A
     66%|████████████████████████▏            | 2476/3780 [00:01<00:00, 1430.33it/s][A
     69%|█████████████████████████▋           | 2622/3780 [00:01<00:00, 1437.98it/s][A
     73%|███████████████████████████          | 2768/3780 [00:01<00:00, 1443.36it/s][A
     77%|████████████████████████████▌        | 2914/3780 [00:02<00:00, 1448.30it/s][A
     81%|█████████████████████████████▉       | 3060/3780 [00:02<00:00, 1451.32it/s][A
     85%|███████████████████████████████▍     | 3206/3780 [00:02<00:00, 1452.87it/s][A
     89%|████████████████████████████████▊    | 3352/3780 [00:02<00:00, 1453.09it/s][A
     93%|██████████████████████████████████▏  | 3498/3780 [00:02<00:00, 1453.05it/s][A
    100%|█████████████████████████████████████| 3780/3780 [00:02<00:00, 1429.83it/s][A
    Testing DataLoader 0:  84%|█████████████████▌   | 26/31 [07:18<01:24,  0.06it/s]Sampling. The number of nodes to sample is 2905.
    Sampling on one graph took 8.53706979751587 seconds.
    
      0%|                                                  | 0/2905 [00:00<?, ?it/s][A
      6%|██▎                                   | 176/2905 [00:00<00:01, 1750.79it/s][A
     12%|████▌                                 | 352/2905 [00:00<00:01, 1747.20it/s][A
     18%|██████▉                               | 528/2905 [00:00<00:01, 1748.65it/s][A
     24%|█████████▏                            | 704/2905 [00:00<00:01, 1750.58it/s][A
     30%|███████████▌                          | 880/2905 [00:00<00:01, 1752.01it/s][A
     36%|█████████████▍                       | 1056/2905 [00:00<00:01, 1754.39it/s][A
     42%|███████████████▋                     | 1232/2905 [00:00<00:00, 1752.45it/s][A
     48%|█████████████████▉                   | 1408/2905 [00:00<00:00, 1680.75it/s][A
     55%|████████████████████▏                | 1584/2905 [00:00<00:00, 1702.70it/s][A
     61%|██████████████████████▍              | 1759/2905 [00:01<00:00, 1716.91it/s][A
     67%|████████████████████████▋            | 1935/2905 [00:01<00:00, 1727.33it/s][A
     73%|██████████████████████████▉          | 2111/2905 [00:01<00:00, 1736.32it/s][A
     79%|█████████████████████████████▏       | 2287/2905 [00:01<00:00, 1740.82it/s][A
     85%|███████████████████████████████▎     | 2463/2905 [00:01<00:00, 1746.05it/s][A
     91%|█████████████████████████████████▌   | 2638/2905 [00:01<00:00, 1745.79it/s][A
    100%|█████████████████████████████████████| 2905/2905 [00:01<00:00, 1737.54it/s][A
    Testing DataLoader 0:  87%|██████████████████▎  | 27/31 [07:30<01:06,  0.06it/s]Sampling. The number of nodes to sample is 3192.
    Sampling on one graph took 8.548822164535522 seconds.
    
      0%|                                                  | 0/3192 [00:00<?, ?it/s][A
      4%|█▋                                    | 139/3192 [00:00<00:02, 1388.65it/s][A
     10%|███▋                                  | 305/3192 [00:00<00:01, 1547.04it/s][A
     15%|█████▌                                | 472/3192 [00:00<00:01, 1599.74it/s][A
     20%|███████▌                              | 638/3192 [00:00<00:01, 1621.54it/s][A
     25%|█████████▌                            | 805/3192 [00:00<00:01, 1636.94it/s][A
     30%|███████████▌                          | 971/3192 [00:00<00:01, 1643.62it/s][A
     36%|█████████████▏                       | 1137/3192 [00:00<00:01, 1648.44it/s][A
     41%|███████████████                      | 1303/3192 [00:00<00:01, 1651.04it/s][A
     46%|█████████████████                    | 1470/3192 [00:00<00:01, 1654.00it/s][A
     51%|██████████████████▉                  | 1636/3192 [00:01<00:00, 1653.55it/s][A
     56%|████████████████████▉                | 1803/3192 [00:01<00:00, 1656.17it/s][A
     62%|██████████████████████▊              | 1969/3192 [00:01<00:00, 1652.54it/s][A
     67%|████████████████████████▋            | 2135/3192 [00:01<00:00, 1652.24it/s][A
     72%|██████████████████████████▋          | 2301/3192 [00:01<00:00, 1522.41it/s][A
     77%|████████████████████████████▌        | 2464/3192 [00:01<00:00, 1552.13it/s][A
     82%|██████████████████████████████▍      | 2630/3192 [00:01<00:00, 1580.67it/s][A
     88%|████████████████████████████████▍    | 2796/3192 [00:01<00:00, 1602.90it/s][A
     93%|██████████████████████████████████▎  | 2963/3192 [00:01<00:00, 1619.98it/s][A
    100%|█████████████████████████████████████| 3192/3192 [00:01<00:00, 1617.12it/s][A
    Testing DataLoader 0:  90%|██████████████████▉  | 28/31 [07:44<00:49,  0.06it/s]Sampling. The number of nodes to sample is 4057.
    Sampling on one graph took 8.982274293899536 seconds.
    
      0%|                                                  | 0/4057 [00:00<?, ?it/s][A
      3%|█▎                                    | 137/4057 [00:00<00:02, 1364.08it/s][A
      7%|██▌                                   | 275/4057 [00:00<00:02, 1370.86it/s][A
     10%|███▊                                  | 413/4057 [00:00<00:02, 1370.75it/s][A
     14%|█████▏                                | 551/4057 [00:00<00:02, 1366.14it/s][A
     17%|██████▍                               | 689/4057 [00:00<00:02, 1368.34it/s][A
     20%|███████▋                              | 826/4057 [00:00<00:02, 1368.42it/s][A
     24%|█████████                             | 963/4057 [00:00<00:02, 1314.74it/s][A
     27%|█████████▉                           | 1095/4057 [00:00<00:02, 1271.90it/s][A
     30%|███████████▏                         | 1233/4057 [00:00<00:02, 1302.93it/s][A
     34%|████████████▍                        | 1370/4057 [00:01<00:02, 1321.19it/s][A
     37%|█████████████▊                       | 1508/4057 [00:01<00:01, 1336.59it/s][A
     41%|███████████████                      | 1646/4057 [00:01<00:01, 1346.71it/s][A
     44%|████████████████▎                    | 1783/4057 [00:01<00:01, 1352.79it/s][A
     47%|█████████████████▌                   | 1921/4057 [00:01<00:01, 1358.40it/s][A
     51%|██████████████████▊                  | 2059/4057 [00:01<00:01, 1362.06it/s][A
     54%|████████████████████                 | 2196/4057 [00:01<00:01, 1363.03it/s][A
     58%|█████████████████████▎               | 2334/4057 [00:01<00:01, 1365.90it/s][A
     61%|██████████████████████▌              | 2472/4057 [00:01<00:01, 1367.86it/s][A
     64%|███████████████████████▊             | 2610/4057 [00:01<00:01, 1368.74it/s][A
     68%|█████████████████████████            | 2747/4057 [00:02<00:00, 1314.66it/s][A
     71%|██████████████████████████▎          | 2885/4057 [00:02<00:00, 1331.95it/s][A
     75%|███████████████████████████▌         | 3023/4057 [00:02<00:00, 1343.86it/s][A
     78%|████████████████████████████▊        | 3161/4057 [00:02<00:00, 1353.08it/s][A
     81%|██████████████████████████████       | 3298/4057 [00:02<00:00, 1357.35it/s][A
     85%|███████████████████████████████▎     | 3435/4057 [00:02<00:00, 1361.05it/s][A
     88%|████████████████████████████████▌    | 3572/4057 [00:02<00:00, 1363.43it/s][A
     91%|█████████████████████████████████▊   | 3710/4057 [00:02<00:00, 1365.69it/s][A
     95%|███████████████████████████████████  | 3847/4057 [00:02<00:00, 1366.36it/s][A
    100%|█████████████████████████████████████| 4057/4057 [00:03<00:00, 1350.70it/s][A
    Testing DataLoader 0:  94%|███████████████████▋ | 29/31 [08:00<00:33,  0.06it/s]Sampling. The number of nodes to sample is 1160.
    Sampling on one graph took 8.577382802963257 seconds.
    
      0%|                                                  | 0/1160 [00:00<?, ?it/s][A
     27%|██████████▎                           | 313/1160 [00:00<00:00, 3121.77it/s][A
     54%|████████████████████▌                 | 626/1160 [00:00<00:00, 3120.04it/s][A
    100%|█████████████████████████████████████| 1160/1160 [00:00<00:00, 3118.47it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  97%|████████████████████▎| 30/31 [08:09<00:16,  0.06it/s]Sampling. The number of nodes to sample is 3316.
    Sampling on one graph took 8.64570140838623 seconds.
    
      0%|                                                  | 0/3316 [00:00<?, ?it/s][A
      5%|█▊                                    | 161/3316 [00:00<00:01, 1600.48it/s][A
     10%|███▋                                  | 322/3316 [00:00<00:01, 1501.04it/s][A
     14%|█████▍                                | 479/3316 [00:00<00:01, 1527.38it/s][A
     19%|███████▎                              | 641/3316 [00:00<00:01, 1560.98it/s][A
     24%|█████████▏                            | 803/3316 [00:00<00:01, 1579.93it/s][A
     29%|███████████                           | 964/3316 [00:00<00:01, 1589.00it/s][A
     34%|████████████▌                        | 1126/3316 [00:00<00:01, 1596.05it/s][A
     39%|██████████████▎                      | 1287/3316 [00:00<00:01, 1600.36it/s][A
     44%|████████████████▏                    | 1449/3316 [00:00<00:01, 1604.49it/s][A
     49%|█████████████████▉                   | 1610/3316 [00:01<00:01, 1606.09it/s][A
     53%|███████████████████▊                 | 1771/3316 [00:01<00:00, 1606.65it/s][A
     58%|█████████████████████▌               | 1932/3316 [00:01<00:00, 1607.04it/s][A
     63%|███████████████████████▎             | 2094/3316 [00:01<00:00, 1608.21it/s][A
     68%|█████████████████████████▏           | 2255/3316 [00:01<00:00, 1608.56it/s][A
     73%|██████████████████████████▉          | 2416/3316 [00:01<00:00, 1534.62it/s][A
     78%|████████████████████████████▋        | 2571/3316 [00:01<00:00, 1487.46it/s][A
     82%|██████████████████████████████▍      | 2731/3316 [00:01<00:00, 1517.83it/s][A
     87%|████████████████████████████████▎    | 2893/3316 [00:01<00:00, 1545.10it/s][A
     92%|██████████████████████████████████   | 3054/3316 [00:01<00:00, 1562.96it/s][A
    100%|█████████████████████████████████████| 3316/3316 [00:02<00:00, 1573.03it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0: 100%|█████████████████████| 31/31 [08:23<00:00,  0.06it/s]
    Testing checkpoint: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=749.ckpt
    Epoch index: 749
    [INFO]: Validation is disabled.
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    [rank: 0] Seed set to 0
    Restoring states from the checkpoint path at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=749.ckpt
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    Loaded model weights from the checkpoint at /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/checkpoints/epoch=749.ckpt
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:492: Your `test_dataloader`'s sampler has shuffling enabled, it is strongly recommended that you turn shuffling off for val/test dataloaders.
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/utils/data/dataloader.py:627: UserWarning: This DataLoader will create 32 worker processes in total. Our suggested max number of worker in current system is 6, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
      warnings.warn(
    /nfs/team361/mv11/.venvs/LUNA/lib/python3.10/site-packages/torch/distributed/distributed_c10d.py:4807: UserWarning: No device id is provided via `init_process_group` or `barrier `. Using the current device set by the user. 
      warnings.warn(  # warn only once
    Testing DataLoader 0:   0%|                              | 0/31 [00:00<?, ?it/s]Sampling. The number of nodes to sample is 3851.
    Sampling on one graph took 8.802773475646973 seconds.
    
      0%|                                                  | 0/3851 [00:00<?, ?it/s][A
      4%|█▍                                    | 141/3851 [00:00<00:02, 1401.59it/s][A
      7%|██▊                                   | 283/3851 [00:00<00:02, 1409.05it/s][A
     11%|████▏                                 | 424/3851 [00:00<00:02, 1408.28it/s][A
     15%|█████▌                                | 565/3851 [00:00<00:02, 1334.35it/s][A
     18%|██████▉                               | 706/3851 [00:00<00:02, 1357.58it/s][A
     22%|████████▎                             | 848/3851 [00:00<00:02, 1375.84it/s][A
     26%|█████████▋                            | 987/3851 [00:00<00:02, 1379.38it/s][A
     29%|██████████▊                          | 1129/3851 [00:00<00:01, 1389.23it/s][A
     33%|████████████▏                        | 1270/3851 [00:00<00:01, 1394.98it/s][A
     37%|█████████████▌                       | 1412/3851 [00:01<00:01, 1399.92it/s][A
     40%|██████████████▉                      | 1553/3851 [00:01<00:01, 1402.92it/s][A
     44%|████████████████▎                    | 1694/3851 [00:01<00:01, 1404.53it/s][A
     48%|█████████████████▋                   | 1836/3851 [00:01<00:01, 1407.34it/s][A
     51%|██████████████████▉                  | 1977/3851 [00:01<00:01, 1405.47it/s][A
     55%|████████████████████▎                | 2119/3851 [00:01<00:01, 1407.26it/s][A
     59%|█████████████████████▋               | 2260/3851 [00:01<00:01, 1407.68it/s][A
     62%|███████████████████████              | 2401/3851 [00:01<00:01, 1407.98it/s][A
     66%|████████████████████████▍            | 2542/3851 [00:01<00:00, 1407.33it/s][A
     70%|█████████████████████████▊           | 2683/3851 [00:01<00:00, 1406.97it/s][A
     73%|███████████████████████████▏         | 2824/3851 [00:02<00:00, 1407.51it/s][A
     77%|████████████████████████████▍        | 2966/3851 [00:02<00:00, 1408.32it/s][A
     81%|█████████████████████████████▊       | 3107/3851 [00:02<00:00, 1408.00it/s][A
     84%|███████████████████████████████▏     | 3248/3851 [00:02<00:00, 1407.57it/s][A
     88%|████████████████████████████████▌    | 3389/3851 [00:02<00:00, 1407.31it/s][A
     92%|█████████████████████████████████▉   | 3531/3851 [00:02<00:00, 1408.75it/s][A
     95%|███████████████████████████████████▎ | 3672/3851 [00:02<00:00, 1407.89it/s][A
    100%|█████████████████████████████████████| 3851/3851 [00:02<00:00, 1400.25it/s][A
    Testing DataLoader 0:   3%|▋                     | 1/31 [00:15<07:40,  0.07it/s]Sampling. The number of nodes to sample is 670.
    Sampling on one graph took 8.396567821502686 seconds.
    
      0%|                                                   | 0/670 [00:00<?, ?it/s][A
    100%|███████████████████████████████████████| 670/670 [00:00<00:00, 3970.53it/s][A
    Testing DataLoader 0:   6%|█▍                    | 2/31 [00:24<05:53,  0.08it/s]Sampling. The number of nodes to sample is 4360.
    Sampling on one graph took 9.53651738166809 seconds.
    
      0%|                                                  | 0/4360 [00:00<?, ?it/s][A
      3%|█                                     | 127/4360 [00:00<00:03, 1265.93it/s][A
      6%|██▏                                   | 255/4360 [00:00<00:03, 1270.83it/s][A
      9%|███▎                                  | 383/4360 [00:00<00:03, 1274.12it/s][A
     12%|████▍                                 | 511/4360 [00:00<00:03, 1275.08it/s][A
     15%|█████▌                                | 639/4360 [00:00<00:02, 1273.89it/s][A
     18%|██████▋                               | 767/4360 [00:00<00:02, 1273.11it/s][A
     21%|███████▊                              | 895/4360 [00:00<00:02, 1274.32it/s][A
     23%|████████▋                            | 1024/4360 [00:00<00:02, 1276.90it/s][A
     26%|█████████▊                           | 1152/4360 [00:00<00:02, 1274.51it/s][A
     29%|██████████▊                          | 1280/4360 [00:01<00:02, 1275.73it/s][A
     32%|███████████▉                         | 1409/4360 [00:01<00:02, 1277.60it/s][A
     35%|█████████████                        | 1537/4360 [00:01<00:02, 1266.67it/s][A
     38%|██████████████▏                      | 1665/4360 [00:01<00:02, 1268.74it/s][A
     41%|███████████████▏                     | 1793/4360 [00:01<00:02, 1271.29it/s][A
     44%|████████████████▎                    | 1921/4360 [00:01<00:01, 1272.93it/s][A
     47%|█████████████████▍                   | 2049/4360 [00:01<00:01, 1274.70it/s][A
     50%|██████████████████▍                  | 2177/4360 [00:01<00:01, 1275.98it/s][A
     53%|███████████████████▌                 | 2305/4360 [00:01<00:01, 1221.95it/s][A
     56%|████████████████████▋                | 2433/4360 [00:01<00:01, 1237.19it/s][A
     59%|█████████████████████▋               | 2561/4360 [00:02<00:01, 1249.17it/s][A
     62%|██████████████████████▊              | 2689/4360 [00:02<00:01, 1257.94it/s][A
     65%|███████████████████████▉             | 2817/4360 [00:02<00:01, 1262.97it/s][A
     68%|████████████████████████▉            | 2945/4360 [00:02<00:01, 1266.36it/s][A
     70%|██████████████████████████           | 3073/4360 [00:02<00:01, 1269.99it/s][A
     73%|███████████████████████████▏         | 3202/4360 [00:02<00:00, 1273.31it/s][A
     76%|████████████████████████████▎        | 3330/4360 [00:02<00:00, 1274.77it/s][A
     79%|█████████████████████████████▎       | 3458/4360 [00:02<00:00, 1275.45it/s][A
     82%|██████████████████████████████▍      | 3586/4360 [00:02<00:00, 1276.06it/s][A
     85%|███████████████████████████████▌     | 3714/4360 [00:02<00:00, 1276.59it/s][A
     88%|████████████████████████████████▌    | 3843/4360 [00:03<00:00, 1278.01it/s][A
     91%|█████████████████████████████████▋   | 3971/4360 [00:03<00:00, 1278.53it/s][A
     94%|██████████████████████████████████▊  | 4100/4360 [00:03<00:00, 1279.04it/s][A
     97%|███████████████████████████████████▉ | 4228/4360 [00:03<00:00, 1279.29it/s][A
    100%|█████████████████████████████████████| 4360/4360 [00:03<00:00, 1270.24it/s][A
    Testing DataLoader 0:  10%|██▏                   | 3/31 [00:42<06:32,  0.07it/s]Sampling. The number of nodes to sample is 5180.
    Sampling on one graph took 10.44511079788208 seconds.
    
      0%|                                                  | 0/5180 [00:00<?, ?it/s][A
      2%|▊                                     | 109/5180 [00:00<00:04, 1086.72it/s][A
      4%|█▌                                    | 218/5180 [00:00<00:04, 1087.81it/s][A
      6%|██▍                                   | 328/5180 [00:00<00:04, 1089.43it/s][A
      8%|███▏                                  | 438/5180 [00:00<00:04, 1090.18it/s][A
     11%|████                                  | 548/5180 [00:00<00:04, 1091.57it/s][A
     13%|████▊                                 | 658/5180 [00:00<00:04, 1090.68it/s][A
     15%|█████▋                                | 769/5180 [00:00<00:04, 1094.18it/s][A
     17%|██████▍                               | 880/5180 [00:00<00:03, 1096.98it/s][A
     19%|███████▎                              | 991/5180 [00:00<00:03, 1098.27it/s][A
     21%|███████▊                             | 1102/5180 [00:01<00:03, 1099.27it/s][A
     23%|████████▋                            | 1212/5180 [00:01<00:03, 1098.24it/s][A
     26%|█████████▍                           | 1322/5180 [00:01<00:03, 1098.18it/s][A
     28%|██████████▏                          | 1432/5180 [00:01<00:03, 1098.48it/s][A
     30%|███████████                          | 1543/5180 [00:01<00:03, 1099.04it/s][A
     32%|███████████▊                         | 1653/5180 [00:01<00:03, 1098.23it/s][A
     34%|████████████▌                        | 1763/5180 [00:01<00:03, 1097.76it/s][A
     36%|█████████████▍                       | 1873/5180 [00:01<00:03, 1095.91it/s][A
     38%|██████████████▏                      | 1983/5180 [00:01<00:02, 1091.07it/s][A
     40%|██████████████▉                      | 2093/5180 [00:01<00:02, 1092.77it/s][A
     43%|███████████████▋                     | 2204/5180 [00:02<00:02, 1095.07it/s][A
     45%|████████████████▌                    | 2315/5180 [00:02<00:02, 1096.78it/s][A
     47%|█████████████████▎                   | 2426/5180 [00:02<00:02, 1098.22it/s][A
     49%|██████████████████                   | 2537/5180 [00:02<00:02, 1099.46it/s][A
     51%|██████████████████▉                  | 2647/5180 [00:02<00:02, 1099.20it/s][A
     53%|███████████████████▋                 | 2757/5180 [00:02<00:02, 1053.69it/s][A
     55%|████████████████████▍                | 2867/5180 [00:02<00:02, 1065.50it/s][A
     57%|█████████████████████▎               | 2977/5180 [00:02<00:02, 1075.29it/s][A
     60%|██████████████████████               | 3085/5180 [00:02<00:01, 1074.90it/s][A
     62%|██████████████████████▊              | 3196/5180 [00:02<00:01, 1082.49it/s][A
     64%|███████████████████████▌             | 3306/5180 [00:03<00:01, 1087.62it/s][A
     66%|████████████████████████▍            | 3416/5180 [00:03<00:01, 1090.95it/s][A
     68%|█████████████████████████▏           | 3527/5180 [00:03<00:01, 1093.86it/s][A
     70%|█████████████████████████▉           | 3638/5180 [00:03<00:01, 1096.77it/s][A
     72%|██████████████████████████▊          | 3748/5180 [00:03<00:01, 1094.88it/s][A
     74%|███████████████████████████▌         | 3858/5180 [00:03<00:01, 1096.37it/s][A
     77%|████████████████████████████▎        | 3969/5180 [00:03<00:01, 1097.58it/s][A
     79%|█████████████████████████████▏       | 4080/5180 [00:03<00:01, 1098.60it/s][A
     81%|█████████████████████████████▉       | 4191/5180 [00:03<00:00, 1099.20it/s][A
     83%|██████████████████████████████▋      | 4302/5180 [00:03<00:00, 1100.21it/s][A
     85%|███████████████████████████████▌     | 4413/5180 [00:04<00:00, 1100.93it/s][A
     87%|████████████████████████████████▎    | 4524/5180 [00:04<00:00, 1097.88it/s][A
     89%|█████████████████████████████████    | 4634/5180 [00:04<00:00, 1097.88it/s][A
     92%|█████████████████████████████████▉   | 4744/5180 [00:04<00:00, 1098.09it/s][A
     94%|██████████████████████████████████▋  | 4854/5180 [00:04<00:00, 1098.52it/s][A
     96%|███████████████████████████████████▍ | 4964/5180 [00:04<00:00, 1097.85it/s][A
    100%|█████████████████████████████████████| 5180/5180 [00:04<00:00, 1093.44it/s][A
    Testing DataLoader 0:  13%|██▊                   | 4/31 [01:03<07:11,  0.06it/s]Sampling. The number of nodes to sample is 4946.
    Sampling on one graph took 10.20914888381958 seconds.
    
      0%|                                                  | 0/4946 [00:00<?, ?it/s][A
      2%|▉                                     | 114/4946 [00:00<00:04, 1132.15it/s][A
      5%|█▊                                    | 228/4946 [00:00<00:04, 1134.93it/s][A
      7%|██▋                                   | 342/4946 [00:00<00:04, 1136.00it/s][A
      9%|███▌                                  | 456/4946 [00:00<00:03, 1135.82it/s][A
     12%|████▍                                 | 571/4946 [00:00<00:03, 1139.30it/s][A
     14%|█████▎                                | 686/4946 [00:00<00:03, 1140.36it/s][A
     16%|██████▏                               | 801/4946 [00:00<00:03, 1142.26it/s][A
     19%|███████                               | 916/4946 [00:00<00:03, 1142.79it/s][A
     21%|███████▋                             | 1031/4946 [00:00<00:03, 1143.49it/s][A
     23%|████████▌                            | 1146/4946 [00:01<00:03, 1142.87it/s][A
     25%|█████████▍                           | 1261/4946 [00:01<00:03, 1143.83it/s][A
     28%|██████████▎                          | 1376/4946 [00:01<00:03, 1144.05it/s][A
     30%|███████████▏                         | 1491/4946 [00:01<00:03, 1144.82it/s][A
     32%|████████████                         | 1606/4946 [00:01<00:02, 1145.06it/s][A
     35%|████████████▊                        | 1721/4946 [00:01<00:02, 1145.23it/s][A
     37%|█████████████▋                       | 1836/4946 [00:01<00:02, 1145.57it/s][A
     39%|██████████████▌                      | 1951/4946 [00:01<00:02, 1145.17it/s][A
     42%|███████████████▍                     | 2066/4946 [00:01<00:02, 1135.93it/s][A
     44%|████████████████▎                    | 2181/4946 [00:01<00:02, 1138.04it/s][A
     46%|█████████████████▏                   | 2296/4946 [00:02<00:02, 1140.18it/s][A
     49%|██████████████████                   | 2411/4946 [00:02<00:02, 1141.28it/s][A
     51%|██████████████████▉                  | 2526/4946 [00:02<00:02, 1142.52it/s][A
     53%|███████████████████▊                 | 2641/4946 [00:02<00:02, 1142.71it/s][A
     56%|████████████████████▌                | 2756/4946 [00:02<00:01, 1143.54it/s][A
     58%|█████████████████████▍               | 2871/4946 [00:02<00:01, 1143.50it/s][A
     60%|██████████████████████▎              | 2986/4946 [00:02<00:01, 1143.97it/s][A
     63%|███████████████████████▏             | 3101/4946 [00:02<00:01, 1144.75it/s][A
     65%|████████████████████████             | 3216/4946 [00:02<00:01, 1143.99it/s][A
     67%|████████████████████████▉            | 3331/4946 [00:02<00:01, 1144.35it/s][A
     70%|█████████████████████████▊           | 3446/4946 [00:03<00:01, 1145.25it/s][A
     72%|██████████████████████████▋          | 3561/4946 [00:03<00:01, 1146.07it/s][A
     74%|███████████████████████████▍         | 3676/4946 [00:03<00:01, 1145.74it/s][A
     77%|████████████████████████████▎        | 3791/4946 [00:03<00:01, 1145.11it/s][A
     79%|█████████████████████████████▏       | 3906/4946 [00:03<00:00, 1104.79it/s][A
     81%|██████████████████████████████       | 4020/4946 [00:03<00:00, 1112.95it/s][A
     84%|██████████████████████████████▉      | 4134/4946 [00:03<00:00, 1118.72it/s][A
     86%|███████████████████████████████▊     | 4249/4946 [00:03<00:00, 1127.00it/s][A
     88%|████████████████████████████████▋    | 4365/4946 [00:03<00:00, 1134.04it/s][A
     91%|█████████████████████████████████▌   | 4481/4946 [00:03<00:00, 1139.46it/s][A
     93%|██████████████████████████████████▍  | 4597/4946 [00:04<00:00, 1143.86it/s][A
     95%|███████████████████████████████████▏ | 4712/4946 [00:04<00:00, 1139.92it/s][A
     98%|████████████████████████████████████ | 4827/4946 [00:04<00:00, 1140.89it/s][A
    100%|█████████████████████████████████████| 4946/4946 [00:04<00:00, 1139.79it/s][A
    Testing DataLoader 0:  16%|███▌                  | 5/31 [01:24<07:18,  0.06it/s]Sampling. The number of nodes to sample is 2037.
    Sampling on one graph took 8.390247344970703 seconds.
    
      0%|                                                  | 0/2037 [00:00<?, ?it/s][A
     11%|████                                  | 221/2037 [00:00<00:00, 2207.96it/s][A
     22%|████████▎                             | 444/2037 [00:00<00:00, 2218.00it/s][A
     33%|████████████▍                         | 668/2037 [00:00<00:00, 2223.95it/s][A
     44%|████████████████▋                     | 892/2037 [00:00<00:00, 2226.89it/s][A
     55%|████████████████████▎                | 1115/2037 [00:00<00:00, 2225.54it/s][A
     66%|████████████████████████▎            | 1339/2037 [00:00<00:00, 2228.88it/s][A
     77%|████████████████████████████▍        | 1563/2037 [00:00<00:00, 2229.40it/s][A
     88%|████████████████████████████████▍    | 1787/2037 [00:00<00:00, 2230.90it/s][A
    100%|█████████████████████████████████████| 2037/2037 [00:00<00:00, 2191.37it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  19%|████▎                 | 6/31 [01:34<06:35,  0.06it/s]Sampling. The number of nodes to sample is 5050.
    Sampling on one graph took 10.375575542449951 seconds.
    
      0%|                                                  | 0/5050 [00:00<?, ?it/s][A
      2%|▊                                     | 112/5050 [00:00<00:04, 1110.32it/s][A
      4%|█▋                                    | 224/5050 [00:00<00:04, 1115.43it/s][A
      7%|██▌                                   | 337/5050 [00:00<00:04, 1119.61it/s][A
      9%|███▍                                  | 450/5050 [00:00<00:04, 1122.18it/s][A
     11%|████▏                                 | 563/5050 [00:00<00:03, 1121.93it/s][A
     13%|█████                                 | 676/5050 [00:00<00:03, 1123.14it/s][A
     16%|█████▉                                | 789/5050 [00:00<00:03, 1124.43it/s][A
     18%|██████▊                               | 902/5050 [00:00<00:03, 1124.97it/s][A
     20%|███████▍                             | 1015/5050 [00:00<00:03, 1074.37it/s][A
     22%|████████▎                            | 1128/5050 [00:01<00:03, 1088.58it/s][A
     25%|█████████                            | 1241/5050 [00:01<00:03, 1099.91it/s][A
     27%|█████████▉                           | 1354/5050 [00:01<00:03, 1107.52it/s][A
     29%|██████████▋                          | 1465/5050 [00:01<00:03, 1106.56it/s][A
     31%|███████████▌                         | 1578/5050 [00:01<00:03, 1110.86it/s][A
     33%|████████████▍                        | 1691/5050 [00:01<00:03, 1115.22it/s][A
     36%|█████████████▏                       | 1804/5050 [00:01<00:02, 1117.83it/s][A
     38%|██████████████                       | 1917/5050 [00:01<00:02, 1119.25it/s][A
     40%|██████████████▊                      | 2030/5050 [00:01<00:02, 1122.32it/s][A
     42%|███████████████▋                     | 2143/5050 [00:01<00:02, 1122.67it/s][A
     45%|████████████████▌                    | 2256/5050 [00:02<00:02, 1123.17it/s][A
     47%|█████████████████▎                   | 2369/5050 [00:02<00:02, 1123.74it/s][A
     49%|██████████████████▏                  | 2482/5050 [00:02<00:02, 1123.73it/s][A
     51%|███████████████████                  | 2595/5050 [00:02<00:02, 1124.02it/s][A
     54%|███████████████████▊                 | 2708/5050 [00:02<00:02, 1076.21it/s][A
     56%|████████████████████▋                | 2821/5050 [00:02<00:02, 1089.96it/s][A
     58%|█████████████████████▍               | 2934/5050 [00:02<00:01, 1100.25it/s][A
     60%|██████████████████████▎              | 3047/5050 [00:02<00:01, 1107.19it/s][A
     63%|███████████████████████▏             | 3160/5050 [00:02<00:01, 1113.32it/s][A
     65%|███████████████████████▉             | 3272/5050 [00:02<00:01, 1110.29it/s][A
     67%|████████████████████████▊            | 3384/5050 [00:03<00:01, 1106.02it/s][A
     69%|█████████████████████████▌           | 3497/5050 [00:03<00:01, 1110.89it/s][A
     71%|██████████████████████████▍          | 3610/5050 [00:03<00:01, 1113.90it/s][A
     74%|███████████████████████████▎         | 3723/5050 [00:03<00:01, 1118.10it/s][A
     76%|████████████████████████████         | 3836/5050 [00:03<00:01, 1119.91it/s][A
     78%|████████████████████████████▉        | 3949/5050 [00:03<00:01, 1081.32it/s][A
     80%|█████████████████████████████▋       | 4058/5050 [00:03<00:00, 1040.48it/s][A
     83%|██████████████████████████████▌      | 4170/5050 [00:03<00:00, 1062.77it/s][A
     85%|███████████████████████████████▍     | 4283/5050 [00:03<00:00, 1080.32it/s][A
     87%|████████████████████████████████▏    | 4396/5050 [00:03<00:00, 1092.49it/s][A
     89%|█████████████████████████████████    | 4507/5050 [00:04<00:00, 1095.60it/s][A
     91%|█████████████████████████████████▊   | 4620/5050 [00:04<00:00, 1103.99it/s][A
     94%|██████████████████████████████████▋  | 4731/5050 [00:04<00:00, 1104.92it/s][A
     96%|███████████████████████████████████▍ | 4845/5050 [00:04<00:00, 1114.26it/s][A
    100%|█████████████████████████████████████| 5050/5050 [00:04<00:00, 1107.32it/s][A
    Testing DataLoader 0:  23%|████▉                 | 7/31 [01:56<06:38,  0.06it/s]Sampling. The number of nodes to sample is 5047.
    Sampling on one graph took 10.428575038909912 seconds.
    
      0%|                                                  | 0/5047 [00:00<?, ?it/s][A
      2%|▊                                     | 111/5047 [00:00<00:04, 1108.07it/s][A
      4%|█▋                                    | 223/5047 [00:00<00:04, 1114.99it/s][A
      7%|██▌                                   | 335/5047 [00:00<00:04, 1113.40it/s][A
      9%|███▎                                  | 447/5047 [00:00<00:04, 1114.36it/s][A
     11%|████▏                                 | 559/5047 [00:00<00:04, 1111.86it/s][A
     13%|█████                                 | 671/5047 [00:00<00:03, 1114.28it/s][A
     16%|█████▉                                | 783/5047 [00:00<00:03, 1067.43it/s][A
     18%|██████▋                               | 896/5047 [00:00<00:03, 1084.20it/s][A
     20%|███████▍                             | 1008/5047 [00:00<00:03, 1094.04it/s][A
     22%|████████▏                            | 1120/5047 [00:01<00:03, 1101.33it/s][A
     24%|█████████                            | 1232/5047 [00:01<00:03, 1106.94it/s][A
     27%|█████████▊                           | 1344/5047 [00:01<00:03, 1110.18it/s][A
     29%|██████████▋                          | 1456/5047 [00:01<00:03, 1112.48it/s][A
     31%|███████████▍                         | 1568/5047 [00:01<00:03, 1114.46it/s][A
     33%|████████████▎                        | 1681/5047 [00:01<00:03, 1116.15it/s][A
     36%|█████████████▏                       | 1793/5047 [00:01<00:02, 1116.57it/s][A
     38%|█████████████▉                       | 1905/5047 [00:01<00:02, 1117.35it/s][A
     40%|██████████████▊                      | 2017/5047 [00:01<00:02, 1116.32it/s][A
     42%|███████████████▌                     | 2129/5047 [00:01<00:02, 1081.86it/s][A
     44%|████████████████▍                    | 2238/5047 [00:02<00:02, 1036.15it/s][A
     47%|█████████████████▏                   | 2350/5047 [00:02<00:02, 1058.93it/s][A
     49%|██████████████████                   | 2462/5047 [00:02<00:02, 1075.73it/s][A
     51%|██████████████████▊                  | 2574/5047 [00:02<00:02, 1088.12it/s][A
     53%|███████████████████▋                 | 2686/5047 [00:02<00:02, 1096.63it/s][A
     55%|████████████████████▌                | 2798/5047 [00:02<00:02, 1102.65it/s][A
     58%|█████████████████████▎               | 2911/5047 [00:02<00:01, 1108.23it/s][A
     60%|██████████████████████▏              | 3023/5047 [00:02<00:01, 1110.98it/s][A
     62%|██████████████████████▉              | 3135/5047 [00:02<00:01, 1113.13it/s][A
     64%|███████████████████████▊             | 3247/5047 [00:02<00:01, 1114.30it/s][A
     67%|████████████████████████▋            | 3359/5047 [00:03<00:01, 1113.85it/s][A
     69%|█████████████████████████▍           | 3471/5047 [00:03<00:01, 1114.51it/s][A
     71%|██████████████████████████▎          | 3583/5047 [00:03<00:01, 1067.95it/s][A
     73%|███████████████████████████          | 3696/5047 [00:03<00:01, 1083.40it/s][A
     75%|███████████████████████████▉         | 3809/5047 [00:03<00:01, 1094.31it/s][A
     78%|████████████████████████████▋        | 3921/5047 [00:03<00:01, 1101.62it/s][A
     80%|█████████████████████████████▌       | 4034/5047 [00:03<00:00, 1108.11it/s][A
     82%|██████████████████████████████▍      | 4146/5047 [00:03<00:00, 1111.07it/s][A
     84%|███████████████████████████████▏     | 4259/5047 [00:03<00:00, 1113.86it/s][A
     87%|████████████████████████████████     | 4371/5047 [00:03<00:00, 1114.10it/s][A
     89%|████████████████████████████████▊    | 4483/5047 [00:04<00:00, 1114.55it/s][A
     91%|█████████████████████████████████▋   | 4595/5047 [00:04<00:00, 1114.91it/s][A
     93%|██████████████████████████████████▌  | 4707/5047 [00:04<00:00, 1115.25it/s][A
     95%|███████████████████████████████████▎ | 4819/5047 [00:04<00:00, 1115.43it/s][A
     98%|████████████████████████████████████▏| 4932/5047 [00:04<00:00, 1117.28it/s][A
    100%|█████████████████████████████████████| 5047/5047 [00:04<00:00, 1099.60it/s][A
    Testing DataLoader 0:  26%|█████▋                | 8/31 [02:17<06:35,  0.06it/s]Sampling. The number of nodes to sample is 5235.
    Sampling on one graph took 10.600151300430298 seconds.
    
      0%|                                                  | 0/5235 [00:00<?, ?it/s][A
      2%|▊                                     | 109/5235 [00:00<00:04, 1088.86it/s][A
      4%|█▌                                    | 219/5235 [00:00<00:04, 1094.18it/s][A
      6%|██▍                                   | 329/5235 [00:00<00:04, 1096.13it/s][A
      8%|███▏                                  | 439/5235 [00:00<00:04, 1097.30it/s][A
     10%|███▉                                  | 549/5235 [00:00<00:04, 1097.43it/s][A
     13%|████▊                                 | 659/5235 [00:00<00:04, 1097.90it/s][A
     15%|█████▌                                | 769/5235 [00:00<00:04, 1096.20it/s][A
     17%|██████▍                               | 879/5235 [00:00<00:03, 1096.72it/s][A
     19%|███████▏                              | 989/5235 [00:00<00:03, 1097.41it/s][A
     21%|███████▊                             | 1099/5235 [00:01<00:03, 1097.01it/s][A
     23%|████████▌                            | 1209/5235 [00:01<00:03, 1097.21it/s][A
     25%|█████████▎                           | 1319/5235 [00:01<00:03, 1097.83it/s][A
     27%|██████████                           | 1429/5235 [00:01<00:03, 1097.13it/s][A
     29%|██████████▉                          | 1539/5235 [00:01<00:03, 1097.26it/s][A
     31%|███████████▋                         | 1649/5235 [00:01<00:03, 1097.83it/s][A
     34%|████████████▍                        | 1759/5235 [00:01<00:03, 1097.92it/s][A
     36%|█████████████▏                       | 1870/5235 [00:01<00:03, 1098.85it/s][A
     38%|█████████████▉                       | 1980/5235 [00:01<00:02, 1097.91it/s][A
     40%|██████████████▊                      | 2090/5235 [00:01<00:02, 1097.61it/s][A
     42%|███████████████▌                     | 2200/5235 [00:02<00:02, 1098.00it/s][A
     44%|████████████████▎                    | 2310/5235 [00:02<00:02, 1098.34it/s][A
     46%|█████████████████                    | 2420/5235 [00:02<00:02, 1098.16it/s][A
     48%|█████████████████▉                   | 2530/5235 [00:02<00:02, 1047.83it/s][A
     50%|██████████████████▋                  | 2640/5235 [00:02<00:02, 1062.38it/s][A
     53%|███████████████████▍                 | 2750/5235 [00:02<00:02, 1072.53it/s][A
     55%|████████████████████▏                | 2860/5235 [00:02<00:02, 1080.23it/s][A
     57%|████████████████████▉                | 2971/5235 [00:02<00:02, 1086.64it/s][A
     59%|█████████████████████▊               | 3081/5235 [00:02<00:01, 1089.00it/s][A
     61%|██████████████████████▌              | 3191/5235 [00:02<00:01, 1091.12it/s][A
     63%|███████████████████████▎             | 3301/5235 [00:03<00:01, 1088.84it/s][A
     65%|████████████████████████             | 3411/5235 [00:03<00:01, 1089.95it/s][A
     67%|████████████████████████▉            | 3521/5235 [00:03<00:01, 1092.70it/s][A
     69%|█████████████████████████▋           | 3631/5235 [00:03<00:01, 1094.11it/s][A
     71%|██████████████████████████▍          | 3741/5235 [00:03<00:01, 1095.80it/s][A
     74%|███████████████████████████▏         | 3851/5235 [00:03<00:01, 1096.01it/s][A
     76%|████████████████████████████         | 3962/5235 [00:03<00:01, 1097.97it/s][A
     78%|████████████████████████████▊        | 4072/5235 [00:03<00:01, 1098.48it/s][A
     80%|█████████████████████████████▌       | 4182/5235 [00:03<00:00, 1097.82it/s][A
     82%|██████████████████████████████▎      | 4292/5235 [00:03<00:00, 1098.38it/s][A
     84%|███████████████████████████████      | 4402/5235 [00:04<00:00, 1098.47it/s][A
     86%|███████████████████████████████▉     | 4512/5235 [00:04<00:00, 1098.30it/s][A
     88%|████████████████████████████████▋    | 4622/5235 [00:04<00:00, 1097.65it/s][A
     90%|█████████████████████████████████▍   | 4733/5235 [00:04<00:00, 1098.69it/s][A
     93%|██████████████████████████████████▏  | 4844/5235 [00:04<00:00, 1099.27it/s][A
     95%|███████████████████████████████████  | 4955/5235 [00:04<00:00, 1099.66it/s][A
     97%|███████████████████████████████████▊ | 5065/5235 [00:04<00:00, 1099.54it/s][A
    100%|█████████████████████████████████████| 5235/5235 [00:04<00:00, 1093.99it/s][A
    Testing DataLoader 0:  29%|██████▍               | 9/31 [02:39<06:30,  0.06it/s]Sampling. The number of nodes to sample is 3870.
    Sampling on one graph took 8.766932487487793 seconds.
    
      0%|                                                  | 0/3870 [00:00<?, ?it/s][A
      4%|█▎                                    | 137/3870 [00:00<00:02, 1368.39it/s][A
      7%|██▋                                   | 279/3870 [00:00<00:02, 1393.32it/s][A
     11%|████▏                                 | 421/3870 [00:00<00:02, 1405.31it/s][A
     15%|█████▌                                | 564/3870 [00:00<00:02, 1411.52it/s][A
     18%|██████▉                               | 707/3870 [00:00<00:02, 1415.01it/s][A
     22%|████████▎                             | 849/3870 [00:00<00:02, 1348.65it/s][A
     26%|█████████▋                            | 991/3870 [00:00<00:02, 1369.96it/s][A
     29%|██████████▊                          | 1133/3870 [00:00<00:01, 1382.95it/s][A
     33%|████████████▏                        | 1274/3870 [00:00<00:01, 1391.03it/s][A
     37%|█████████████▌                       | 1416/3870 [00:01<00:01, 1398.28it/s][A
     40%|██████████████▉                      | 1558/3870 [00:01<00:01, 1402.74it/s][A
     44%|████████████████▎                    | 1700/3870 [00:01<00:01, 1406.41it/s][A
     48%|█████████████████▌                   | 1842/3870 [00:01<00:01, 1407.98it/s][A
     51%|██████████████████▉                  | 1983/3870 [00:01<00:01, 1399.96it/s][A
     55%|████████████████████▎                | 2125/3870 [00:01<00:01, 1404.57it/s][A
     59%|█████████████████████▋               | 2267/3870 [00:01<00:01, 1407.49it/s][A
     62%|███████████████████████              | 2409/3870 [00:01<00:01, 1409.21it/s][A
     66%|████████████████████████▍            | 2550/3870 [00:01<00:00, 1359.68it/s][A
     69%|█████████████████████████▋           | 2689/3870 [00:01<00:00, 1366.84it/s][A
     73%|███████████████████████████          | 2831/3870 [00:02<00:00, 1380.55it/s][A
     77%|████████████████████████████▍        | 2973/3870 [00:02<00:00, 1390.79it/s][A
     80%|█████████████████████████████▊       | 3115/3870 [00:02<00:00, 1398.17it/s][A
     84%|███████████████████████████████▏     | 3257/3870 [00:02<00:00, 1403.01it/s][A
     88%|████████████████████████████████▍    | 3399/3870 [00:02<00:00, 1405.58it/s][A
     91%|█████████████████████████████████▊   | 3541/3870 [00:02<00:00, 1408.18it/s][A
     95%|███████████████████████████████████▏ | 3683/3870 [00:02<00:00, 1411.20it/s][A
    100%|█████████████████████████████████████| 3870/3870 [00:02<00:00, 1396.94it/s][A
    Testing DataLoader 0:  32%|██████▊              | 10/31 [02:55<06:07,  0.06it/s]Sampling. The number of nodes to sample is 4317.
    Sampling on one graph took 9.598732948303223 seconds.
    
      0%|                                                  | 0/4317 [00:00<?, ?it/s][A
      3%|█▏                                    | 130/4317 [00:00<00:03, 1294.78it/s][A
      6%|██▎                                   | 260/4317 [00:00<00:03, 1197.39it/s][A
      9%|███▍                                  | 390/4317 [00:00<00:03, 1238.83it/s][A
     12%|████▌                                 | 520/4317 [00:00<00:03, 1261.51it/s][A
     15%|█████▋                                | 650/4317 [00:00<00:02, 1274.53it/s][A
     18%|██████▊                               | 780/4317 [00:00<00:02, 1282.68it/s][A
     21%|████████                              | 909/4317 [00:00<00:02, 1284.55it/s][A
     24%|████████▉                            | 1039/4317 [00:00<00:02, 1288.45it/s][A
     27%|██████████                           | 1169/4317 [00:00<00:02, 1292.02it/s][A
     30%|███████████▏                         | 1299/4317 [00:01<00:02, 1294.38it/s][A
     33%|████████████▏                        | 1429/4317 [00:01<00:02, 1294.92it/s][A
     36%|█████████████▎                       | 1559/4317 [00:01<00:02, 1292.66it/s][A
     39%|██████████████▍                      | 1689/4317 [00:01<00:02, 1293.56it/s][A
     42%|███████████████▌                     | 1820/4317 [00:01<00:01, 1296.01it/s][A
     45%|████████████████▋                    | 1950/4317 [00:01<00:01, 1297.18it/s][A
     48%|█████████████████▊                   | 2080/4317 [00:01<00:01, 1246.03it/s][A
     51%|██████████████████▉                  | 2210/4317 [00:01<00:01, 1261.18it/s][A
     54%|████████████████████                 | 2340/4317 [00:01<00:01, 1272.26it/s][A
     57%|█████████████████████▏               | 2470/4317 [00:01<00:01, 1279.15it/s][A
     60%|██████████████████████▎              | 2600/4317 [00:02<00:01, 1284.29it/s][A
     63%|███████████████████████▍             | 2731/4317 [00:02<00:01, 1290.20it/s][A
     66%|████████████████████████▌            | 2861/4317 [00:02<00:01, 1292.23it/s][A
     69%|█████████████████████████▋           | 2991/4317 [00:02<00:01, 1293.52it/s][A
     72%|██████████████████████████▋          | 3121/4317 [00:02<00:00, 1295.21it/s][A
     75%|███████████████████████████▊         | 3251/4317 [00:02<00:00, 1294.90it/s][A
     78%|████████████████████████████▉        | 3381/4317 [00:02<00:00, 1295.09it/s][A
     81%|██████████████████████████████       | 3511/4317 [00:02<00:00, 1295.98it/s][A
     84%|███████████████████████████████▏     | 3641/4317 [00:02<00:00, 1296.73it/s][A
     87%|████████████████████████████████▎    | 3771/4317 [00:02<00:00, 1241.60it/s][A
     90%|█████████████████████████████████▍   | 3902/4317 [00:03<00:00, 1258.98it/s][A
     93%|██████████████████████████████████▌  | 4032/4317 [00:03<00:00, 1270.62it/s][A
     96%|███████████████████████████████████▋ | 4162/4317 [00:03<00:00, 1278.94it/s][A
    100%|█████████████████████████████████████| 4317/4317 [00:03<00:00, 1280.67it/s][A
    Testing DataLoader 0:  35%|███████▍             | 11/31 [03:12<05:50,  0.06it/s]Sampling. The number of nodes to sample is 4081.
    Sampling on one graph took 8.924293994903564 seconds.
    
      0%|                                                  | 0/4081 [00:00<?, ?it/s][A
      3%|█▎                                    | 135/4081 [00:00<00:02, 1345.37it/s][A
      7%|██▌                                   | 271/4081 [00:00<00:02, 1352.80it/s][A
     10%|███▊                                  | 407/4081 [00:00<00:02, 1354.55it/s][A
     13%|█████                                 | 543/4081 [00:00<00:02, 1355.13it/s][A
     17%|██████▎                               | 680/4081 [00:00<00:02, 1357.75it/s][A
     20%|███████▌                              | 816/4081 [00:00<00:02, 1357.20it/s][A
     23%|████████▊                             | 952/4081 [00:00<00:02, 1357.86it/s][A
     27%|█████████▊                           | 1089/4081 [00:00<00:02, 1358.58it/s][A
     30%|███████████                          | 1225/4081 [00:00<00:02, 1358.10it/s][A
     33%|████████████▎                        | 1361/4081 [00:01<00:02, 1357.58it/s][A
     37%|█████████████▌                       | 1498/4081 [00:01<00:01, 1358.44it/s][A
     40%|██████████████▊                      | 1634/4081 [00:01<00:01, 1356.67it/s][A
     43%|████████████████                     | 1770/4081 [00:01<00:01, 1357.66it/s][A
     47%|█████████████████▎                   | 1906/4081 [00:01<00:01, 1356.93it/s][A
     50%|██████████████████▌                  | 2042/4081 [00:01<00:01, 1357.72it/s][A
     53%|███████████████████▋                 | 2178/4081 [00:01<00:01, 1358.25it/s][A
     57%|████████████████████▉                | 2314/4081 [00:01<00:01, 1252.52it/s][A
     60%|██████████████████████▏              | 2446/4081 [00:01<00:01, 1270.51it/s][A
     63%|███████████████████████▍             | 2582/4081 [00:01<00:01, 1295.94it/s][A
     67%|████████████████████████▋            | 2718/4081 [00:02<00:01, 1314.35it/s][A
     70%|█████████████████████████▉           | 2855/4081 [00:02<00:00, 1328.35it/s][A
     73%|███████████████████████████          | 2991/4081 [00:02<00:00, 1336.28it/s][A
     77%|████████████████████████████▎        | 3127/4081 [00:02<00:00, 1343.11it/s][A
     80%|█████████████████████████████▌       | 3264/4081 [00:02<00:00, 1348.31it/s][A
     83%|██████████████████████████████▊      | 3401/4081 [00:02<00:00, 1352.22it/s][A
     87%|████████████████████████████████     | 3537/4081 [00:02<00:00, 1354.51it/s][A
     90%|█████████████████████████████████▎   | 3673/4081 [00:02<00:00, 1347.70it/s][A
     93%|██████████████████████████████████▌  | 3809/4081 [00:02<00:00, 1350.16it/s][A
    100%|█████████████████████████████████████| 4081/4081 [00:03<00:00, 1337.05it/s][A
    Testing DataLoader 0:  39%|████████▏            | 12/31 [03:28<05:30,  0.06it/s]Sampling. The number of nodes to sample is 4052.
    Sampling on one graph took 8.861066102981567 seconds.
    
      0%|                                                  | 0/4052 [00:00<?, ?it/s][A
      3%|█▎                                    | 136/4052 [00:00<00:02, 1358.54it/s][A
      7%|██▌                                   | 273/4052 [00:00<00:02, 1363.24it/s][A
     10%|███▊                                  | 410/4052 [00:00<00:02, 1365.09it/s][A
     13%|█████▏                                | 547/4052 [00:00<00:02, 1366.35it/s][A
     17%|██████▍                               | 684/4052 [00:00<00:02, 1365.37it/s][A
     20%|███████▋                              | 821/4052 [00:00<00:02, 1365.85it/s][A
     24%|████████▉                             | 958/4052 [00:00<00:02, 1365.70it/s][A
     27%|█████████▉                           | 1095/4052 [00:00<00:02, 1366.83it/s][A
     30%|███████████▏                         | 1232/4052 [00:00<00:02, 1363.83it/s][A
     34%|████████████▌                        | 1369/4052 [00:01<00:01, 1364.05it/s][A
     37%|█████████████▊                       | 1506/4052 [00:01<00:01, 1364.29it/s][A
     41%|███████████████                      | 1643/4052 [00:01<00:01, 1364.79it/s][A
     44%|████████████████▎                    | 1780/4052 [00:01<00:01, 1365.27it/s][A
     47%|█████████████████▌                   | 1917/4052 [00:01<00:01, 1365.82it/s][A
     51%|██████████████████▊                  | 2054/4052 [00:01<00:01, 1366.70it/s][A
     54%|████████████████████                 | 2191/4052 [00:01<00:01, 1367.42it/s][A
     57%|█████████████████████▎               | 2328/4052 [00:01<00:01, 1366.76it/s][A
     61%|██████████████████████▌              | 2465/4052 [00:01<00:01, 1367.36it/s][A
     64%|███████████████████████▊             | 2602/4052 [00:01<00:01, 1367.88it/s][A
     68%|█████████████████████████            | 2739/4052 [00:02<00:00, 1368.08it/s][A
     71%|██████████████████████████▎          | 2876/4052 [00:02<00:00, 1367.50it/s][A
     74%|███████████████████████████▌         | 3013/4052 [00:02<00:00, 1365.36it/s][A
     78%|████████████████████████████▊        | 3151/4052 [00:02<00:00, 1367.16it/s][A
     81%|██████████████████████████████       | 3288/4052 [00:02<00:00, 1365.87it/s][A
     85%|███████████████████████████████▎     | 3425/4052 [00:02<00:00, 1365.90it/s][A
     88%|████████████████████████████████▌    | 3562/4052 [00:02<00:00, 1365.33it/s][A
     91%|█████████████████████████████████▊   | 3700/4052 [00:02<00:00, 1367.30it/s][A
     95%|███████████████████████████████████  | 3838/4052 [00:02<00:00, 1369.60it/s][A
    100%|█████████████████████████████████████| 4052/4052 [00:02<00:00, 1366.77it/s][A
    Testing DataLoader 0:  42%|████████▊            | 13/31 [03:44<05:11,  0.06it/s]Sampling. The number of nodes to sample is 3564.
    Sampling on one graph took 8.6836416721344 seconds.
    
      0%|                                                  | 0/3564 [00:00<?, ?it/s][A
      4%|█▌                                    | 148/3564 [00:00<00:02, 1475.62it/s][A
      8%|███▏                                  | 299/3564 [00:00<00:02, 1492.88it/s][A
     13%|████▊                                 | 450/3564 [00:00<00:02, 1500.00it/s][A
     17%|██████▍                               | 601/3564 [00:00<00:01, 1502.64it/s][A
     21%|████████                              | 752/3564 [00:00<00:01, 1503.87it/s][A
     25%|█████████▋                            | 903/3564 [00:00<00:01, 1505.09it/s][A
     30%|██████████▉                          | 1054/3564 [00:00<00:01, 1505.74it/s][A
     34%|████████████▌                        | 1205/3564 [00:00<00:01, 1505.37it/s][A
     38%|██████████████                       | 1356/3564 [00:00<00:01, 1503.73it/s][A
     42%|███████████████▋                     | 1507/3564 [00:01<00:01, 1503.26it/s][A
     47%|█████████████████▏                   | 1659/3564 [00:01<00:01, 1506.09it/s][A
     51%|██████████████████▊                  | 1810/3564 [00:01<00:01, 1433.04it/s][A
     55%|████████████████████▎                | 1955/3564 [00:01<00:01, 1392.30it/s][A
     59%|█████████████████████▊               | 2106/3564 [00:01<00:01, 1425.56it/s][A
     63%|███████████████████████▍             | 2257/3564 [00:01<00:00, 1450.10it/s][A
     68%|████████████████████████▉            | 2408/3564 [00:01<00:00, 1466.33it/s][A
     72%|██████████████████████████▌          | 2559/3564 [00:01<00:00, 1476.47it/s][A
     76%|████████████████████████████         | 2708/3564 [00:01<00:00, 1480.13it/s][A
     80%|█████████████████████████████▋       | 2859/3564 [00:01<00:00, 1488.00it/s][A
     84%|███████████████████████████████▏     | 3008/3564 [00:02<00:00, 1488.25it/s][A
     89%|████████████████████████████████▊    | 3159/3564 [00:02<00:00, 1493.16it/s][A
     93%|██████████████████████████████████▎  | 3311/3564 [00:02<00:00, 1499.07it/s][A
    100%|█████████████████████████████████████| 3564/3564 [00:02<00:00, 1483.69it/s][A
    Testing DataLoader 0:  45%|█████████▍           | 14/31 [03:59<04:50,  0.06it/s]Sampling. The number of nodes to sample is 4186.
    Sampling on one graph took 9.027669906616211 seconds.
    
      0%|                                                  | 0/4186 [00:00<?, ?it/s][A
      3%|█▏                                    | 132/4186 [00:00<00:03, 1313.56it/s][A
      6%|██▍                                   | 265/4186 [00:00<00:02, 1318.07it/s][A
     10%|███▌                                  | 398/4186 [00:00<00:02, 1321.55it/s][A
     13%|████▊                                 | 531/4186 [00:00<00:02, 1323.13it/s][A
     16%|██████                                | 664/4186 [00:00<00:02, 1323.03it/s][A
     19%|███████▏                              | 797/4186 [00:00<00:02, 1324.70it/s][A
     22%|████████▍                             | 930/4186 [00:00<00:02, 1325.96it/s][A
     25%|█████████▍                           | 1064/4186 [00:00<00:02, 1327.33it/s][A
     29%|██████████▌                          | 1197/4186 [00:00<00:02, 1271.72it/s][A
     32%|███████████▊                         | 1330/4186 [00:01<00:02, 1287.05it/s][A
     35%|████████████▉                        | 1463/4186 [00:01<00:02, 1299.45it/s][A
     38%|██████████████                       | 1596/4186 [00:01<00:01, 1308.44it/s][A
     41%|███████████████▎                     | 1730/4186 [00:01<00:01, 1316.06it/s][A
     45%|████████████████▍                    | 1863/4186 [00:01<00:01, 1318.57it/s][A
     48%|█████████████████▋                   | 1996/4186 [00:01<00:01, 1321.23it/s][A
     51%|██████████████████▊                  | 2129/4186 [00:01<00:01, 1322.41it/s][A
     54%|███████████████████▉                 | 2262/4186 [00:01<00:01, 1323.95it/s][A
     57%|█████████████████████▏               | 2396/4186 [00:01<00:01, 1326.05it/s][A
     60%|██████████████████████▎              | 2529/4186 [00:01<00:01, 1324.85it/s][A
     64%|███████████████████████▌             | 2662/4186 [00:02<00:01, 1324.47it/s][A
     67%|████████████████████████▋            | 2795/4186 [00:02<00:01, 1321.85it/s][A
     70%|█████████████████████████▉           | 2928/4186 [00:02<00:00, 1267.83it/s][A
     73%|███████████████████████████          | 3061/4186 [00:02<00:00, 1285.27it/s][A
     76%|████████████████████████████▏        | 3193/4186 [00:02<00:00, 1293.71it/s][A
     79%|█████████████████████████████▍       | 3326/4186 [00:02<00:00, 1303.60it/s][A
     83%|██████████████████████████████▌      | 3459/4186 [00:02<00:00, 1310.80it/s][A
     86%|███████████████████████████████▋     | 3592/4186 [00:02<00:00, 1316.07it/s][A
     89%|████████████████████████████████▉    | 3725/4186 [00:02<00:00, 1319.19it/s][A
     92%|██████████████████████████████████   | 3858/4186 [00:02<00:00, 1320.65it/s][A
     95%|███████████████████████████████████▎ | 3991/4186 [00:03<00:00, 1322.51it/s][A
    100%|█████████████████████████████████████| 4186/4186 [00:03<00:00, 1313.18it/s][A
    Testing DataLoader 0:  48%|██████████▏          | 15/31 [04:15<04:32,  0.06it/s]Sampling. The number of nodes to sample is 2800.
    Sampling on one graph took 8.65767502784729 seconds.
    
      0%|                                                  | 0/2800 [00:00<?, ?it/s][A
      6%|██▍                                   | 182/2800 [00:00<00:01, 1817.22it/s][A
     13%|████▉                                 | 364/2800 [00:00<00:01, 1808.58it/s][A
     20%|███████▍                              | 546/2800 [00:00<00:01, 1812.94it/s][A
     26%|█████████▉                            | 729/2800 [00:00<00:01, 1817.12it/s][A
     33%|████████████▍                         | 912/2800 [00:00<00:01, 1819.67it/s][A
     39%|██████████████▍                      | 1095/2800 [00:00<00:00, 1820.19it/s][A
     46%|████████████████▉                    | 1278/2800 [00:00<00:00, 1822.78it/s][A
     52%|███████████████████▎                 | 1461/2800 [00:00<00:00, 1743.50it/s][A
     59%|█████████████████████▋               | 1644/2800 [00:00<00:00, 1768.23it/s][A
     65%|████████████████████████▏            | 1826/2800 [00:01<00:00, 1782.33it/s][A
     72%|██████████████████████████▌          | 2008/2800 [00:01<00:00, 1793.43it/s][A
     78%|████████████████████████████▉        | 2190/2800 [00:01<00:00, 1799.53it/s][A
     85%|███████████████████████████████▎     | 2373/2800 [00:01<00:00, 1807.99it/s][A
     91%|█████████████████████████████████▊   | 2556/2800 [00:01<00:00, 1813.25it/s][A
    100%|█████████████████████████████████████| 2800/2800 [00:01<00:00, 1803.24it/s][A
    Testing DataLoader 0:  52%|██████████▊          | 16/31 [04:28<04:11,  0.06it/s]Sampling. The number of nodes to sample is 4529.
    Sampling on one graph took 9.847249984741211 seconds.
    
      0%|                                                  | 0/4529 [00:00<?, ?it/s][A
      3%|█                                     | 124/4529 [00:00<00:03, 1231.65it/s][A
      5%|██                                    | 249/4529 [00:00<00:03, 1236.76it/s][A
      8%|███▏                                  | 373/4529 [00:00<00:03, 1229.29it/s][A
     11%|████▏                                 | 497/4529 [00:00<00:03, 1233.33it/s][A
     14%|█████▏                                | 621/4529 [00:00<00:03, 1235.20it/s][A
     16%|██████▎                               | 745/4529 [00:00<00:03, 1236.79it/s][A
     19%|███████▎                              | 869/4529 [00:00<00:02, 1237.75it/s][A
     22%|████████▎                             | 994/4529 [00:00<00:02, 1239.20it/s][A
     25%|█████████▏                           | 1118/4529 [00:00<00:02, 1239.36it/s][A
     27%|██████████▏                          | 1242/4529 [00:01<00:02, 1182.74it/s][A
     30%|███████████▏                         | 1366/4529 [00:01<00:02, 1199.43it/s][A
     33%|████████████▏                        | 1491/4529 [00:01<00:02, 1211.74it/s][A
     36%|█████████████▏                       | 1616/4529 [00:01<00:02, 1222.00it/s][A
     38%|██████████████▏                      | 1739/4529 [00:01<00:02, 1223.41it/s][A
     41%|███████████████▏                     | 1863/4529 [00:01<00:02, 1226.44it/s][A
     44%|████████████████▏                    | 1987/4529 [00:01<00:02, 1230.01it/s][A
     47%|█████████████████▎                   | 2112/4529 [00:01<00:01, 1233.63it/s][A
     49%|██████████████████▎                  | 2236/4529 [00:01<00:01, 1234.55it/s][A
     52%|███████████████████▎                 | 2360/4529 [00:01<00:01, 1235.43it/s][A
     55%|████████████████████▎                | 2485/4529 [00:02<00:01, 1237.60it/s][A
     58%|█████████████████████▎               | 2610/4529 [00:02<00:01, 1238.56it/s][A
     60%|██████████████████████▎              | 2735/4529 [00:02<00:01, 1239.27it/s][A
     63%|███████████████████████▎             | 2859/4529 [00:02<00:01, 1192.27it/s][A
     66%|████████████████████████▍            | 2984/4529 [00:02<00:01, 1206.83it/s][A
     69%|█████████████████████████▍           | 3109/4529 [00:02<00:01, 1216.88it/s][A
     71%|██████████████████████████▍          | 3234/4529 [00:02<00:01, 1224.23it/s][A
     74%|███████████████████████████▍         | 3359/4529 [00:02<00:00, 1229.39it/s][A
     77%|████████████████████████████▍        | 3484/4529 [00:02<00:00, 1233.61it/s][A
     80%|█████████████████████████████▍       | 3609/4529 [00:02<00:00, 1237.04it/s][A
     82%|██████████████████████████████▍      | 3733/4529 [00:03<00:00, 1237.75it/s][A
     85%|███████████████████████████████▌     | 3857/4529 [00:03<00:00, 1238.19it/s][A
     88%|████████████████████████████████▌    | 3982/4529 [00:03<00:00, 1239.16it/s][A
     91%|█████████████████████████████████▌   | 4107/4529 [00:03<00:00, 1239.81it/s][A
     93%|██████████████████████████████████▌  | 4232/4529 [00:03<00:00, 1240.45it/s][A
     96%|███████████████████████████████████▌ | 4357/4529 [00:03<00:00, 1193.18it/s][A
    100%|█████████████████████████████████████| 4529/4529 [00:03<00:00, 1224.87it/s][A
    Testing DataLoader 0:  55%|███████████▌         | 17/31 [04:47<03:56,  0.06it/s]Sampling. The number of nodes to sample is 5021.
    Sampling on one graph took 10.38461184501648 seconds.
    
      0%|                                                  | 0/5021 [00:00<?, ?it/s][A
      2%|▋                                       | 94/5021 [00:00<00:05, 927.02it/s][A
      4%|█▌                                    | 200/5021 [00:00<00:04, 1001.66it/s][A
      6%|██▎                                   | 312/5021 [00:00<00:04, 1054.82it/s][A
      8%|███▏                                  | 424/5021 [00:00<00:04, 1080.41it/s][A
     11%|████                                  | 537/5021 [00:00<00:04, 1095.29it/s][A
     13%|████▉                                 | 650/5021 [00:00<00:03, 1103.81it/s][A
     15%|█████▊                                | 762/5021 [00:00<00:03, 1108.22it/s][A
     17%|██████▌                               | 874/5021 [00:00<00:03, 1111.50it/s][A
     20%|███████▍                              | 986/5021 [00:00<00:03, 1113.24it/s][A
     22%|████████                             | 1099/5021 [00:01<00:03, 1115.85it/s][A
     24%|████████▉                            | 1211/5021 [00:01<00:03, 1116.47it/s][A
     26%|█████████▋                           | 1323/5021 [00:01<00:03, 1115.93it/s][A
     29%|██████████▌                          | 1436/5021 [00:01<00:03, 1117.31it/s][A
     31%|███████████▍                         | 1548/5021 [00:01<00:03, 1066.34it/s][A
     33%|████████████▏                        | 1661/5021 [00:01<00:03, 1082.10it/s][A
     35%|█████████████                        | 1774/5021 [00:01<00:02, 1093.55it/s][A
     38%|█████████████▉                       | 1886/5021 [00:01<00:02, 1101.06it/s][A
     40%|██████████████▋                      | 1998/5021 [00:01<00:02, 1106.62it/s][A
     42%|███████████████▌                     | 2110/5021 [00:01<00:02, 1109.51it/s][A
     44%|████████████████▎                    | 2222/5021 [00:02<00:02, 1112.41it/s][A
     46%|█████████████████▏                   | 2334/5021 [00:02<00:02, 1114.14it/s][A
     49%|██████████████████                   | 2446/5021 [00:02<00:02, 1115.48it/s][A
     51%|██████████████████▊                  | 2559/5021 [00:02<00:02, 1118.45it/s][A
     53%|███████████████████▋                 | 2671/5021 [00:02<00:02, 1118.32it/s][A
     55%|████████████████████▌                | 2783/5021 [00:02<00:02, 1117.79it/s][A
     58%|█████████████████████▎               | 2895/5021 [00:02<00:01, 1110.66it/s][A
     60%|██████████████████████▏              | 3007/5021 [00:02<00:01, 1067.10it/s][A
     62%|██████████████████████▉              | 3120/5021 [00:02<00:01, 1083.22it/s][A
     64%|███████████████████████▊             | 3233/5021 [00:02<00:01, 1095.54it/s][A
     67%|████████████████████████▋            | 3346/5021 [00:03<00:01, 1103.04it/s][A
     69%|█████████████████████████▍           | 3458/5021 [00:03<00:01, 1107.96it/s][A
     71%|██████████████████████████▎          | 3569/5021 [00:03<00:01, 1104.31it/s][A
     73%|███████████████████████████▏         | 3682/5021 [00:03<00:01, 1109.47it/s][A
     76%|███████████████████████████▉         | 3794/5021 [00:03<00:01, 1110.18it/s][A
     78%|████████████████████████████▊        | 3906/5021 [00:03<00:01, 1108.90it/s][A
     80%|█████████████████████████████▌       | 4018/5021 [00:03<00:00, 1110.97it/s][A
     82%|██████████████████████████████▍      | 4130/5021 [00:03<00:00, 1113.43it/s][A
     85%|███████████████████████████████▎     | 4243/5021 [00:03<00:00, 1115.65it/s][A
     87%|████████████████████████████████     | 4355/5021 [00:03<00:00, 1115.55it/s][A
     89%|████████████████████████████████▉    | 4467/5021 [00:04<00:00, 1073.56it/s][A
     91%|█████████████████████████████████▊   | 4580/5021 [00:04<00:00, 1088.25it/s][A
     93%|██████████████████████████████████▌  | 4693/5021 [00:04<00:00, 1098.13it/s][A
     96%|███████████████████████████████████▍ | 4806/5021 [00:04<00:00, 1106.47it/s][A
    100%|█████████████████████████████████████| 5021/5021 [00:04<00:00, 1101.90it/s][A
    Testing DataLoader 0:  58%|████████████▏        | 18/31 [05:08<03:42,  0.06it/s]Sampling. The number of nodes to sample is 4991.
    Sampling on one graph took 10.389180421829224 seconds.
    
      0%|                                                  | 0/4991 [00:00<?, ?it/s][A
      2%|▊                                     | 113/4991 [00:00<00:04, 1122.56it/s][A
      5%|█▋                                    | 227/4991 [00:00<00:04, 1127.10it/s][A
      7%|██▌                                   | 340/4991 [00:00<00:04, 1128.12it/s][A
      9%|███▍                                  | 453/4991 [00:00<00:04, 1122.71it/s][A
     11%|████▎                                 | 567/4991 [00:00<00:03, 1125.53it/s][A
     14%|█████▏                                | 680/4991 [00:00<00:03, 1126.62it/s][A
     16%|██████                                | 794/4991 [00:00<00:03, 1128.29it/s][A
     18%|██████▉                               | 908/4991 [00:00<00:03, 1129.09it/s][A
     20%|███████▌                             | 1021/4991 [00:00<00:03, 1081.61it/s][A
     23%|████████▍                            | 1130/4991 [00:01<00:03, 1052.65it/s][A
     25%|█████████▏                           | 1244/4991 [00:01<00:03, 1075.54it/s][A
     27%|██████████                           | 1357/4991 [00:01<00:03, 1091.10it/s][A
     29%|██████████▉                          | 1470/4991 [00:01<00:03, 1101.43it/s][A
     32%|███████████▋                         | 1584/4991 [00:01<00:03, 1110.81it/s][A
     34%|████████████▌                        | 1698/4991 [00:01<00:02, 1117.35it/s][A
     36%|█████████████▍                       | 1812/4991 [00:01<00:02, 1121.95it/s][A
     39%|██████████████▎                      | 1926/4991 [00:01<00:02, 1125.67it/s][A
     41%|███████████████                      | 2040/4991 [00:01<00:02, 1127.78it/s][A
     43%|███████████████▉                     | 2154/4991 [00:01<00:02, 1129.42it/s][A
     45%|████████████████▊                    | 2268/4991 [00:02<00:02, 1130.54it/s][A
     48%|█████████████████▋                   | 2382/4991 [00:02<00:02, 1129.98it/s][A
     50%|██████████████████▌                  | 2496/4991 [00:02<00:02, 1083.75it/s][A
     52%|███████████████████▎                 | 2605/4991 [00:02<00:02, 1053.51it/s][A
     54%|████████████████████▏                | 2719/4991 [00:02<00:02, 1076.19it/s][A
     57%|████████████████████▉                | 2832/4991 [00:02<00:01, 1091.61it/s][A
     59%|█████████████████████▊               | 2946/4991 [00:02<00:01, 1104.33it/s][A
     61%|██████████████████████▋              | 3060/4991 [00:02<00:01, 1112.12it/s][A
     64%|███████████████████████▌             | 3174/4991 [00:02<00:01, 1119.74it/s][A
     66%|████████████████████████▍            | 3288/4991 [00:02<00:01, 1123.48it/s][A
     68%|█████████████████████████▏           | 3402/4991 [00:03<00:01, 1125.61it/s][A
     70%|██████████████████████████           | 3516/4991 [00:03<00:01, 1127.66it/s][A
     73%|██████████████████████████▉          | 3630/4991 [00:03<00:01, 1129.90it/s][A
     75%|███████████████████████████▊         | 3744/4991 [00:03<00:01, 1129.64it/s][A
     77%|████████████████████████████▌        | 3858/4991 [00:03<00:01, 1129.95it/s][A
     80%|█████████████████████████████▍       | 3972/4991 [00:03<00:00, 1081.29it/s][A
     82%|██████████████████████████████▎      | 4086/4991 [00:03<00:00, 1096.66it/s][A
     84%|███████████████████████████████▏     | 4200/4991 [00:03<00:00, 1106.90it/s][A
     86%|███████████████████████████████▉     | 4314/4991 [00:03<00:00, 1115.54it/s][A
     89%|████████████████████████████████▊    | 4428/4991 [00:03<00:00, 1120.82it/s][A
     91%|█████████████████████████████████▋   | 4542/4991 [00:04<00:00, 1123.95it/s][A
     93%|██████████████████████████████████▌  | 4656/4991 [00:04<00:00, 1126.19it/s][A
     96%|███████████████████████████████████▎ | 4770/4991 [00:04<00:00, 1127.73it/s][A
    100%|█████████████████████████████████████| 4991/4991 [00:04<00:00, 1112.49it/s][A
    Testing DataLoader 0:  61%|████████████▊        | 19/31 [05:29<03:28,  0.06it/s]Sampling. The number of nodes to sample is 4389.
    Sampling on one graph took 9.687657594680786 seconds.
    
      0%|                                                  | 0/4389 [00:00<?, ?it/s][A
      3%|█                                     | 128/4389 [00:00<00:03, 1272.08it/s][A
      6%|██▏                                   | 256/4389 [00:00<00:03, 1275.04it/s][A
      9%|███▎                                  | 385/4389 [00:00<00:03, 1277.45it/s][A
     12%|████▍                                 | 513/4389 [00:00<00:03, 1208.60it/s][A
     15%|█████▌                                | 641/4389 [00:00<00:03, 1231.98it/s][A
     18%|██████▋                               | 770/4389 [00:00<00:02, 1248.81it/s][A
     20%|███████▊                              | 899/4389 [00:00<00:02, 1259.68it/s][A
     23%|████████▋                            | 1028/4389 [00:00<00:02, 1266.03it/s][A
     26%|█████████▊                           | 1157/4389 [00:00<00:02, 1271.16it/s][A
     29%|██████████▊                          | 1286/4389 [00:01<00:02, 1275.21it/s][A
     32%|███████████▉                         | 1414/4389 [00:01<00:02, 1273.71it/s][A
     35%|████████████▉                        | 1542/4389 [00:01<00:02, 1271.18it/s][A
     38%|██████████████                       | 1670/4389 [00:01<00:02, 1271.40it/s][A
     41%|███████████████▏                     | 1798/4389 [00:01<00:02, 1270.42it/s][A
     44%|████████████████▏                    | 1927/4389 [00:01<00:01, 1274.12it/s][A
     47%|█████████████████▎                   | 2055/4389 [00:01<00:01, 1225.44it/s][A
     50%|██████████████████▍                  | 2182/4389 [00:01<00:01, 1237.16it/s][A
     53%|███████████████████▍                 | 2311/4389 [00:01<00:01, 1250.24it/s][A
     56%|████████████████████▌                | 2439/4389 [00:01<00:01, 1258.31it/s][A
     58%|█████████████████████▋               | 2567/4389 [00:02<00:01, 1264.50it/s][A
     61%|██████████████████████▋              | 2696/4389 [00:02<00:01, 1269.76it/s][A
     64%|███████████████████████▊             | 2825/4389 [00:02<00:01, 1273.65it/s][A
     67%|████████████████████████▉            | 2954/4389 [00:02<00:01, 1275.94it/s][A
     70%|█████████████████████████▉           | 3083/4389 [00:02<00:01, 1278.47it/s][A
     73%|███████████████████████████          | 3211/4389 [00:02<00:00, 1277.34it/s][A
     76%|████████████████████████████▏        | 3340/4389 [00:02<00:00, 1279.07it/s][A
     79%|█████████████████████████████▏       | 3468/4389 [00:02<00:00, 1279.34it/s][A
     82%|██████████████████████████████▎      | 3597/4389 [00:02<00:00, 1280.43it/s][A
     85%|███████████████████████████████▍     | 3726/4389 [00:02<00:00, 1281.35it/s][A
     88%|████████████████████████████████▍    | 3855/4389 [00:03<00:00, 1229.30it/s][A
     91%|█████████████████████████████████▌   | 3984/4389 [00:03<00:00, 1244.47it/s][A
     94%|██████████████████████████████████▋  | 4113/4389 [00:03<00:00, 1255.45it/s][A
     97%|███████████████████████████████████▊ | 4242/4389 [00:03<00:00, 1262.99it/s][A
    100%|█████████████████████████████████████| 4389/4389 [00:03<00:00, 1263.04it/s][A
    Testing DataLoader 0:  65%|█████████████▌       | 20/31 [05:47<03:11,  0.06it/s]Sampling. The number of nodes to sample is 3560.
    Sampling on one graph took 8.618864297866821 seconds.
    
      0%|                                                  | 0/3560 [00:00<?, ?it/s][A
      4%|█▌                                    | 152/3560 [00:00<00:02, 1510.39it/s][A
      9%|███▎                                  | 305/3560 [00:00<00:02, 1519.33it/s][A
     13%|████▉                                 | 458/3560 [00:00<00:02, 1521.86it/s][A
     17%|██████▌                               | 611/3560 [00:00<00:01, 1522.51it/s][A
     21%|████████▏                             | 764/3560 [00:00<00:01, 1522.94it/s][A
     26%|█████████▊                            | 917/3560 [00:00<00:01, 1450.74it/s][A
     30%|███████████                          | 1070/3560 [00:00<00:01, 1473.86it/s][A
     34%|████████████▋                        | 1223/3560 [00:00<00:01, 1490.19it/s][A
     39%|██████████████▎                      | 1373/3560 [00:00<00:01, 1490.77it/s][A
     43%|███████████████▊                     | 1526/3560 [00:01<00:01, 1500.75it/s][A
     47%|█████████████████▍                   | 1679/3560 [00:01<00:01, 1508.04it/s][A
     51%|███████████████████                  | 1832/3560 [00:01<00:01, 1512.92it/s][A
     56%|████████████████████▋                | 1985/3560 [00:01<00:01, 1517.16it/s][A
     60%|██████████████████████▏              | 2137/3560 [00:01<00:00, 1517.05it/s][A
     64%|███████████████████████▊             | 2290/3560 [00:01<00:00, 1519.78it/s][A
     69%|█████████████████████████▍           | 2443/3560 [00:01<00:00, 1521.20it/s][A
     73%|██████████████████████████▉          | 2596/3560 [00:01<00:00, 1522.06it/s][A
     77%|████████████████████████████▌        | 2749/3560 [00:01<00:00, 1523.50it/s][A
     82%|██████████████████████████████▏      | 2902/3560 [00:01<00:00, 1465.03it/s][A
     86%|███████████████████████████████▋     | 3053/3560 [00:02<00:00, 1477.92it/s][A
     90%|█████████████████████████████████▎   | 3206/3560 [00:02<00:00, 1491.24it/s][A
     94%|██████████████████████████████████▉  | 3359/3560 [00:02<00:00, 1501.23it/s][A
    100%|█████████████████████████████████████| 3560/3560 [00:02<00:00, 1503.46it/s][A
    Testing DataLoader 0:  68%|██████████████▏      | 21/31 [06:02<02:52,  0.06it/s]Sampling. The number of nodes to sample is 1706.
    Sampling on one graph took 8.470772743225098 seconds.
    
      0%|                                                  | 0/1706 [00:00<?, ?it/s][A
     14%|█████▌                                | 247/1706 [00:00<00:00, 2464.74it/s][A
     29%|███████████                           | 496/1706 [00:00<00:00, 2476.44it/s][A
     44%|████████████████▋                     | 747/1706 [00:00<00:00, 2489.41it/s][A
     58%|██████████████████████▏               | 996/1706 [00:00<00:00, 2489.32it/s][A
     73%|███████████████████████████          | 1245/1706 [00:00<00:00, 2481.81it/s][A
    100%|█████████████████████████████████████| 1706/1706 [00:00<00:00, 2487.71it/s][A
    Testing DataLoader 0:  71%|██████████████▉      | 22/31 [06:12<02:32,  0.06it/s]Sampling. The number of nodes to sample is 3296.
    Sampling on one graph took 8.49496054649353 seconds.
    
      0%|                                                  | 0/3296 [00:00<?, ?it/s][A
      5%|█▊                                    | 159/3296 [00:00<00:01, 1589.11it/s][A
     10%|███▋                                  | 319/3296 [00:00<00:01, 1589.96it/s][A
     15%|█████▌                                | 480/3296 [00:00<00:01, 1596.03it/s][A
     19%|███████▍                              | 640/3296 [00:00<00:01, 1594.30it/s][A
     24%|█████████▏                            | 800/3296 [00:00<00:01, 1595.28it/s][A
     29%|███████████                           | 960/3296 [00:00<00:01, 1594.33it/s][A
     34%|████████████▌                        | 1120/3296 [00:00<00:01, 1593.38it/s][A
     39%|██████████████▎                      | 1280/3296 [00:00<00:01, 1588.77it/s][A
     44%|████████████████▏                    | 1440/3296 [00:00<00:01, 1591.70it/s][A
     49%|█████████████████▉                   | 1600/3296 [00:01<00:01, 1521.32it/s][A
     53%|███████████████████▊                 | 1760/3296 [00:01<00:00, 1544.18it/s][A
     58%|█████████████████████▌               | 1917/3296 [00:01<00:00, 1551.66it/s][A
     63%|███████████████████████▎             | 2077/3296 [00:01<00:00, 1565.15it/s][A
     68%|█████████████████████████            | 2238/3296 [00:01<00:00, 1575.84it/s][A
     73%|██████████████████████████▉          | 2398/3296 [00:01<00:00, 1580.51it/s][A
     78%|████████████████████████████▋        | 2557/3296 [00:01<00:00, 1576.39it/s][A
     82%|██████████████████████████████▍      | 2715/3296 [00:01<00:00, 1557.89it/s][A
     87%|████████████████████████████████▎    | 2875/3296 [00:01<00:00, 1569.14it/s][A
     92%|██████████████████████████████████   | 3035/3296 [00:01<00:00, 1577.72it/s][A
    100%|█████████████████████████████████████| 3296/3296 [00:02<00:00, 1576.11it/s][A
    Testing DataLoader 0:  74%|███████████████▌     | 23/31 [06:25<02:14,  0.06it/s]Sampling. The number of nodes to sample is 5155.
    Sampling on one graph took 10.484142065048218 seconds.
    
      0%|                                                  | 0/5155 [00:00<?, ?it/s][A
      2%|▊                                     | 110/5155 [00:00<00:04, 1098.65it/s][A
      4%|█▋                                    | 221/5155 [00:00<00:04, 1102.27it/s][A
      6%|██▍                                   | 333/5155 [00:00<00:04, 1105.81it/s][A
      9%|███▎                                  | 444/5155 [00:00<00:04, 1106.11it/s][A
     11%|████                                  | 555/5155 [00:00<00:04, 1106.22it/s][A
     13%|████▉                                 | 666/5155 [00:00<00:04, 1106.06it/s][A
     15%|█████▋                                | 777/5155 [00:00<00:03, 1106.63it/s][A
     17%|██████▌                               | 888/5155 [00:00<00:03, 1102.22it/s][A
     19%|███████▎                              | 999/5155 [00:00<00:03, 1058.94it/s][A
     22%|███████▉                             | 1110/5155 [00:01<00:03, 1073.49it/s][A
     24%|████████▊                            | 1221/5155 [00:01<00:03, 1083.65it/s][A
     26%|█████████▌                           | 1332/5155 [00:01<00:03, 1089.63it/s][A
     28%|██████████▎                          | 1443/5155 [00:01<00:03, 1094.27it/s][A
     30%|███████████▏                         | 1554/5155 [00:01<00:03, 1098.43it/s][A
     32%|███████████▉                         | 1666/5155 [00:01<00:03, 1102.30it/s][A
     34%|████████████▊                        | 1777/5155 [00:01<00:03, 1103.80it/s][A
     37%|█████████████▌                       | 1888/5155 [00:01<00:02, 1104.19it/s][A
     39%|██████████████▎                      | 1999/5155 [00:01<00:02, 1104.92it/s][A
     41%|███████████████▏                     | 2110/5155 [00:01<00:02, 1104.73it/s][A
     43%|███████████████▉                     | 2221/5155 [00:02<00:02, 1105.86it/s][A
     45%|████████████████▋                    | 2332/5155 [00:02<00:02, 1106.91it/s][A
     47%|█████████████████▌                   | 2443/5155 [00:02<00:02, 1019.25it/s][A
     50%|██████████████████▎                  | 2552/5155 [00:02<00:02, 1038.98it/s][A
     52%|███████████████████                  | 2663/5155 [00:02<00:02, 1058.43it/s][A
     54%|███████████████████▉                 | 2773/5155 [00:02<00:02, 1069.92it/s][A
     56%|████████████████████▋                | 2884/5155 [00:02<00:02, 1080.36it/s][A
     58%|█████████████████████▍               | 2995/5155 [00:02<00:01, 1087.67it/s][A
     60%|██████████████████████▎              | 3106/5155 [00:02<00:01, 1092.88it/s][A
     62%|███████████████████████              | 3218/5155 [00:02<00:01, 1097.98it/s][A
     65%|███████████████████████▉             | 3329/5155 [00:03<00:01, 1101.48it/s][A
     67%|████████████████████████▋            | 3441/5155 [00:03<00:01, 1104.11it/s][A
     69%|█████████████████████████▌           | 3553/5155 [00:03<00:01, 1106.64it/s][A
     71%|██████████████████████████▎          | 3664/5155 [00:03<00:01, 1107.46it/s][A
     73%|███████████████████████████          | 3775/5155 [00:03<00:01, 1062.65it/s][A
     75%|███████████████████████████▉         | 3887/5155 [00:03<00:01, 1077.47it/s][A
     78%|████████████████████████████▋        | 3998/5155 [00:03<00:01, 1086.63it/s][A
     80%|█████████████████████████████▍       | 4110/5155 [00:03<00:00, 1094.41it/s][A
     82%|██████████████████████████████▎      | 4220/5155 [00:03<00:00, 1092.44it/s][A
     84%|███████████████████████████████      | 4332/5155 [00:03<00:00, 1098.88it/s][A
     86%|███████████████████████████████▉     | 4444/5155 [00:04<00:00, 1103.22it/s][A
     88%|████████████████████████████████▋    | 4556/5155 [00:04<00:00, 1105.43it/s][A
     91%|█████████████████████████████████▌   | 4668/5155 [00:04<00:00, 1107.64it/s][A
     93%|██████████████████████████████████▎  | 4779/5155 [00:04<00:00, 1108.05it/s][A
     95%|███████████████████████████████████  | 4890/5155 [00:04<00:00, 1102.29it/s][A
     97%|███████████████████████████████████▉ | 5001/5155 [00:04<00:00, 1098.20it/s][A
    100%|█████████████████████████████████████| 5155/5155 [00:04<00:00, 1092.12it/s][A
    Testing DataLoader 0:  77%|████████████████▎    | 24/31 [06:47<01:58,  0.06it/s]Sampling. The number of nodes to sample is 3733.
    Sampling on one graph took 8.786031723022461 seconds.
    
      0%|                                                  | 0/3733 [00:00<?, ?it/s][A
      4%|█▍                                    | 146/3733 [00:00<00:02, 1453.05it/s][A
      8%|██▉                                   | 292/3733 [00:00<00:02, 1454.91it/s][A
     12%|████▍                                 | 438/3733 [00:00<00:02, 1453.54it/s][A
     16%|█████▉                                | 584/3733 [00:00<00:02, 1375.03it/s][A
     20%|███████▍                              | 730/3733 [00:00<00:02, 1402.39it/s][A
     23%|████████▉                             | 876/3733 [00:00<00:02, 1419.67it/s][A
     27%|██████████▏                          | 1022/3733 [00:00<00:01, 1429.93it/s][A
     31%|███████████▌                         | 1168/3733 [00:00<00:01, 1438.19it/s][A
     35%|█████████████                        | 1314/3733 [00:00<00:01, 1443.91it/s][A
     39%|██████████████▍                      | 1460/3733 [00:01<00:01, 1448.28it/s][A
     43%|███████████████▉                     | 1606/3733 [00:01<00:01, 1451.65it/s][A
     47%|█████████████████▎                   | 1752/3733 [00:01<00:01, 1453.78it/s][A
     51%|██████████████████▊                  | 1898/3733 [00:01<00:01, 1454.53it/s][A
     55%|████████████████████▎                | 2044/3733 [00:01<00:01, 1453.41it/s][A
     59%|█████████████████████▋               | 2190/3733 [00:01<00:01, 1454.09it/s][A
     63%|███████████████████████▏             | 2336/3733 [00:01<00:00, 1455.65it/s][A
     66%|████████████████████████▌            | 2482/3733 [00:01<00:00, 1400.78it/s][A
     70%|██████████████████████████           | 2628/3733 [00:01<00:00, 1418.03it/s][A
     74%|███████████████████████████▍         | 2774/3733 [00:01<00:00, 1429.64it/s][A
     78%|████████████████████████████▉        | 2918/3733 [00:02<00:00, 1430.59it/s][A
     82%|██████████████████████████████▎      | 3064/3733 [00:02<00:00, 1438.11it/s][A
     86%|███████████████████████████████▊     | 3210/3733 [00:02<00:00, 1444.26it/s][A
     90%|█████████████████████████████████▎   | 3356/3733 [00:02<00:00, 1448.68it/s][A
     94%|██████████████████████████████████▋  | 3503/3733 [00:02<00:00, 1452.67it/s][A
    100%|█████████████████████████████████████| 3733/3733 [00:02<00:00, 1439.94it/s][A
    Testing DataLoader 0:  81%|████████████████▉    | 25/31 [07:02<01:41,  0.06it/s]Sampling. The number of nodes to sample is 3780.
    Sampling on one graph took 8.76235818862915 seconds.
    
      0%|                                                  | 0/3780 [00:00<?, ?it/s][A
      4%|█▍                                    | 145/3780 [00:00<00:02, 1449.33it/s][A
      8%|██▉                                   | 291/3780 [00:00<00:02, 1452.00it/s][A
     12%|████▍                                 | 437/3780 [00:00<00:02, 1454.71it/s][A
     15%|█████▊                                | 583/3780 [00:00<00:02, 1453.97it/s][A
     19%|███████▎                              | 729/3780 [00:00<00:02, 1454.24it/s][A
     23%|████████▊                             | 875/3780 [00:00<00:02, 1451.75it/s][A
     27%|█████████▉                           | 1021/3780 [00:00<00:01, 1450.17it/s][A
     31%|███████████▍                         | 1167/3780 [00:00<00:01, 1447.88it/s][A
     35%|████████████▊                        | 1312/3780 [00:00<00:01, 1447.53it/s][A
     39%|██████████████▎                      | 1457/3780 [00:01<00:01, 1447.59it/s][A
     42%|███████████████▋                     | 1602/3780 [00:01<00:01, 1448.06it/s][A
     46%|█████████████████                    | 1747/3780 [00:01<00:01, 1386.28it/s][A
     50%|██████████████████▌                  | 1892/3780 [00:01<00:01, 1403.88it/s][A
     54%|███████████████████▉                 | 2033/3780 [00:01<00:01, 1360.14it/s][A
     58%|█████████████████████▎               | 2178/3780 [00:01<00:01, 1385.86it/s][A
     61%|██████████████████████▋              | 2323/3780 [00:01<00:01, 1403.98it/s][A
     65%|████████████████████████▏            | 2468/3780 [00:01<00:00, 1416.00it/s][A
     69%|█████████████████████████▌           | 2613/3780 [00:01<00:00, 1425.42it/s][A
     73%|██████████████████████████▉          | 2758/3780 [00:01<00:00, 1431.25it/s][A
     77%|████████████████████████████▍        | 2903/3780 [00:02<00:00, 1435.94it/s][A
     81%|█████████████████████████████▊       | 3048/3780 [00:02<00:00, 1438.56it/s][A
     84%|███████████████████████████████▏     | 3192/3780 [00:02<00:00, 1436.88it/s][A
     88%|████████████████████████████████▋    | 3337/3780 [00:02<00:00, 1438.44it/s][A
     92%|██████████████████████████████████   | 3482/3780 [00:02<00:00, 1440.10it/s][A
     96%|███████████████████████████████████▌ | 3628/3780 [00:02<00:00, 1443.60it/s][A
    100%|█████████████████████████████████████| 3780/3780 [00:02<00:00, 1426.15it/s][A
    Testing DataLoader 0:  84%|█████████████████▌   | 26/31 [07:17<01:24,  0.06it/s]Sampling. The number of nodes to sample is 2905.
    Sampling on one graph took 8.580444097518921 seconds.
    
      0%|                                                  | 0/2905 [00:00<?, ?it/s][A
      6%|██▎                                   | 174/2905 [00:00<00:01, 1737.26it/s][A
     12%|████▌                                 | 352/2905 [00:00<00:01, 1758.03it/s][A
     18%|██████▉                               | 528/2905 [00:00<00:01, 1746.13it/s][A
     24%|█████████▏                            | 705/2905 [00:00<00:01, 1754.84it/s][A
     30%|███████████▌                          | 882/2905 [00:00<00:01, 1759.43it/s][A
     36%|█████████████▍                       | 1059/2905 [00:00<00:01, 1762.82it/s][A
     43%|███████████████▋                     | 1236/2905 [00:00<00:00, 1761.77it/s][A
     49%|█████████████████▉                   | 1413/2905 [00:00<00:00, 1761.76it/s][A
     55%|████████████████████▎                | 1590/2905 [00:00<00:00, 1763.29it/s][A
     61%|██████████████████████▌              | 1767/2905 [00:01<00:00, 1764.33it/s][A
     67%|████████████████████████▊            | 1944/2905 [00:01<00:00, 1751.98it/s][A
     73%|███████████████████████████          | 2120/2905 [00:01<00:00, 1682.25it/s][A
     79%|█████████████████████████████▎       | 2297/2905 [00:01<00:00, 1706.04it/s][A
     85%|███████████████████████████████▌     | 2474/2905 [00:01<00:00, 1724.43it/s][A
     91%|█████████████████████████████████▊   | 2651/2905 [00:01<00:00, 1737.89it/s][A
    100%|█████████████████████████████████████| 2905/2905 [00:01<00:00, 1744.66it/s][A
    Testing DataLoader 0:  87%|██████████████████▎  | 27/31 [07:30<01:06,  0.06it/s]Sampling. The number of nodes to sample is 3192.
    Sampling on one graph took 8.51638388633728 seconds.
    
      0%|                                                  | 0/3192 [00:00<?, ?it/s][A
      5%|█▉                                    | 163/3192 [00:00<00:01, 1627.55it/s][A
     10%|███▉                                  | 329/3192 [00:00<00:01, 1641.72it/s][A
     16%|█████▉                                | 495/3192 [00:00<00:01, 1647.22it/s][A
     21%|███████▊                              | 660/3192 [00:00<00:01, 1554.64it/s][A
     26%|█████████▊                            | 826/3192 [00:00<00:01, 1589.35it/s][A
     31%|███████████▊                          | 992/3192 [00:00<00:01, 1612.61it/s][A
     36%|█████████████▍                       | 1158/3192 [00:00<00:01, 1627.26it/s][A
     41%|███████████████▎                     | 1324/3192 [00:00<00:01, 1637.36it/s][A
     47%|█████████████████▎                   | 1491/3192 [00:00<00:01, 1645.08it/s][A
     52%|███████████████████▏                 | 1658/3192 [00:01<00:00, 1650.26it/s][A
     57%|█████████████████████▏               | 1824/3192 [00:01<00:00, 1651.60it/s][A
     62%|███████████████████████              | 1991/3192 [00:01<00:00, 1655.13it/s][A
     68%|█████████████████████████            | 2158/3192 [00:01<00:00, 1656.99it/s][A
     73%|██████████████████████████▉          | 2325/3192 [00:01<00:00, 1658.00it/s][A
     78%|████████████████████████████▉        | 2492/3192 [00:01<00:00, 1659.10it/s][A
     83%|██████████████████████████████▊      | 2659/3192 [00:01<00:00, 1659.70it/s][A
     89%|████████████████████████████████▋    | 2825/3192 [00:01<00:00, 1589.46it/s][A
     94%|██████████████████████████████████▋  | 2992/3192 [00:01<00:00, 1610.42it/s][A
    100%|█████████████████████████████████████| 3192/3192 [00:01<00:00, 1631.54it/s][A
    Testing DataLoader 0:  90%|██████████████████▉  | 28/31 [07:43<00:49,  0.06it/s]Sampling. The number of nodes to sample is 4057.
    Sampling on one graph took 8.966466426849365 seconds.
    
      0%|                                                  | 0/4057 [00:00<?, ?it/s][A
      3%|█▎                                    | 134/4057 [00:00<00:02, 1334.64it/s][A
      7%|██▌                                   | 271/4057 [00:00<00:02, 1352.35it/s][A
     10%|███▊                                  | 408/4057 [00:00<00:02, 1357.69it/s][A
     13%|█████                                 | 544/4057 [00:00<00:02, 1352.67it/s][A
     17%|██████▍                               | 681/4057 [00:00<00:02, 1357.50it/s][A
     20%|███████▋                              | 818/4057 [00:00<00:02, 1360.91it/s][A
     24%|████████▉                             | 955/4057 [00:00<00:02, 1361.81it/s][A
     27%|█████████▉                           | 1092/4057 [00:00<00:02, 1364.40it/s][A
     30%|███████████▏                         | 1229/4057 [00:00<00:02, 1306.73it/s][A
     34%|████████████▍                        | 1361/4057 [00:01<00:02, 1266.67it/s][A
     37%|█████████████▋                       | 1498/4057 [00:01<00:01, 1295.57it/s][A
     40%|██████████████▉                      | 1635/4057 [00:01<00:01, 1316.75it/s][A
     44%|████████████████▏                    | 1772/4057 [00:01<00:01, 1330.95it/s][A
     47%|█████████████████▍                   | 1909/4057 [00:01<00:01, 1341.63it/s][A
     50%|██████████████████▋                  | 2046/4057 [00:01<00:01, 1349.69it/s][A
     54%|███████████████████▉                 | 2183/4057 [00:01<00:01, 1354.59it/s][A
     57%|█████████████████████▏               | 2320/4057 [00:01<00:01, 1357.92it/s][A
     61%|██████████████████████▍              | 2457/4057 [00:01<00:01, 1359.96it/s][A
     64%|███████████████████████▋             | 2594/4057 [00:01<00:01, 1361.29it/s][A
     67%|████████████████████████▉            | 2731/4057 [00:02<00:00, 1362.19it/s][A
     71%|██████████████████████████▏          | 2868/4057 [00:02<00:00, 1364.12it/s][A
     74%|███████████████████████████▍         | 3005/4057 [00:02<00:00, 1306.52it/s][A
     77%|████████████████████████████▋        | 3142/4057 [00:02<00:00, 1324.35it/s][A
     81%|█████████████████████████████▊       | 3275/4057 [00:02<00:00, 1280.35it/s][A
     84%|███████████████████████████████      | 3412/4057 [00:02<00:00, 1304.55it/s][A
     87%|████████████████████████████████▎    | 3548/4057 [00:02<00:00, 1320.49it/s][A
     91%|█████████████████████████████████▌   | 3686/4057 [00:02<00:00, 1336.63it/s][A
     94%|██████████████████████████████████▊  | 3823/4057 [00:02<00:00, 1345.98it/s][A
    100%|█████████████████████████████████████| 4057/4057 [00:03<00:00, 1338.16it/s][A
    Testing DataLoader 0:  94%|███████████████████▋ | 29/31 [07:59<00:33,  0.06it/s]Sampling. The number of nodes to sample is 1160.
    Sampling on one graph took 8.502700567245483 seconds.
    
      0%|                                                  | 0/1160 [00:00<?, ?it/s][A
     26%|██████████                            | 307/1160 [00:00<00:00, 3061.45it/s][A
     53%|████████████████████                  | 614/1160 [00:00<00:00, 3065.38it/s][A
    100%|█████████████████████████████████████| 1160/1160 [00:00<00:00, 3075.65it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0:  97%|████████████████████▎| 30/31 [08:09<00:16,  0.06it/s]Sampling. The number of nodes to sample is 3316.
    Sampling on one graph took 8.556459903717041 seconds.
    
      0%|                                                  | 0/3316 [00:00<?, ?it/s][A
      5%|█▊                                    | 159/3316 [00:00<00:01, 1588.91it/s][A
     10%|███▋                                  | 320/3316 [00:00<00:01, 1596.89it/s][A
     15%|█████▌                                | 481/3316 [00:00<00:01, 1598.77it/s][A
     19%|███████▎                              | 642/3316 [00:00<00:01, 1600.41it/s][A
     24%|█████████▏                            | 803/3316 [00:00<00:01, 1600.75it/s][A
     29%|███████████                           | 964/3316 [00:00<00:01, 1592.12it/s][A
     34%|████████████▌                        | 1125/3316 [00:00<00:01, 1595.84it/s][A
     39%|██████████████▎                      | 1286/3316 [00:00<00:01, 1597.39it/s][A
     44%|████████████████▏                    | 1447/3316 [00:00<00:01, 1598.74it/s][A
     48%|█████████████████▉                   | 1608/3316 [00:01<00:01, 1600.55it/s][A
     53%|███████████████████▋                 | 1769/3316 [00:01<00:01, 1533.37it/s][A
     58%|█████████████████████▌               | 1930/3316 [00:01<00:00, 1554.94it/s][A
     63%|███████████████████████▎             | 2090/3316 [00:01<00:00, 1567.70it/s][A
     68%|█████████████████████████            | 2250/3316 [00:01<00:00, 1576.26it/s][A
     73%|██████████████████████████▉          | 2410/3316 [00:01<00:00, 1583.01it/s][A
     78%|████████████████████████████▋        | 2570/3316 [00:01<00:00, 1587.54it/s][A
     82%|██████████████████████████████▍      | 2731/3316 [00:01<00:00, 1592.22it/s][A
     87%|████████████████████████████████▎    | 2891/3316 [00:01<00:00, 1593.15it/s][A
     92%|██████████████████████████████████   | 3052/3316 [00:01<00:00, 1595.79it/s][A
    100%|█████████████████████████████████████| 3316/3316 [00:02<00:00, 1588.16it/s][A
    /nfs/team361/mv11/LUNA/metrics/evaluation_statistics.py:43: UserWarning: Optimal rotation is not uniquely or poorly defined for the given sets of vectors.
      rot, rssd, sens = R.align_vectors(
    Testing DataLoader 0: 100%|█████████████████████| 31/31 [08:22<00:00,  0.06it/s]
    [rank0]:[W822 12:29:01.981047104 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())



```python

```


```python
import glob, os, pandas as pd, matplotlib.pyplot as plt

# find your latest training run dir
runs = sorted(glob.glob("/nfs/team361/mv11/outputs/*/??-??-??-MERFISH_mouse_cortex"), key=os.path.getmtime)
assert runs, "No training outputs found"
run = runs[-1]
metrics_csv = os.path.join(run, "lightning_logs", "version_0", "metrics.csv")
print("metrics:", metrics_csv)

df = pd.read_csv(metrics_csv)

# Show available columns so you can pick what to plot
print(sorted([c for c in df.columns if c not in ("step","epoch","time")])[:10], "...")

# Try common keys the repo logs
candidates = [c for c in df.columns if "train" in c or "val" in c]
print("candidate series:", candidates)

# Plot any columns that look like losses
plt.figure(figsize=(8,4))
for col in candidates:
    if df[col].dtype != "O":  # numeric only
        plt.plot(df["step"] if "step" in df else df["epoch"], df[col], label=col)
plt.xlabel("step" if "step" in df else "epoch")
plt.ylabel("value")
plt.legend()
plt.title("Training metrics (CSVLogger)")
plt.show()

```

    metrics: /nfs/team361/mv11/outputs/2025-08-22/11-06-41-MERFISH_mouse_cortex/lightning_logs/version_0/metrics.csv
    ['lr-AdamW', 'train_epoch/position_mse', 'train_loss/position_mse'] ...
    candidate series: ['train_epoch/position_mse', 'train_loss/position_mse']



    
![png](output_25_1.png)
    



```python
import os, glob, re, pandas as pd

# where test outputs were written
EXP_DIR = "/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA"

# Prefer test_only results; fall back to train_and_test if present
roots = [p for p in [
    os.path.join(EXP_DIR, "MERFISH_mouse_cortex_test_only"),
    os.path.join(EXP_DIR, "MERFISH_mouse_cortex"),
] if os.path.isdir(p)]
assert roots, f"No results under {EXP_DIR}"
print("Searching in:", roots)

# Find checkpoint result dirs like: model_<...>_epoch_<N>
ckpt_dirs = []
for r in roots:
    ckpt_dirs += glob.glob(os.path.join(r, "model_*_epoch_*"))
assert ckpt_dirs, "No 'model_*_epoch_*' result dirs found."

def epoch_from_dir(d):
    m = re.search(r"epoch_(\d+)$", d)
    return int(m.group(1)) if m else None

# Deduplicate by epoch (favor test_only if both exist)
dedup = {}
for d in sorted(ckpt_dirs, key=epoch_from_dir):
    e = epoch_from_dir(d)
    if e not in dedup or "test_only" in d:
        dedup[e] = d
ckpt_dirs = [dedup[e] for e in sorted(dedup)]
print("Checkpoints:", {epoch_from_dir(d): os.path.basename(d) for d in ckpt_dirs})

```

    Searching in: ['/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/MERFISH_mouse_cortex_test_only', '/nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/MERFISH_mouse_cortex']
    Checkpoints: {249: 'model_11-06-41-MERFISH_epoch_249', 499: 'model_11-06-41-MERFISH_epoch_499', 749: 'model_11-06-41-MERFISH_epoch_749', 999: 'model_11-06-41-MERFISH_epoch_999'}



```python
import matplotlib.pyplot as plt

all_rows = []
for d in ckpt_dirs:
    epoch = epoch_from_dir(d)
    # the CSV is named like "<general.name>_results.csv"
    cands = glob.glob(os.path.join(d, "*_results.csv"))
    if not cands:
        print("No results CSV in", d); 
        continue
    csv = cands[0]
    df = pd.read_csv(csv)
    df["epoch"] = epoch
    df["checkpoint_dir"] = d
    all_rows.append(df)

results = pd.concat(all_rows, ignore_index=True)
print("Rows:", len(results))
display(results.head(-5))

```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[1], line 4
          1 import matplotlib.pyplot as plt
          3 all_rows = []
    ----> 4 for d in ckpt_dirs:
          5     epoch = epoch_from_dir(d)
          6     # the CSV is named like "<general.name>_results.csv"


    NameError: name 'ckpt_dirs' is not defined



```python
metrics = [
    "test/precision",
    "test/F1",
    "test/Spearman's Rank Correlation (Median)",
    "test/Spearman's Rank Correlation (Average)",
    "test/rssd_absolute",
    "test/mean_rssd",
    "test/sum_rssd",
]
summary = results.groupby("epoch")[metrics].mean().reset_index().sort_values("epoch")
display(summary)

best_by = {
    "F1↑": summary.loc[summary["test/F1"].idxmax(), "epoch"],
    "Spearman(median)↑": summary.loc[summary["test/Spearman's Rank Correlation (Median)"].idxmax(), "epoch"],
    "Absolute RSSD↓": summary.loc[summary["test/rssd_absolute"].idxmin(), "epoch"],
}
print("Best epochs:", best_by)

# Save tidy outputs for reporting
FIG_DIR = os.path.join(EXP_DIR, "figs"); os.makedirs(FIG_DIR, exist_ok=True)
summary_path = os.path.join(FIG_DIR, "checkpoint_metrics_summary.csv")
results.to_csv(os.path.join(FIG_DIR, "checkpoint_metrics_all_sections.csv"), index=False)
summary.to_csv(summary_path, index=False)
print("Saved:", summary_path)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>epoch</th>
      <th>test/precision</th>
      <th>test/F1</th>
      <th>test/Spearman's Rank Correlation (Median)</th>
      <th>test/Spearman's Rank Correlation (Average)</th>
      <th>test/rssd_absolute</th>
      <th>test/mean_rssd</th>
      <th>test/sum_rssd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>249</td>
      <td>0.253212</td>
      <td>0.253212</td>
      <td>0.445054</td>
      <td>0.366442</td>
      <td>19.012534</td>
      <td>3.540934</td>
      <td>79.246208</td>
    </tr>
    <tr>
      <th>1</th>
      <td>499</td>
      <td>0.258289</td>
      <td>0.258289</td>
      <td>0.430924</td>
      <td>0.351618</td>
      <td>19.562682</td>
      <td>3.631927</td>
      <td>81.291483</td>
    </tr>
    <tr>
      <th>2</th>
      <td>749</td>
      <td>0.265557</td>
      <td>0.265557</td>
      <td>0.442510</td>
      <td>0.363965</td>
      <td>19.206757</td>
      <td>3.567502</td>
      <td>79.829527</td>
    </tr>
    <tr>
      <th>3</th>
      <td>999</td>
      <td>0.266894</td>
      <td>0.266894</td>
      <td>0.449755</td>
      <td>0.373062</td>
      <td>18.993377</td>
      <td>3.528381</td>
      <td>78.961858</td>
    </tr>
  </tbody>
</table>
</div>


    Best epochs: {'F1↑': 999, 'Spearman(median)↑': 999, 'Absolute RSSD↓': 999}
    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/checkpoint_metrics_summary.csv



```python
import numpy as np, pandas as pd, os, glob, seaborn as sns, matplotlib.pyplot as plt, colorcet as cc

def _auto_flip_GT_to_match_pred(gt_df: pd.DataFrame, pred_df: pd.DataFrame):
    """
    Decide whether to flip GT around its own centroid along X and/or Y
    so that it best matches the prediction. We only consider four cases:
    identity, flip-x, flip-y, flip-both. No scaling, no rotation.
    Returns: flipped_gt_df, chosen_label, score
    """
    # join on cell IDs (index); keep common cells only
    df = gt_df.join(pred_df[["coord_X", "coord_Y"]]
                    .rename(columns={"coord_X":"coord_X_pred","coord_Y":"coord_Y_pred"}),
                    how="inner")
    xg, yg = df["coord_X"].values, df["coord_Y"].values
    xp, yp = df["coord_X_pred"].values, df["coord_Y_pred"].values

    # center to compare correlations robustly
    xg0, yg0 = xg - xg.mean(), yg - yg.mean()
    xp0, yp0 = xp - xp.mean(), yp - yp.mean()

    opts = {
        "identity": ( 1,  1),
        "flip_x"  : (-1,  1),
        "flip_y"  : ( 1, -1),
        "flip_xy" : (-1, -1),
    }

    best_lab, best_score = None, -np.inf
    for lab, (sx, sy) in opts.items():
        xx = sx * xg0
        yy = sy * yg0
        # sum of Pearson correlations; NaNs -> very bad
        c1 = np.corrcoef(xx, xp0)[0,1]
        c2 = np.corrcoef(yy, yp0)[0,1]
        score = (0 if np.isnan(c1) else c1) + (0 if np.isnan(c2) else c2)
        if score > best_score:
            best_lab, best_score = lab, score
            best_sx, best_sy = sx, sy

    # flip GT about its own centroid (keeps GT scale & bbox)
    mx, my = gt_df["coord_X"].mean(), gt_df["coord_Y"].mean()
    gt_fix = gt_df.copy()
    gt_fix["coord_X"] = best_sx * (gt_fix["coord_X"] - mx) + mx
    gt_fix["coord_Y"] = best_sy * (gt_fix["coord_Y"] - my) + my
    return gt_fix, best_lab, best_score


def plot_section_across_epochs_fixGT(section, ckpt_dirs, save_dir, ref_dir=None):
    """
    Same layout as your original function, but GT is auto-flipped (if needed)
    to match orientation of predictions from ref_dir (default: last checkpoint).
    """
    # load GT
    gt_path = os.path.join(ckpt_dirs[0], section, "metadata_true.csv")
    t = pd.read_csv(gt_path, index_col=0)

    # choose a reference prediction (use latest checkpoint by default)
    if ref_dir is None:
        ref_dir = ckpt_dirs[-1]
    ref_pred_path = os.path.join(ref_dir, section, "metadata_pred.csv")
    if os.path.exists(ref_pred_path):
        p_ref = pd.read_csv(ref_pred_path, index_col=0)
        t_fixed, choice, score = _auto_flip_GT_to_match_pred(t, p_ref)
        # uncomment if you want to see what it chose:
        # print(f"{section}: GT flip = {choice} (score {score:.3f})")
        t = t_fixed  # use flipped GT for plotting
    else:
        # nothing to flip against; keep original GT
        choice = "identity"

    uniques = np.unique(t["cell_class"])
    palette = dict(zip(uniques, sns.color_palette(cc.glasbey, n_colors=len(uniques))))

    n_epochs = len(ckpt_dirs)
    cols = n_epochs + 1
    fig, axes = plt.subplots(1, cols, figsize=(5*cols, 5))

    # GT
    ax = axes[0]
    sns.scatterplot(data=t, x="coord_X", y="coord_Y", hue="cell_class",
                    s=5, ax=ax, palette=palette, legend=False)
    ax.set_title(f"{section} — GT ({choice})"); ax.set_xlabel("X"); ax.set_ylabel("Y")

    # predictions per epoch (unchanged, your original view)
    for j, d in enumerate(ckpt_dirs, start=1):
        p_csv = os.path.join(d, section, "metadata_pred.csv")
        ax = axes[j]
        if not os.path.exists(p_csv):
            ax.axis("off"); ax.set_title(f"missing: {os.path.basename(d)}")
            continue
        p = pd.read_csv(p_csv, index_col=0).replace([np.inf, -np.inf], np.nan).fillna(0)
        sns.scatterplot(data=p, x="coord_X", y="coord_Y", hue="cell_class",
                        s=5, ax=ax, palette=palette, legend=False)
        ax.set_title(f"epoch {epoch_from_dir(d)}"); ax.set_xlabel("X"); ax.set_ylabel("Y")

    plt.tight_layout()
    out = os.path.join(save_dir, f"section_{section}_epochs_grid_fixedGT.png")
    plt.savefig(out, dpi=150); print("Saved:", out)
    plt.show()


for s in top_sections:
    plot_section_across_epochs_fixGT(s, ckpt_dirs, FIG_DIR) 

```

    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/section_mouse2_slice99_0_epochs_grid_fixedGT.png



    
![png](output_29_1.png)
    


    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/section_mouse2_slice109_0_epochs_grid_fixedGT.png



    
![png](output_29_3.png)
    


    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/section_mouse2_slice129_0_epochs_grid_fixedGT.png



    
![png](output_29_5.png)
    


    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/section_mouse2_slice169_0_epochs_grid_fixedGT.png



    
![png](output_29_7.png)
    


    Saved: /nfs/team361/mv11/scgg-reproducibility/experiments/LUNA/figs/section_mouse2_slice229_0_epochs_grid_fixedGT.png



    
![png](output_29_9.png)
    



```python

```
