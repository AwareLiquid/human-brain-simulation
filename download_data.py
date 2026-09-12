"""download_data.py — 下载所有数据到 data/ 目录。

MaleCNS (CC-BY 4.0, Janelia/Google/Cambridge) + MNIST + Fashion-MNIST。
运行: python download_data.py
"""
import urllib.request
import os
import time


def fetch(url, out, retries=6):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"  skip (exists): {out}")
        return
    for i in range(retries):
        try:
            print(f"  downloading: {os.path.basename(out)} ...", flush=True)
            urllib.request.urlretrieve(url, out)
            sz = os.path.getsize(out)
            if sz > 1000:
                print(f"    -> {sz/1e6:.1f} MB")
                return
        except Exception as e:
            print(f"    retry {i+1}: {e}", flush=True)
            time.sleep(3)
    print(f"  FAILED: {out}")


GCS = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
TF = "https://storage.googleapis.com/tensorflow/tf-keras-datasets"
FASHION = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"

# MaleCNS connectome (CC-BY 4.0)
fetch(f"{GCS}/connectome-weights-male-cns-v1.0-minconf-0.5.feather",
      "data/connectome-weights-male-cns-v1.0-minconf-0.5.feather")
fetch(f"{GCS}/body-neurotransmitters-male-cns-v1.0.feather",
      "data/body-neurotransmitters-male-cns-v1.0.feather")
fetch(f"{GCS}/body-annotations-male-cns-v1.0-minconf-0.5.feather",
      "data/body-annotations-male-cns-v1.0-minconf-0.5.feather")

# MNIST
fetch(f"{TF}/mnist.npz", "data/mnist.npz")

# Fashion-MNIST
fetch(f"{FASHION}/train-images-idx3-ubyte.gz", "data/fashion/train-images-idx3-ubyte.gz")
fetch(f"{FASHION}/train-labels-idx1-ubyte.gz", "data/fashion/train-labels-idx1-ubyte.gz")
fetch(f"{FASHION}/t10k-images-idx3-ubyte.gz", "data/fashion/t10k-images-idx3-ubyte.gz")
fetch(f"{FASHION}/t10k-labels-idx1-ubyte.gz", "data/fashion/t10k-labels-idx1-ubyte.gz")

print("\nDone. Data ready in data/")
