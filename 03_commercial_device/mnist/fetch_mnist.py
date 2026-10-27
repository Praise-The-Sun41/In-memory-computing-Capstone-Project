"""
fetch_mnist.py —— download the four official MNIST idx.gz files into this directory.

    cd md-python/mnist
    python fetch_mnist.py

Afterwards point example_digits.py / example_digits_softmax.py to this directory with `--mnist mnist`
(an existing .npz with keys Xtr/ytr/Xte/yte is also accepted).

yann.lecun.com now often returns 403, so a GitHub mirror is used by default;
--url can switch to another source.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request

FILES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz":  "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz":  "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
DEFAULT_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"


def main():
    ap = argparse.ArgumentParser(description="Download MNIST into the current directory")
    ap.add_argument("--url", default=DEFAULT_URL, help="Mirror URL prefix")
    ap.add_argument("--no-verify", action="store_true", help="Skip SHA256 verification")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    for name, sha in FILES.items():
        path = os.path.join(here, name)
        if os.path.exists(path):
            print(f"  {name} already present, skipping")
            continue
        url = args.url.rstrip("/") + "/" + name
        print(f"  downloading {name} ...", end="", flush=True)
        urllib.request.urlretrieve(url, path)
        got = hashlib.sha256(open(path, "rb").read()).hexdigest()
        if not args.no_verify and got != sha:
            os.remove(path)
            raise SystemExit(f"\n  ✗ {name} failed verification ({got[:12]}...), deleted. "
                             f"Try another --url.")
        print(f" {os.path.getsize(path)/1e6:.1f} MB ✓")
    print("\nDone. You can now run:")
    print("  python example_digits_softmax.py --mnist mnist --n-test 2000")
    print("  python example_digits.py --mnist mnist --n-test 1000 --bits 4 --bias --tol 1")


if __name__ == "__main__":
    main()
