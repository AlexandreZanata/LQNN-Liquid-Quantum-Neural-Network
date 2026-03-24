# Scientific principle: a fixed open-source baseline is needed to compare
# dynamic-topology behavior against a reproducible external reference model.

from __future__ import annotations

from pathlib import Path
import urllib.request

from tqdm import tqdm


MODEL_FILES = {
    "config.json": "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/config.json",
    "tokenizer_config.json": "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/tokenizer_config.json",
    "vocab.json": "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/vocab.json",
    "merges.txt": "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/merges.txt",
    "pytorch_model.bin": "https://huggingface.co/sshleifer/tiny-gpt2/resolve/main/pytorch_model.bin",
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination.name} ja existe")
        return

    with urllib.request.urlopen(url) as response:  # nosec B310
        total = int(response.headers.get("Content-Length", 0))
        with destination.open("wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as pbar:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    target_dir = Path("models/open_source/tiny-gpt2")
    print(f"Baixando modelo open source para: {target_dir}")

    for file_name, url in MODEL_FILES.items():
        download_file(url, target_dir / file_name)

    print("\nDownload concluido.")
    print("Arquivos prontos para testes de baseline em experiments/vs_mlp.py")


if __name__ == "__main__":
    main()
