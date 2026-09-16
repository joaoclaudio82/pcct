"""Catálogo de TCs públicas e download para o laboratório do MVP."""

import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

SLICER = "https://github.com/Slicer/SlicerTestingData/releases/download/SHA256"
NIIVUE = "https://raw.githubusercontent.com/neurolabusc/niivue-images/main"

UA = "AI-Photon-MVP/0.1 (research prototype; +https://github.com)"

SAMPLES = [
    {
        "id": "ct-chest",
        "nome": "Tórax · 3D Slicer",
        "arquivo": "CTChest.nrrd",
        "modulo": "torax",
        "url": f"{SLICER}/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e",
        "licenca": "3D Slicer SampleData",
        "nota": "TC de tórax usada na avaliação original do MVP.",
    },
    {
        "id": "ct-liver",
        "nome": "Fígado · MSD liver_100",
        "arquivo": "CTLiver.nrrd",
        "modulo": "abdome",
        "url": f"{SLICER}/e16eae0ae6fefa858c5c11e58f0f1bb81834d81b7102e021571056324ef6f37e",
        "licenca": "CC-BY-SA (Medical Segmentation Decathlon)",
        "nota": "Tumor hepático multifocal; caso das avaliações originais.",
    },
    {
        "id": "ct-head-philips",
        "nome": "Crânio · Philips",
        "arquivo": "CT_head.nii.gz",
        "modulo": "neuro",
        "url": f"{NIIVUE}/CT_Philips.nii.gz",
        "licenca": "BSD-2 (Rorden Lab / niivue-images)",
        "nota": "TC de crânio da avaliação de morfometria.",
    },
    {
        "id": "ct-brain-slicer",
        "nome": "Crânio · CT-MR Brain",
        "arquivo": "CT_brain.nrrd",
        "modulo": "neuro",
        "url": f"{SLICER}/6a5b6caccb76576a863beb095e3bfb910c50ca78f4c9bf043aa42f976cfa53d1",
        "licenca": "3D Slicer SampleData (doação)",
        "nota": "Segundo crânio para testar o módulo neuro.",
    },
    {
        "id": "ct-avm",
        "nome": "Crânio · AVM",
        "arquivo": "CT_AVM.nii.gz",
        "modulo": "neuro",
        "url": f"{NIIVUE}/CT_AVM.nii.gz",
        "licenca": "BSD-2 (niivue-images)",
        "nota": "Malformação arteriovenosa; volume pequeno, útil para QA.",
    },
    {
        "id": "ct-electrodes",
        "nome": "Crânio · eletrodos",
        "arquivo": "CT_Electrodes.nii.gz",
        "modulo": "neuro",
        "url": f"{NIIVUE}/CT_Electrodes.nii.gz",
        "licenca": "BSD-2 (niivue-images)",
        "nota": "Pós-cirúrgico com eletrodos; testa robustez da ICV.",
    },
    {
        "id": "ct-abdo-niivue",
        "nome": "Abdome · niivue",
        "arquivo": "CT_Abdo.nii.gz",
        "modulo": "abdome",
        "url": f"{NIIVUE}/CT_Abdo.nii.gz",
        "licenca": "BSD-2 (niivue-images)",
        "nota": "Abdome adicional, mais leve que o caso MSD.",
    },
    {
        "id": "cta-abdomen",
        "nome": "CTA abdome · Panoramix",
        "arquivo": "CTA_Panoramix.nrrd",
        "modulo": "abdome",
        "url": f"{SLICER}/146af87511520c500a3706b7b2bfb545f40d5d04dd180be3a7a2c6940e447433",
        "licenca": "OsiriX DICOM library / 3D Slicer",
        "nota": "Angio-TC abdominal recortada.",
    },
    {
        "id": "cta-cardio",
        "nome": "CTA cardíaca · 3D Slicer",
        "arquivo": "CTA_cardio.nrrd",
        "modulo": "auto",
        "url": f"{SLICER}/3b0d4eb1a7d8ebb0c5a89cc0504640f76a030b4e869e33ff34c564c3d3b88ad2",
        "licenca": "3D Slicer SampleData",
        "nota": "FOV cardíaco; a detecção automática pode cair em tórax ou abdome.",
    },
]

EXTS = (".nii", ".nii.gz", ".nrrd", ".mha", ".mhd")


def path_of(sample):
    return os.path.join(DATA, sample["arquivo"])


def exists(sample):
    p = path_of(sample)
    return os.path.isfile(p) and os.path.getsize(p) > 0


def by_id(sid):
    for s in SAMPLES:
        if s["id"] == sid:
            return s
    raise KeyError(sid)


def local_volumes():
    if not os.path.isdir(DATA):
        return []
    out = []
    for name in sorted(os.listdir(DATA)):
        low = name.lower()
        if low.endswith(EXTS):
            out.append(os.path.join(DATA, name))
    return out


def download_file(url, dest, timeout=600):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as src, open(tmp, "wb") as dst:
        while True:
            chunk = src.read(256 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    os.replace(tmp, dest)
    return dest


def download_sample(sample):
    dest = path_of(sample)
    if exists(sample):
        return dest, False
    download_file(sample["url"], dest)
    return dest, True


def download_url(url, filename=None):
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError("a URL precisa começar com http:// ou https://")
    if filename is None:
        from urllib.parse import unquote, urlparse
        filename = os.path.basename(unquote(urlparse(url).path)) or "exame_baixado.nii.gz"
    low = filename.lower()
    if not low.endswith(EXTS):
        filename += ".nii.gz"
    dest = os.path.join(DATA, filename)
    download_file(url, dest)
    return dest


def download_all(log=print):
    os.makedirs(DATA, exist_ok=True)
    for s in SAMPLES:
        dest = path_of(s)
        if exists(s):
            mb = os.path.getsize(dest) / 1e6
            log(f"já existe  {s['arquivo']:22s}  {mb:6.1f} MB")
            continue
        log(f"baixando   {s['arquivo']}  ←  {s['nome']}")
        download_sample(s)
        mb = os.path.getsize(dest) / 1e6
        log(f"ok         {s['arquivo']:22s}  {mb:6.1f} MB")
    log(f"amostras em {DATA}")


if __name__ == "__main__":
    download_all()
