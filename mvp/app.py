"""AI-Photon MVP: interface local de testes com imagens.

Execute com:  streamlit run app.py
Envie um arquivo de TC (.nii, .nii.gz, .nrrd, .mha) ou aponte um caminho
local (arquivo ou pasta DICOM), escolha o módulo e veja métricas e figuras.
Protótipo de pesquisa, sem validação clínica.
"""

import json
import os
import tempfile

import streamlit as st

from analyze import MODULES, detect_module, load_any
from ai_photon_mvp import pipeline as pl

st.set_page_config(page_title="AI-Photon MVP", page_icon="🩻", layout="wide")
st.title("AI-Photon MVP · teste com imagens")
st.caption("Protótipo de pesquisa do projeto AI-Photon. Não é dispositivo "
           "médico e não deve apoiar decisão clínica.")

fonte = st.radio("Fonte da imagem", ["Enviar arquivo", "Caminho local"],
                 horizontal=True)
path = None
tmp = None
if fonte == "Enviar arquivo":
    up = st.file_uploader("TC em .nii, .nii.gz, .nrrd ou .mha",
                          type=["nii", "gz", "nrrd", "mha", "mhd"])
    if up is not None:
        suffix = up.name[up.name.find("."):]
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(up.read())
        tmp.close()
        path = tmp.name
else:
    path = st.text_input("Caminho do arquivo ou da pasta DICOM") or None

modulo = st.selectbox("Módulo", ["auto", "neuro", "torax", "abdome"])

if path and st.button("Analisar", type="primary"):
    with st.spinner("Processando..."):
        ct, spacing = load_any(path)
        qa = pl.qa_check(ct, spacing)
        mod = modulo if modulo != "auto" else detect_module(ct, spacing)
        outdir = tempfile.mkdtemp(prefix="aiphoton_")
        metrics = MODULES[mod](ct, spacing, outdir, "exame")

    st.subheader(f"Módulo executado: {mod}")
    c1, c2 = st.columns([2, 1])
    with c1:
        for f in sorted(os.listdir(outdir)):
            if f.endswith(".png"):
                st.image(os.path.join(outdir, f))
    with c2:
        st.metric("QA score", f"{qa['score']}/100")
        for a in qa["alerts"]:
            st.warning(a)
        st.json(json.loads(json.dumps(metrics, default=str)))
    if tmp is not None:
        os.unlink(tmp.name)
