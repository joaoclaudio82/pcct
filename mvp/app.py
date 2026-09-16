"""AI-Photon MVP: laboratório local para analisar TCs.

Execute com:  streamlit run app.py
Protótipo de pesquisa, sem validação clínica.
"""

import os
import tempfile

import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze import MODULES, detect_module, load_any
from ai_photon_mvp import alzheimer as az
from ai_photon_mvp import pipeline as pl
from demos import run_alzheimer, run_longitudinal, run_phantom
from samples import DATA, SAMPLES, download_all, download_sample, download_url, exists, local_volumes, path_of

st.set_page_config(page_title="AI-Photon laboratório", page_icon="🩻", layout="wide")
st.title("AI-Photon · laboratório")
st.caption("Protótipo de pesquisa. Não é dispositivo médico e não deve apoiar decisão clínica.")

WINDOWS = {
    "neuro": (0, 80),
    "torax": (-1000, 300),
    "abdome": (-150, 250),
}


@st.cache_resource(show_spinner="Carregando volume…")
def _load(path):
    return load_any(path)


def _slice_uint8(vol, z, vmin, vmax):
    s = np.clip((vol[z] - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    return (s * 255).astype(np.uint8)


def _render_metrics(mod, metrics, qa):
    c1, c2, c3 = st.columns(3)
    c1.metric("QA", f"{qa['score']}/100")
    c2.metric("Shape", "×".join(str(s) for s in qa["shape"]))
    c3.metric("Spacing (mm)", " ".join(f"{s:.2f}" for s in qa["spacing"]))
    for a in qa.get("alerts") or []:
        st.warning(a)

    if mod == "neuro":
        a, b, c, d = st.columns(4)
        a.metric("ICV (mL)", metrics.get("icv_mL"))
        b.metric("Ventrículos (mL)", metrics.get("ventricles_mL"))
        c.metric("Fração de líquor (%)", metrics.get("csf_fraction_pct"))
        d.metric("Evans-like", metrics.get("evans_like_index"))
        ic = metrics.get("ventricles_IC_mL")
        if ic:
            st.caption(f"Intervalo de ventrículos por perturbação da máscara: [{ic[0]}; {ic[1]}] mL")
    elif mod == "torax":
        a, b, c = st.columns(3)
        a.metric("Pulmões (L)", metrics.get("volume_L"))
        b.metric("HU médio", metrics.get("mean_hu"))
        c.metric("Ênfisema (%)", metrics.get("emphysema_pct"))
        nods = metrics.get("candidatos_nodulo") or []
        if nods:
            st.subheader("Candidatos a nódulo")
            st.dataframe(nods, width="stretch", hide_index=True)
        else:
            st.info("Nenhum candidato a nódulo no filtro atual.")
    elif mod == "abdome":
        a, b = st.columns(2)
        a.metric("Fígado (mL)", metrics.get("figado_mL"))
        b.metric("Carga hipodensa (mL)", metrics.get("carga_tumoral_mL"))
        les = metrics.get("lesoes_hipodensas") or []
        if les:
            st.subheader("Lesões hipodensas")
            st.dataframe(les, width="stretch", hide_index=True)
        else:
            st.info("Nenhuma lesão hipodensa detectada.")
    else:
        st.json(metrics)


def _run_analyze(path, modulo):
    ct, spacing = _load(path)
    qa = pl.qa_check(ct, spacing)
    mod = modulo if modulo != "auto" else detect_module(ct, spacing)
    outdir = tempfile.mkdtemp(prefix="aiphoton_")
    name = os.path.splitext(os.path.basename(path.rstrip("/")))[0].replace(".nii", "")
    metrics = MODULES[mod](ct, spacing, outdir, name)
    st.session_state["resultado"] = {
        "path": path, "mod": mod, "qa": qa, "metrics": metrics,
        "outdir": outdir, "ct": ct, "spacing": spacing,
    }


def _show_resultado():
    res = st.session_state.get("resultado")
    if not res:
        return
    st.subheader(f"Módulo executado: {res['mod']}")
    imgs = [os.path.join(res["outdir"], f) for f in sorted(os.listdir(res["outdir"]))
            if f.endswith(".png")]
    left, right = st.columns([2, 1])
    with left:
        for p in imgs:
            st.image(p, width="stretch")
        ct = res["ct"]
        vmin, vmax = WINDOWS.get(res["mod"], (-200, 400))
        z = st.slider("Corte axial", 0, int(ct.shape[0]) - 1, int(ct.shape[0]) // 2)
        st.image(_slice_uint8(ct, z, vmin, vmax), caption=f"z={z}  janela [{vmin}, {vmax}] HU",
                 width="stretch")
    with right:
        _render_metrics(res["mod"], res["metrics"], res["qa"])


def pagina_exame():
    st.subheader("Analisar um exame")
    fonte = st.radio(
        "Fonte",
        ["Amostras públicas", "Enviar arquivo", "URL da internet", "Caminho local"],
        horizontal=True,
    )
    path = None
    modulo_sugerido = st.session_state.get("modulo_sugerido", "auto")
    catalog = None

    if fonte == "Amostras públicas":
        st.markdown("Volumes de TC com licença aberta, gravados em `mvp/data/`.")
        nomes = [s["nome"] for s in SAMPLES]
        atual = st.session_state.get("exame")
        default_i = 0
        if atual:
            for i, s in enumerate(SAMPLES):
                if path_of(s) == atual:
                    default_i = i
                    break
        escolha = st.selectbox("Amostra", nomes, index=default_i)
        sample = next(s for s in SAMPLES if s["nome"] == escolha)
        b1, b2 = st.columns(2)
        with b1:
            if not exists(sample):
                if st.button("Baixar esta amostra"):
                    with st.spinner(f"Baixando {sample['arquivo']}…"):
                        download_sample(sample)
                    st.rerun()
        with b2:
            if st.button("Baixar todas as que faltam"):
                with st.spinner("Baixando amostras públicas…"):
                    download_all(log=lambda m: st.write(m))
                st.rerun()
        st.caption(f"{sample['nota']}  ·  `{sample['arquivo']}`  ·  {sample['licenca']}  ·  "
                   f"{'no disco' if exists(sample) else 'ausente'}")
        if exists(sample):
            path = path_of(sample)
            modulo_sugerido = sample["modulo"]
        catalog = [
            {"exame": s["nome"], "arquivo": s["arquivo"], "módulo": s["modulo"],
             "status": "no disco" if exists(s) else "ausente"} for s in SAMPLES
        ]

    elif fonte == "Enviar arquivo":
        up = st.file_uploader(
            "TC em .nii, .nii.gz, .nrrd, .mha",
            type=["nii", "gz", "nrrd", "mha", "mhd"],
        )
        if up is not None:
            suffix = up.name[up.name.find("."):]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.read())
            tmp.close()
            path = tmp.name

    elif fonte == "URL da internet":
        url = st.text_input(
            "URL direta do arquivo (.nii.gz, .nrrd, .mha)",
            placeholder="https://…/exame.nii.gz",
        )
        nome = st.text_input("Nome local (opcional)", placeholder="meu_exame.nii.gz")
        if st.button("Baixar URL") and url:
            try:
                with st.spinner("Baixando…"):
                    path = download_url(url, nome or None)
                st.session_state["exame"] = path
                st.success(f"Salvo em `{path}`")
            except Exception as e:
                st.error(str(e))
        path = st.session_state.get("exame") if path is None else path
        extras = local_volumes()
        if extras:
            st.caption("Já baixados em data/: " + ", ".join(os.path.basename(p) for p in extras))

    else:
        path = st.text_input("Arquivo ou pasta DICOM") or None

    if path and os.path.exists(path):
        st.success(f"Selecionado: `{os.path.basename(path)}`")
        opts = ["auto", "neuro", "torax", "abdome"]
        default_mod = modulo_sugerido if modulo_sugerido in opts else "auto"
        modulo = st.selectbox("Módulo", opts, index=opts.index(default_mod))
        if st.button("Analisar", type="primary"):
            with st.spinner("Processando…"):
                _run_analyze(path, modulo)

    _show_resultado()
    if catalog:
        with st.expander("Catálogo de amostras"):
            st.dataframe(catalog, width="stretch", hide_index=True)


def pagina_phantom():
    st.subheader("Phantom espectral")
    st.markdown(
        "Simula duas VMI (50 e 70 keV) de um abdome digital com insertos de iodo "
        "conhecido e recupera o mapa em mg/mL."
    )
    dose = st.select_slider("Fração da dose", options=[1.0, 0.5, 0.25, 0.125], value=1.0)
    if st.button("Simular e decompor", type="primary"):
        outdir = tempfile.mkdtemp(prefix="aiphoton_ph_")
        with st.spinner("Gerando phantom e decompondo…"):
            st.session_state["phantom"] = run_phantom(outdir, dose_factor=float(dose))
    ph = st.session_state.get("phantom")
    if not ph:
        return
    st.image(ph["figura"], width="stretch")
    st.dataframe(ph["acuracia"], width="stretch", hide_index=True)
    err = max(abs(r["erro_pct"]) for r in ph["acuracia"])
    st.metric("Maior erro relativo", f"{err:.1f}%")


def pagina_longitudinal():
    st.subheader("Acompanhamento longitudinal")
    st.markdown(
        "Exame basal e reavaliação após tratamento simulado: registro rígido, "
        "segmentação no mapa de iodo e pareamento lesão a lesão."
    )
    if st.button("Rodar demo", type="primary"):
        outdir = tempfile.mkdtemp(prefix="aiphoton_lg_")
        with st.spinner("Simulando basal, reavaliação e registro…"):
            st.session_state["longit"] = run_longitudinal(outdir)
    lg = st.session_state.get("longit")
    if not lg:
        return
    st.image(lg["figura"], width="stretch")
    a, b, c = st.columns(3)
    a.metric("Erro de registro (voxel)", lg["registro"]["erro_vox"])
    b.metric("Q basal (mg)", lg["carga"]["Q_basal_mg"])
    c.metric("ΔQ", f"{lg['carga']['delta_Q_pct']:+.1f}%")
    st.dataframe(lg["tabela_lesoes"], width="stretch", hide_index=True)
    st.caption("Verdade simulada: L1 completa, L2 e L3 parciais, L4 progressão.")


def pagina_alzheimer():
    st.subheader("Alzheimer · teste PCCT (Demonstrador 4)")
    st.markdown("""
Esta aba **não diagnostica Alzheimer**. Amiloide e tau são PET ou líquor.
O que a proposta pede ao PCCT é outra coisa: **medir atrofia em TC de crânio**
(ventrículos, líquor, Evans) com menos ruído do que a TC convencional, para
triagem e acompanhamento.

Como não temos exame de NAEOTOM, o teste PCCT é um **cérebro digital** com
atrofia conhecida, fotografado duas vezes:
- **TC convencional (EID):** mais ruído, pouco contraste cinzenta/branca
- **PCCT simulado:** menos ruído eletrônico, melhor contraste

Depois o mesmo algoritmo de morfometria tenta recuperar o índice de Evans e o
volume ventricular. O PCCT deve errar menos. Em paralelo, você pode rodar a
**triagem oportunista** num crânio público (TC comum, não espectral).
""")
    modo = st.radio(
        "O que testar",
        ["Phantom: PCCT vs TC convencional", "Triagem em crânio real"],
        horizontal=True,
    )

    if modo.startswith("Phantom"):
        fen = st.selectbox(
            "Fenótipo do cérebro digital",
            ["atrofia_alzheimer", "controle"],
            format_func=lambda x: "atrofia tipo Alzheimer (ventrículos grandes)"
            if x == "atrofia_alzheimer" else "controle (ventrículos pequenos)",
        )
        if st.button("Rodar teste PCCT", type="primary"):
            outdir = tempfile.mkdtemp(prefix="aiphoton_az_")
            with st.spinner("Simulando EID e PCCT e recuperando a morfometria…"):
                st.session_state["alzheimer"] = run_alzheimer(outdir, phenotype=fen)
        azr = st.session_state.get("alzheimer")
        if not azr:
            return
        st.info(azr["o_que_e"])
        st.image(azr["figura"], width="stretch")
        t, e, p = azr["truth"], azr["eid"], azr["pcct"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Evans verdadeiro", t["evans_like_index"])
        c2.metric("Evans PCCT", p["evans_like"])
        c3.metric("Evans TC conv.", e["evans_like"])
        c4.metric("CNR GM/WM (PCCT / EID)", f"{p['cnr_gm_wm']} / {e['cnr_gm_wm']}")
        st.caption(
            f"Erro no Evans: PCCT {p['erro_evans_pct']}% · TC convencional {e['erro_evans_pct']}%. "
            f"Erro no volume ventricular: PCCT {p['erro_ventriculos_pct']}% · "
            f"TC convencional {e['erro_ventriculos_pct']}%."
        )
        st.dataframe(
            [
                {"scanner": "verdade", **{k: t[k] for k in
                    ("ventricles_mL", "evans_like_index", "vent_icv_pct")}},
                {"scanner": "PCCT", "ventricles_mL": p["ventricles_mL"],
                 "evans_like_index": p["evans_like"],
                 "erro_ventriculos_pct": p["erro_ventriculos_pct"],
                 "erro_evans_pct": p["erro_evans_pct"], "cnr_gm_wm": p["cnr_gm_wm"]},
                {"scanner": "TC convencional", "ventricles_mL": e["ventricles_mL"],
                 "evans_like_index": e["evans_like"],
                 "erro_ventriculos_pct": e["erro_ventriculos_pct"],
                 "erro_evans_pct": e["erro_evans_pct"], "cnr_gm_wm": e["cnr_gm_wm"]},
            ],
            width="stretch", hide_index=True,
        )
        return

    heads = [s for s in SAMPLES if s["modulo"] == "neuro"]
    nomes = [s["nome"] for s in heads]
    escolha = st.selectbox("Crânio público", nomes)
    sample = next(s for s in heads if s["nome"] == escolha)
    if not exists(sample):
        st.warning("Amostra ausente. Baixe na aba Exame real ou rode `./download_data.sh`.")
        return
    st.caption(f"{sample['nota']} · {sample['licenca']}")
    if st.button("Triagem morfométrica", type="primary"):
        with st.spinner("Segmentando ICV, ventrículos e calculando Evans…"):
            ct, spacing = _load(path_of(sample))
            st.session_state["az_exam"] = az.analyze_exam(ct, spacing)
            st.session_state["az_exam_ct"] = ct
    ex = st.session_state.get("az_exam")
    if not ex:
        return
    m, tri = ex["metrics"], ex["triagem"]
    st.warning(tri["aviso"])
    st.markdown(f"**Leitura do protótipo:** {tri['nivel']}")
    for s in tri["sinais"]:
        st.write(f"- {s}")
    a, b, c, d = st.columns(4)
    a.metric("ICV (mL)", m.get("icv_mL"))
    b.metric("Ventrículos (mL)", m.get("ventricles_mL"))
    c.metric("Evans-like", m.get("evans_like_index"))
    d.metric("Cálcio intracraniano (mL)", m.get("calcium_intracraniano_mL"))
    ct = st.session_state["az_exam_ct"]
    ms = ex["masks"]
    zc = int(np.argmax(ms["ventricles"].sum(axis=(1, 2)))) if ms["ventricles"].any() \
        else ct.shape[0] // 2
    z = st.slider("Corte axial", 0, int(ct.shape[0]) - 1, zc, key="az_z")
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(ct[z], cmap="gray", vmin=0, vmax=80)
    if ms["icv"][z].any():
        ax.contour(ms["icv"][z], colors="lime", linewidths=0.8)
    if ms["ventricles"][z].any():
        ax.contour(ms["ventricles"][z], colors="deepskyblue", linewidths=1.1)
    ax.set_title("verde = ICV · azul = ventrículos")
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


os.makedirs(DATA, exist_ok=True)
tab1, tab2, tab3, tab4 = st.tabs(
    ["Exame real", "Phantom espectral", "Longitudinal", "Alzheimer / PCCT"])
with tab1:
    pagina_exame()
with tab2:
    pagina_phantom()
with tab3:
    pagina_longitudinal()
with tab4:
    pagina_alzheimer()
