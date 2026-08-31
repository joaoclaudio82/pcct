"""Gera o dashboard HTML do MVP AI-Photon a partir dos resultados."""

import base64
import json

R = "results"


def img64(name):
    with open(f"{R}/{name}", "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


ph = json.load(open(f"{R}/phantom_metrics.json"))
lg = json.load(open(f"{R}/longitudinal_metrics.json"))
rc = json.load(open(f"{R}/real_ct_metrics.json"))

acc = ph["acuracia"]
sweep = ph["dose_sweep"]
carga = lg["carga"]
tor = rc["torax"]
abd = rc["abdome"]

max_err = max(abs(r["erro_pct"]) for r in acc)
err_low_dose = sweep["erro_Q_pct"][-1]

acc_rows = "".join(
    f"<tr><td>{r['lesao']}</td><td>{r['diametro_mm']:.0f}</td>"
    f"<td>{r['conc_verdadeira']:.1f}</td><td>{r['conc_estimada']:.2f}</td>"
    f"<td class='{'ok' if abs(r['erro_pct'])<5 else 'warn'}'>{r['erro_pct']:+.1f}%</td>"
    f"<td>{r['Q_verdadeiro_mg']}</td><td>{r['Q_estimado_mg']}</td></tr>"
    for r in acc)

sweep_rows = "".join(
    f"<tr><td>{100*d:.1f}%</td><td>{sweep['ruido_hu50'][i]}</td>"
    f"<td class='ok'>{sweep['erro_Q_pct'][i]:.2f}%</td>"
    f"<td class='warn'>{sweep['erro_Q_pct_denoised'][i]:.2f}%</td></tr>"
    for i, d in enumerate(sweep["dose"]))

verdade = {1: "resposta parcial (L3)", 2: "progressão (L4)",
           4: "resposta parcial (L2)", 3: "resposta completa (L1)"}
les_rows = ""
for t in lg["tabela_lesoes"]:
    reav = t["lesao_reav"] if t["lesao_reav"] else "não detectada"
    cls = "ok" if t["delta_Q_pct"] < 0 else "bad"
    les_rows += (f"<tr><td>{t['lesao_basal']}</td><td>{reav}</td>"
                 f"<td>{t['vol_basal_mL']}</td><td>{t['vol_reav_mL']}</td>"
                 f"<td>{t['Q_basal_mg']}</td><td>{t['Q_reav_mg']}</td>"
                 f"<td class='{cls}'>{t['delta_Q_pct']:+.1f}%</td>"
                 f"<td>{verdade.get(t['lesao_basal'],'')}</td></tr>")

nod_rows = "".join(
    f"<tr><td>{i+1}</td><td>{c['diam_eq_mm']}</td><td>{c['volume_mL']}</td>"
    f"<td>{c['mean_hu']}</td><td>{c['elongation']}</td></tr>"
    for i, c in enumerate(tor["candidatos_nodulo"][:6]))

hep_rows = "".join(
    f"<tr><td>{i+1}</td><td>{c['diam_eq_mm']}</td><td>{c['volume_mL']}</td>"
    f"<td>[{c['volume_IC_mL'][0]}; {c['volume_IC_mL'][1]}]</td>"
    f"<td>{c['mean_hu']}</td></tr>"
    for i, c in enumerate(abd["lesoes_hipodensas"][:6]))

html = f"""<title>AI-Photon MVP</title>
<style>
:root {{
  --paper:#F6F8FA; --surface:#FFFFFF; --ink:#1B2B3A; --muted:#5A6E80;
  --line:#D8E0E8; --accent:#1F4E79; --accent-soft:#E8EFF6;
  --iodo:#C2501F; --ok:#22694B; --bad:#A63A22; --warn:#8A5A18;
  --mono:'JetBrains Mono',ui-monospace,monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --paper:#0F1822; --surface:#16222F; --ink:#DFE8F1; --muted:#8FA3B5;
    --line:#2A3947; --accent:#7FB0DC; --accent-soft:#1C2E40;
    --iodo:#E58B5A; --ok:#6CC29A; --bad:#E08A72; --warn:#D8AC5F;
  }}
}}
:root[data-theme="dark"] {{
  --paper:#0F1822; --surface:#16222F; --ink:#DFE8F1; --muted:#8FA3B5;
  --line:#2A3947; --accent:#7FB0DC; --accent-soft:#1C2E40;
  --iodo:#E58B5A; --ok:#6CC29A; --bad:#E08A72; --warn:#D8AC5F;
}}
* {{ box-sizing:border-box; }}
body {{ background:var(--paper); color:var(--ink); margin:0;
  font-family:'Source Sans 3',system-ui,sans-serif; font-size:16px; line-height:1.55; }}
.wrap {{ max-width:1060px; margin:0 auto; padding:40px 24px 64px; }}
header h1 {{ font-family:'Source Serif 4',Georgia,serif; font-size:2.1rem;
  margin:0 0 4px; color:var(--accent); text-wrap:balance; }}
header p.sub {{ margin:0 0 14px; color:var(--muted); max-width:65ch; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-bottom:28px; }}
.chip {{ font-size:.78rem; letter-spacing:.04em; text-transform:uppercase;
  padding:3px 10px; border:1px solid var(--line); border-radius:999px;
  color:var(--muted); background:var(--surface); }}
.chip.alert {{ color:var(--iodo); border-color:var(--iodo); }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
  gap:12px; margin-bottom:36px; }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:8px;
  padding:14px 16px; }}
.kpi .v {{ font-family:var(--mono); font-size:1.45rem; font-weight:600;
  color:var(--accent); font-variant-numeric:tabular-nums; }}
.kpi .l {{ font-size:.8rem; color:var(--muted); margin-top:2px; }}
section {{ margin-bottom:44px; }}
h2 {{ font-family:'Source Serif 4',Georgia,serif; font-size:1.35rem;
  color:var(--accent); border-bottom:2px solid var(--accent);
  padding-bottom:6px; margin:0 0 6px; }}
h2 .tag {{ font-family:var(--mono); font-size:.72rem; color:var(--muted);
  font-weight:400; margin-left:8px; letter-spacing:.05em; }}
section > p {{ max-width:70ch; color:var(--ink); }}
figure {{ margin:16px 0; background:var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:10px; overflow-x:auto; }}
figure img {{ max-width:100%; display:block; margin:0 auto; border-radius:4px; }}
figcaption {{ font-size:.82rem; color:var(--muted); padding:8px 6px 2px; }}
.tbl {{ overflow-x:auto; }}
table {{ border-collapse:collapse; width:100%; background:var(--surface);
  border:1px solid var(--line); border-radius:8px; font-size:.9rem; }}
th {{ text-align:left; font-size:.75rem; letter-spacing:.05em; text-transform:uppercase;
  color:var(--muted); padding:9px 12px; border-bottom:2px solid var(--line); }}
td {{ padding:8px 12px; border-bottom:1px solid var(--line);
  font-family:var(--mono); font-size:.85rem; font-variant-numeric:tabular-nums; }}
td:first-child {{ font-family:'Source Sans 3',system-ui,sans-serif; }}
tr:last-child td {{ border-bottom:none; }}
.ok {{ color:var(--ok); font-weight:600; }}
.bad {{ color:var(--bad); font-weight:600; }}
.warn {{ color:var(--warn); font-weight:600; }}
.note {{ background:var(--accent-soft); border-left:3px solid var(--accent);
  padding:12px 16px; border-radius:0 8px 8px 0; max-width:75ch; font-size:.92rem; }}
.note b {{ color:var(--accent); }}
footer {{ border-top:1px solid var(--line); padding-top:18px; font-size:.82rem;
  color:var(--muted); max-width:75ch; }}
footer a {{ color:var(--accent); }}
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap">
<div class="wrap">
<header>
<h1>AI-Photon MVP</h1>
<p class="sub">Resultados iniciais do protótipo de quantificação espectral e acompanhamento
oncológico descrito na proposta AI-Photon (UECE &times; Siemens Healthineers).
Cinco avaliações: exatidão do iodo em phantom espectral, redução de dose,
acompanhamento longitudinal e pipeline anatômico em duas imagens públicas reais de TC.</p>
<div class="chips">
<span class="chip">MVP 1 · Auto-Quant</span>
<span class="chip">phantom espectral</span>
<span class="chip">TC públicas reais</span>
<span class="chip alert">protótipo de pesquisa · sem validação clínica</span>
</div>
</header>

<div class="kpis">
<div class="kpi"><div class="v">&le;{max_err:.1f}%</div><div class="l">erro de concentração de iodo, dose plena (lesões 8–24 mm)</div></div>
<div class="kpi"><div class="v">{err_low_dose:.2f}%</div><div class="l">erro no biomarcador Q com 12,5% da dose</div></div>
<div class="kpi"><div class="v">0,0 vox</div><div class="l">erro do registro rígido (correlação de fase)</div></div>
<div class="kpi"><div class="v">4 &rarr; 3</div><div class="l">lesões rastreadas; 1 resposta completa detectada</div></div>
<div class="kpi"><div class="v">{tor['pulmoes']['volume_L']:.2f} L</div><div class="l">volume pulmonar na TC de tórax pública</div></div>
<div class="kpi"><div class="v">{abd['carga_tumoral_mL']:.0f} mL</div><div class="l">carga tumoral hepática na TC pública (IC por segmentação)</div></div>
</div>

<section>
<h2>A · Exatidão da quantificação espectral<span class="tag">phantom digital</span></h2>
<p>Phantom abdominal com aorta (8 mg/mL) e quatro lesões de concentração conhecida.
Duas VMI (50 e 70 keV) são simuladas com ruído realista e a decomposição de
materiais recupera o mapa de iodo. O erro fica abaixo de {max_err:.1f}% em todas
as estruturas, incluindo a lesão de 8 mm.</p>
<figure><img src="{img64('phantom_maps.png')}" alt="VMI e mapas de iodo">
<figcaption>VMI simuladas e mapa de iodo verdadeiro vs. estimado (corte axial).</figcaption></figure>
<div class="tbl"><table>
<tr><th>estrutura</th><th>diâmetro (mm)</th><th>C verdadeira (mg/mL)</th><th>C estimada</th><th>erro</th><th>Q verdadeiro (mg)</th><th>Q estimado</th></tr>
{acc_rows}
</table></div>
</section>

<section>
<h2>B · Redução de dose com preservação do biomarcador<span class="tag">12 realizações por nível</span></h2>
<p>O ruído cresce com 1/&radic;dose e, ainda assim, o biomarcador integrado
Q<sub>&Omega;</sub> da lesão de 24 mm permanece com erro médio abaixo de 1%
até 12,5% da dose, porque a integração espacial cancela ruído de média zero.
O denoising TV melhora a aparência, mas introduz viés sistemático de 1–2%
no iodo: exatamente o risco que a proposta trata como
<b>erro silencioso de quantificação</b>.</p>
<figure><img src="{img64('dose_sweep.png')}" alt="Erro vs dose">
<figcaption>Erro no biomarcador de iodo vs. dose relativa, com e sem denoising.</figcaption></figure>
<div class="tbl"><table>
<tr><th>dose relativa</th><th>ruído a 50 keV (HU)</th><th>erro em Q (sem denoising)</th><th>erro em C (com denoising TV)</th></tr>
{sweep_rows}
</table></div>
</section>

<section>
<h2>C · Acompanhamento longitudinal adaptativo<span class="tag">basal &rarr; reavaliação</span></h2>
<p>Exame de reavaliação simulado com deslocamento rígido de paciente, lesões
alteradas por "tratamento" e novas realizações de ruído. O pipeline registra
os exames (erro de {lg['registro']['erro_vox']} voxel), segmenta as lesões no
mapa de iodo, pareia lesão a lesão e recupera o padrão verdadeiro de resposta:
resposta completa detectada como desaparecimento, duas respostas parciais e
uma progressão.</p>
<figure><img src="{img64('longitudinal.png')}" alt="Longitudinal">
<figcaption>Mapas de iodo basal e de reavaliação (registrado) e mapa de mudança.</figcaption></figure>
<div class="tbl"><table>
<tr><th>lesão basal</th><th>reavaliação</th><th>vol basal (mL)</th><th>vol reav (mL)</th><th>Q basal (mg)</th><th>Q reav (mg)</th><th>&Delta;Q</th><th>verdade simulada</th></tr>
{les_rows}
</table></div>
<p class="note"><b>Carga total:</b> Q passou de {carga['Q_basal_mg']} mg para
{carga['Q_reav_mg']} mg ({carga['delta_Q_pct']:+.1f}%,
IC95% [{carga['IC95_delta_Q_pct'][0]}; {carga['IC95_delta_Q_pct'][1]}] por
realizações de ruído): resposta mista dominada pela lesão em progressão,
o tipo de leitura que o RECIST dimensional tende a diluir.</p>
</section>

<section>
<h2>D · TC de tórax pública<span class="tag">{tor['fonte']}</span></h2>
<p>Pipeline anatômico em imagem real: QA score {tor['qa']['score']}/100
(alerta de HU de preenchimento fora da faixa, típico de FOV circular),
segmentação pulmonar de {tor['pulmoes']['volume_L']:.2f} L
(IC por perturbação de segmentação [{tor['pulmoes']['volume_IC_L'][0]};
{tor['pulmoes']['volume_IC_L'][1]}] L), densidade média
{tor['pulmoes']['mean_hu']:.0f} HU, índice de enfisema
{tor['pulmoes']['emphysema_pct']:.1f}% e {len(tor['candidatos_nodulo'])}
candidatos a nódulo por regras clássicas de tamanho e esfericidade.</p>
<figure><img src="{img64('chest_eval.png')}" alt="Tórax">
<figcaption>Axial, contorno pulmonar e reconstrução coronal com candidatos.</figcaption></figure>
<figure><img src="{img64('chest_nodules.png')}" alt="Candidatos a nódulo"></figure>
<div class="tbl"><table>
<tr><th>candidato</th><th>diâmetro eq. (mm)</th><th>volume (mL)</th><th>HU média</th><th>elongação</th></tr>
{nod_rows}
</table></div>
</section>

<section>
<h2>E · TC de abdome pública com tumor hepático<span class="tag">{abd['fonte']}</span></h2>
<p>Caso real com lesão hepática do Medical Segmentation Decathlon. QA score
{abd['qa']['score']}/100, fígado aproximado de {abd['figado_mL']:.0f} mL por
morfologia clássica e {len(abd['lesoes_hipodensas'])} lesões hipodensas
detectadas, com carga tumoral total de {abd['carga_tumoral_mL']} mL. A maior
lesão ({abd['lesoes_hipodensas'][0]['volume_mL']} mL) corresponde ao tumor
multifocal visível no exame.</p>
<figure><img src="{img64('liver_eval.png')}" alt="Abdome">
<figcaption>Axial, contorno hepático e lesões hipodensas detectadas.</figcaption></figure>
<div class="tbl"><table>
<tr><th>lesão</th><th>diâmetro eq. (mm)</th><th>volume (mL)</th><th>IC volume (mL)</th><th>HU média</th></tr>
{hep_rows}
</table></div>
</section>

<section>
<h2>Limitações e próximos passos</h2>
<p>O phantom espectral opera em nível de imagem reconstruída (sem modelo de
projeções por bin de energia), as segmentações em TC real usam regras
clássicas sem aprendizado e não há, ainda, dados espectrais reais de PCCT:
nenhum resultado aqui é evidência clínica. Próximos passos naturais, na ordem
dos work packages da proposta: substituir as segmentações clássicas por
nnU-Net treinada nos conjuntos públicos (MSD, LIDC-IDRI, KiTS), incluir
simulação por bin de energia com espalhamento, ancorar a calibração em
phantom físico e iniciar o inventário de dados retrospectivos de PCCT com os
parceiros.</p>
</section>

<footer>
Imagens públicas: CTChest (3D Slicer SampleData) e CTLiver, caso liver_100 do
<a href="http://medicaldecathlon.com/">Medical Segmentation Decathlon</a>
(CC-BY-SA), redistribuído pelo projeto 3D Slicer. Protótipo de pesquisa do
projeto AI-Photon; não é dispositivo médico e não deve apoiar decisão clínica.
</footer>
</div>
"""

with open(f"{R}/ai_photon_mvp_dashboard.html", "w") as f:
    f.write(html)
print("html:", len(html) / 1e6, "MB")
