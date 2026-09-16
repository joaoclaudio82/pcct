# AI-Photon

Plataforma de Inteligência Artificial para imagem espectral quantitativa em tomografia computadorizada por contagem de fótons (PCCT) aplicada à oncologia e à neurodegeneração.

Este repositório reúne dois componentes:

1. **Proposta de P&D** (`proposta/`): documento completo de pesquisa aplicada e inovação tecnológica, com contexto da tecnologia PCCT, problema de pesquisa, metodologia em 12 etapas, work packages, cronograma de 30 meses, MVPs, plano de validação, aspectos éticos e regulatórios (LGPD, CEP/CONEP, RDC 657/2022) e rota de transferência tecnológica.
2. **MVP inicial** (`mvp/`): protótipo funcional do MVP 1 da proposta, com simulador espectral, decomposição de materiais, quantificação de iodo, acompanhamento longitudinal e pipeline anatômico avaliado em imagens públicas reais de TC.

## Visão central

Quantificar, caracterizar, prever e acompanhar a doença oncológica e neurodegenerativa a partir do dado espectral nativo do PCCT, combinando imagens multienergéticas, dados clínicos, física de imagem e IA, em um ciclo contínuo:

```
Quantificar -> Prever -> Tratar -> Medir -> Aprender
```

Casos demonstradores iniciais: câncer de pulmão (ultra-baixa dose), metástases hepáticas de câncer colorretal (mapas de iodo como biomarcador de resposta), massas renais (caracterização por decomposição de materiais) e neuroimagem quantitativa de demências, com a doença de Alzheimer como caso central (morfometria de atrofia, carga cerebrovascular e rastreamento oportunista em TC de crânio).

## O MVP

O protótipo implementa, em nível de imagem reconstruída:

- **Simulador espectral**: phantom digital abdominal com aorta e lesões de concentração de iodo conhecida; duas VMI (50 e 70 keV) simuladas com ruído dependente da dose;
- **Decomposição de materiais**: sistema 2x2 água + iodo, voxel a voxel, gerando mapas de iodo em mg/mL;
- **Biomarcadores**: volume, concentração e carga de iodo Q por lesão, órgão e paciente, com intervalos de incerteza por realizações de ruído e por perturbação de segmentação;
- **Acompanhamento longitudinal**: registro rígido por correlação de fase, segmentação no mapa de iodo, pareamento lesão a lesão e classificação de resposta;
- **Pipeline anatômico em TC real**: controle de qualidade, segmentação pulmonar e hepática por morfologia clássica, candidatos a nódulo e lesões hipodensas;
- **Morfometria cerebral (Demonstrador 4, demências)**: cavidade intracraniana, parênquima, líquor, ventrículos, fração de líquor e índice tipo Evans em TC de crânio, com incerteza por perturbação de segmentação.
- **Teste Alzheimer / PCCT**: phantom cerebral controle vs atrofia, adquirido como TC convencional e como PCCT simulado (ruído e contraste GM/WM); triagem oportunista nos crânios públicos. Não é diagnóstico de Alzheimer.

### Resultados das avaliações iniciais

| Avaliação | Resultado |
|---|---|
| Exatidão do iodo em phantom (dose plena, lesões de 8 a 24 mm) | erro abaixo de 2,4% |
| Erro no biomarcador Q com 12,5% da dose | abaixo de 1% |
| Denoising TV sobre as VMI | viés sistemático de 1 a 2% no iodo (erro silencioso de quantificação) |
| Registro rígido basal vs. reavaliação | erro de 0,0 voxel |
| Rastreamento de 4 lesões após tratamento simulado | resposta completa, 2 parciais e 1 progressão recuperadas corretamente |
| TC de tórax pública | pulmões de 6,45 L segmentados; candidatos a nódulo listados |
| TC de abdome pública com tumor hepático | fígado de ~2,45 L; lesão multifocal de ~95 mL detectada |
| TC de crânio pública (morfometria de demências) | ICV 1452 mL; ventrículos 23 mL [15,8; 30,7]; índice tipo Evans 0,40, compatível com o alargamento ventricular visível |
| Phantom Alzheimer PCCT vs TC convencional | volume ventricular: erro −6,4% no PCCT simulado vs −81% na TC convencional; Evans −1,3% vs −14%; CNR GM/WM 2,67 vs 0,33 |

O relatório visual completo está em `mvp/results/ai_photon_mvp_dashboard.html`.

## Como executar

```bash
cd mvp
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh          # baixa o catálogo de TCs públicas (~400 MB)
python run_phantom_eval.py       # exatidao espectral + varredura de dose
python run_longitudinal_demo.py  # acompanhamento basal -> reavaliacao
python run_real_ct_eval.py       # torax e abdome em TC real
python run_neuro_eval.py         # morfometria cerebral em TC de cranio
python run_alzheimer_eval.py     # teste Alzheimer: PCCT simulado vs TC convencional
python build_report.py           # gera o dashboard HTML
```

Os resultados (figuras, métricas em JSON e dashboard) são gravados em `mvp/results/`.

## Testando com as suas próprias imagens

Qualquer TC nos formatos `.nii`, `.nii.gz`, `.nrrd`, `.mha` ou pasta DICOM pode ser analisada diretamente:

```bash
cd mvp
python analyze.py caminho/para/exame.nii.gz                  # detecta o módulo
python analyze.py caminho/para/pasta_dicom --modulo neuro    # ou força o módulo
```

O módulo é detectado automaticamente pelo conteúdo (crânio, tórax ou abdome) e pode ser forçado com `--modulo neuro|torax|abdome`. As métricas em JSON e as figuras são gravadas no diretório de saída (`--saida`, padrão `results/`).

Há uma interface local para as análises: amostras públicas, envio de arquivo, URL, phantom espectral, demo longitudinal e o teste Alzheimer/PCCT.

```bash
streamlit run app.py
```


## Dados públicos utilizados

- **CTChest.nrrd**: TC de tórax do 3D Slicer SampleData;
- **CTLiver.nrrd**: caso `liver_100` do [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (CC-BY-SA), redistribuído pelo projeto 3D Slicer;
- **CT_head.nii.gz**: TC de crânio `CT_Philips` do repositório [niivue-images](https://github.com/neurolabusc/niivue-images) (Rorden Lab, licença BSD-2);
- **CT_brain.nrrd**, **CTA_cardio.nrrd**, **CTA_Panoramix.nrrd**: outras amostras do 3D Slicer;
- **CT_Abdo.nii.gz**, **CT_AVM.nii.gz**, **CT_Electrodes.nii.gz**: volumes adicionais do niivue-images (BSD-2).

Os arquivos de imagem não são versionados neste repositório; use `download_data.sh`.

## Aviso

Protótipo de pesquisa. Não é dispositivo médico, não possui validação clínica e não deve ser utilizado para apoiar decisão diagnóstica ou terapêutica. O phantom espectral opera em nível de imagem reconstruída, sem modelo de projeções por bin de energia, e as segmentações em TC real usam regras clássicas sem aprendizado.
