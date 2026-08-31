# AI-Photon

Plataforma de Inteligência Artificial para imagem espectral quantitativa em tomografia computadorizada por contagem de fótons (PCCT) aplicada à oncologia.

Este repositório reúne dois componentes:

1. **Proposta de P&D** (`proposta/`): documento completo de pesquisa aplicada e inovação tecnológica, com contexto da tecnologia PCCT, problema de pesquisa, metodologia em 12 etapas, work packages, cronograma de 30 meses, MVPs, plano de validação, aspectos éticos e regulatórios (LGPD, CEP/CONEP, RDC 657/2022) e rota de transferência tecnológica.
2. **MVP inicial** (`mvp/`): protótipo funcional do MVP 1 da proposta, com simulador espectral, decomposição de materiais, quantificação de iodo, acompanhamento longitudinal e pipeline anatômico avaliado em imagens públicas reais de TC.

## Visão central

Quantificar, caracterizar, prever e acompanhar a doença oncológica a partir do dado espectral nativo do PCCT, combinando imagens multienergéticas, dados clínicos, física de imagem e IA, em um ciclo contínuo:

```
Quantificar -> Prever -> Tratar -> Medir -> Aprender
```

Casos demonstradores iniciais: câncer de pulmão (ultra-baixa dose), metástases hepáticas de câncer colorretal (mapas de iodo como biomarcador de resposta) e massas renais (caracterização por decomposição de materiais).

## O MVP

O protótipo implementa, em nível de imagem reconstruída:

- **Simulador espectral**: phantom digital abdominal com aorta e lesões de concentração de iodo conhecida; duas VMI (50 e 70 keV) simuladas com ruído dependente da dose;
- **Decomposição de materiais**: sistema 2x2 água + iodo, voxel a voxel, gerando mapas de iodo em mg/mL;
- **Biomarcadores**: volume, concentração e carga de iodo Q por lesão, órgão e paciente, com intervalos de incerteza por realizações de ruído e por perturbação de segmentação;
- **Acompanhamento longitudinal**: registro rígido por correlação de fase, segmentação no mapa de iodo, pareamento lesão a lesão e classificação de resposta;
- **Pipeline anatômico em TC real**: controle de qualidade, segmentação pulmonar e hepática por morfologia clássica, candidatos a nódulo e lesões hipodensas.

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

O relatório visual completo está em `mvp/results/ai_photon_mvp_dashboard.html`.

## Como executar

```bash
cd mvp
pip install -r requirements.txt
./download_data.sh          # baixa as TCs publicas (~330 MB)
python run_phantom_eval.py       # exatidao espectral + varredura de dose
python run_longitudinal_demo.py  # acompanhamento basal -> reavaliacao
python run_real_ct_eval.py       # torax e abdome em TC real
python build_report.py           # gera o dashboard HTML
```

Os resultados (figuras, métricas em JSON e dashboard) são gravados em `mvp/results/`.

## Dados públicos utilizados

- **CTChest.nrrd**: TC de tórax do 3D Slicer SampleData;
- **CTLiver.nrrd**: caso `liver_100` do [Medical Segmentation Decathlon](http://medicaldecathlon.com/) (CC-BY-SA), redistribuído pelo projeto 3D Slicer.

Os arquivos de imagem não são versionados neste repositório; use `download_data.sh`.

## Aviso

Protótipo de pesquisa. Não é dispositivo médico, não possui validação clínica e não deve ser utilizado para apoiar decisão diagnóstica ou terapêutica. O phantom espectral opera em nível de imagem reconstruída, sem modelo de projeções por bin de energia, e as segmentações em TC real usam regras clássicas sem aprendizado.
