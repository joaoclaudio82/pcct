# Melhorias de robustez e validação

Esta revisão corrige contratos numéricos e operacionais do protótipo.
Ela não modifica os resultados históricos publicados em `mvp/results/` nem
representa uma nova validação clínica.

## Cálculos e geometria

- Carga de iodo e biomarcadores aceitam espaçamento isotrópico ou `(z, y, x)`.
  Volume de voxel em mL: `dz * dy * dx / 1000`.
- Decomposição espectral, métricas de segmentação e radiômica rejeitam imagens
  de formatos diferentes, evitando o broadcasting silencioso do NumPy.
- Dose, espaçamento, probabilidades e descritores de lesões são validados.
- Monte Carlo exige pelo menos duas amostras. Probabilidades inválidas geram
  erro, em vez de serem silenciosamente truncadas.
- Zero iterações de perturbação geométrica retorna o volume original.
- Reamostragem preserva metadados e usa ar (-1024 HU) fora de imagens de TC;
  rótulos usam zero e vizinho mais próximo. `default_value` permite outro fundo.
- Imagens devem ser escalares 3D e ter geometria física válida. Na leitura
  DICOM, a série com mais instâncias também fornece os metadados descritivos.

## Mudança no contrato longitudinal

`classify_response` mantém categorias experimentais, sem equivalência clínica
com RECIST. O desaparecimento volumétrico usa o limiar configurado. Iodo zero
em uma lesão ainda volumetricamente presente não implica resposta completa.
As variações percentuais são as calculadas, inclusive em resposta completa.
Se o volume ou a carga basal forem zero, a categoria será `not_evaluable`;
a variação cujo denominador é zero será `None` (`null` em JSON).
Consumidores devem aceitar essa categoria e esses valores ausentes.

Os pesos de pareamento são normalizados para soma unitária; isso preserva a
atribuição para pesos proporcionais, mas altera a escala do custo retornado.

## Verificação realizada

```bash
python -m pip install -e '.[dev,api]'
python -m ruff check mvp/ai_photon_mvp tests api
python -m pytest --cov=ai_photon_mvp --cov-report=term --cov-fail-under=25
python -m ai_photon_mvp.benchmark
```

Verificação local com Python 3.12: 68 testes aprovados, cobertura total de
43,76% no pacote e lint sem erros. Os testes incluem uploads reais sintéticos
nos quatro formatos da API, limpeza de temporários, limites de arquivo e
voxels, geometria oblíqua, espaçamento anisotrópico e entradas inválidas.
Dois avisos de descontinuação vêm de dependências da infraestrutura de testes.

A configuração de CI executa Python 3.11 e 3.12 com o extra de API. O resultado
remoto deve ser consultado no GitHub Actions após o push. O build Docker não
foi executado neste ambiente, pois não havia motor Docker disponível. As
integrações opcionais MONAI, nnU-Net, PyRadiomics e MLflow e os experimentos
com imagens públicas completas não foram revalidados nesta revisão.
