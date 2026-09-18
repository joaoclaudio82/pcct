# API de pesquisa

Na raiz do repositório:

```bash
python -m pip install -e '.[api,dev]'
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

A documentação interativa fica em `http://127.0.0.1:8000/docs`.

- `GET /health`: estado do serviço, versão e indicação de uso em pesquisa.
- `POST /v1/exams/inspect`: recebe `file` como multipart e retorna região corporal estimada, dimensões, espaçamento e controle de qualidade.

```bash
curl --fail-with-body -F 'file=@exame.nii.gz' http://127.0.0.1:8000/v1/exams/inspect
```

## Contrato de entrada

Somente volumes escalares 3D autocontidos: `.nii`, `.nii.gz`, `.nrrd` e `.mha`.
O limite de arquivo é 128 MiB e o de imagem decodificada é 64 milhões de voxels.
O tamanho em voxels é conferido no cabeçalho antes de carregar o volume inteiro.
Arquivos NRRD/MetaImage com referências a dados externos são rejeitados.
Pastas DICOM continuam disponíveis pela biblioteca e pela CLI, não por este endpoint.

| HTTP | Significado |
|---|---|
| 200 | Inspeção concluída; consultar também os alertas de QA |
| 413 | Arquivo ou quantidade de voxels acima do limite |
| 415 | Extensão não suportada |
| 422 | Arquivo vazio, inválido ou imagem incompatível |

`shape` e `spacing_mm` usam ordem **z, y, x**. A matriz de direção no objeto
`MedicalVolume` segue a convenção nativa do SimpleITK. Variações de HU são
calculadas somente sobre valores finitos; se não houver nenhum, `hu_range`
contém dois valores `null`. A região corporal é uma heurística de pesquisa.

O processamento pesado é executado fora do event loop, e os arquivos temporários
são removidos ao concluir ou falhar. Mensagens de erro não retornam caminhos
internos nem a mensagem original do leitor de imagens.

## Implantação

A API não implementa autenticação, limitação de frequência ou isolamento de
processos. Mantenha-a local ou atrás de uma camada autenticada. Configure limite
do corpo HTTP, concorrência, tempo e memória no proxy/servidor: o parser multipart
pode receber e armazenar temporariamente o corpo antes da validação do endpoint.
Os limites internos não substituem essa proteção de entrada.

O Compose publica portas apenas em `127.0.0.1`. A imagem instala os extras de
API e Streamlit e executa com usuário sem privilégios de root.

Protótipo de pesquisa, sem validação clínica.
