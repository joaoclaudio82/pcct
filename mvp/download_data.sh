#!/usr/bin/env bash
# Baixa as imagens publicas de TC usadas nas avaliacoes do MVP.
# CTChest: 3D Slicer SampleData. CTLiver: caso liver_100 do Medical
# Segmentation Decathlon (CC-BY-SA), redistribuido pelo projeto 3D Slicer.
set -e
mkdir -p data
curl -L -o data/CTChest.nrrd https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/4507b664690840abb6cb9af2d919377ffc4ef75b167cb6fd0f747befdb12e38e
curl -L -o data/CTLiver.nrrd https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/e16eae0ae6fefa858c5c11e58f0f1bb81834d81b7102e021571056324ef6f37e
echo "dados baixados em ./data"
