# Research security baseline

AI-Photon may process sensitive medical imaging. Deployments that ingest non-public data should enforce, outside this repository as appropriate:

- authenticated access;
- TLS in transit;
- encrypted storage;
- least-privilege service identities;
- audit logging;
- upload size/type limits;
- malware/content scanning where required;
- DICOM de-identification before research processing;
- controlled retention/deletion;
- secrets supplied through environment/secret managers, never committed.

The sample FastAPI service is intentionally minimal and should not be exposed directly to the public internet with identifiable clinical data.
