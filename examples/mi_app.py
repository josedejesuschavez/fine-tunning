"""Ejemplo de integracion: procesa una carpeta de documentos con el modelo entrenado.

Uso:
    uv run python examples/mi_app.py                  # procesa examples/docs
    uv run python examples/mi_app.py otra/carpeta     # procesa los .txt de otra carpeta
"""

import json
import sys
from pathlib import Path

from fine_tunning.extract import Extractor, ExtractionError

HERE = Path(__file__).resolve().parent
# En tu aplicacion, pon aqui la ruta absoluta del adaptador.
ADAPTER = HERE.parent / "outputs" / "lora"


def main(carpeta):
    docs = sorted(Path(carpeta).glob("*.txt"))
    if not docs:
        print(f"No hay archivos .txt en {carpeta}")
        return

    extractor = Extractor(ADAPTER)  # cargar el modelo una sola vez, al arrancar

    for doc in docs:
        texto = doc.read_text(encoding="utf-8")
        try:
            datos = extractor.extract(texto)
        except ExtractionError as e:
            # El modelo respondio algo invalido: mandar a revision manual.
            print(f"[REVISAR]   {doc.name}: {e} | salida: {e.raw!r}")
            continue
        except ValueError as e:
            # Documento vacio o demasiado largo.
            print(f"[RECHAZADO] {doc.name}: {e}")
            continue
        print(f"[OK]        {doc.name}: {json.dumps(datos, ensure_ascii=False)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "docs")
