#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="$(pwd)/output_clients"

for i in {1..10}; do
  CLIENT="client_${i}"
  SRC="${DATASET_ROOT}/${CLIENT}"
  VOL="client${i}_data"

  echo "==> Initializing $CLIENT -> $VOL"

  docker volume create "$VOL" >/dev/null

  docker run --rm \
    -v "$VOL:/data" \
    -v "$SRC:/src:ro" \
    alpine sh -c '
      if [ ! -f /data/.initialized ]; then
        cp -r /src/* /data/ &&
        touch /data/.initialized &&
        echo "Initialized"
      else
        echo "Already initialized"
      fi
    '
done

echo "All datasets ready."
