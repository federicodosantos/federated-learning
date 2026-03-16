#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="$(pwd)/output_clients"

declare -A CLIENTS=(
  ["client_1"]="client1_data"
  ["client_2"]="client2_data"
  ["client_3"]="client3_data"
)

for CLIENT in "${!CLIENTS[@]}"; do
  SRC="${DATASET_ROOT}/${CLIENT}"
  VOL="${CLIENTS[$CLIENT]}"

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
