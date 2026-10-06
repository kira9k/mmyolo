#!/bin/bash

CONTAINER_NAME=mmyolo_temp

# Удаляем старый контейнер, если есть
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  docker rm -f $CONTAINER_NAME
fi

# Запускаем ЖИВОЙ контейнер
docker run --name $CONTAINER_NAME --gpus all --shm-size=16g \
  -v $(pwd):/mmyolo \
  -d mmyolo:dev sleep infinity

sleep 2

docker exec $CONTAINER_NAME sed -i \
"s/checkpoint = torch.load(filename, map_location=map_location)/checkpoint = torch.load(filename, map_location=map_location, weights_only=False)/" \
/opt/conda/lib/python3.11/site-packages/mmengine/runner/checkpoint.py

# Заходим внутрь
docker exec -it $CONTAINER_NAME /bin/bash
