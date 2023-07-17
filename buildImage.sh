#!/bin/bash

# get the image ID of the container coding_learner_container #--no-cache 

IMAGE_ID=$(docker images --filter=reference=abt_ds --format "{{.ID}}")

docker build -t abt_ds . 

docker rmi $IMAGE_ID
