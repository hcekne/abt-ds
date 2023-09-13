#!/bin/bash

# get the image ID of the container coding_learner_container #--no-cache 

IMAGE_ID=$(docker images --filter=reference=ds_abm --format "{{.ID}}")

docker build -t ds_abm . 

docker rmi $IMAGE_ID
