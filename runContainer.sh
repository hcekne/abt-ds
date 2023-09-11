#!/bin/bash
# if port 8888 is already used on the dsvm use -p 8889:8888 instead or similar port

# use this option when running in detached mode
#--user "$(id -u)" --group-add users \
# running as root now so changing out the variables with my default variables
# -e NB_USER="$(whoami)" \
# -e NB_UID="$(id -u)" \
# -e NB_GID="$(id -g)"
#
#


docker run -it --rm \
	-p 8888:8888 \
	--user root \
	-e NB_USER=hcekne \
	-e NB_UID=1000 \
	-e NB_GID=984 \
	-e CHOWN_HOME=yes \
	-e CHOWN_HOME_OPTS="-R" \
	-e RETRAIN_MODELS="yes" \
	--env-file ./.env \
	-w "/home/hcekne" \
	-v "${PWD}":/home/hcekne/work/ \
	-v /home/hcekne/ml_data/:/home/hcekne/data/ \
	abt_ds
