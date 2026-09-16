MODEL= /st1_0

ssh tp41139@musica.vie.asc.ac.at "tar czf - -C /data/fs201368/tp41139/models$MODEL model_latest.pt" | \
ssh tpausch@hepgpu2.hephy.oeaw.ac.at "mkdir -p /scratch/tpausch/models$MODEL && tar xzf - -C /scratch/tpausch/models$MODEL"