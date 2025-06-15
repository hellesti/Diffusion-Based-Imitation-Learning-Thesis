#!/bin/bash
# Bidirectional Decoding (BID)

TH=16               # Tids-horisont for observasjoner
OH=2                # Observasjonshorisont (antall frames eller steps)
OUTDIR=outputs      # Hvor resultatene skal lagres
METHOD=bid          # Evalueringsmetoden
DECAY=0.5           # Brukes for BID og EMA, styrer hvor raskt tidligere handlinger fades ut
NSAMPLE=15          # Antall action chunks som samles inn for vurdering
NMODE=3             # Antall moduser (f.eks. clusterings eller samples som beholdes)

# ===================================================================================================

TASK=pendulum
DIR=data/outputs/2025.03.21/15.26.11_train_diffusion_unet_image_pendulum_image/checkpoints  # mappen hvor sjekkpunktene er
CKPT='latest.ckpt'                 # hovedmodell
REF='latest.ckpt'                  # referansemodell (brukes i BID), Bruker samme som referanse

# ===================================================================================================
# Første test
NOISE=0.0

# open-loop
AH=8
python eval_bid.py \
--checkpoint ${DIR}/${CKPT} \
--output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}/${NOISE}/th${TH}_oh${OH}_ah${AH} \
--sampler ${METHOD} -ah ${AH} \
--nsample ${NSAMPLE} --nmode ${NMODE} --decay ${DECAY} \
--reference ${DIR}/${REF} --noise ${NOISE} --ntest 100

# ===================================================================================================
# Andre test
NOISE=1.0

# closed-loop
AH=1
python eval_bid.py \
--checkpoint ${DIR}/${CKPT} \
--output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}/${NOISE}/th${TH}_oh${OH}_ah${AH} \
--sampler ${METHOD} -ah ${AH} \
--nsample ${NSAMPLE} --nmode ${NMODE} --decay ${DECAY} \
--reference ${DIR}/${REF} --noise ${NOISE} --ntest 100
