#!/usr/bin/env bash

# The goal of this bash script is to illustrate
# how to run the aforementioned scripts to obtain an optimal
# generative model based on the information from
# the DrugEx paper (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0355-6).
#
# That is:
#
# - RF classifier as reward in RL
# - 300 epochs for the pre-trained (exploitation) network
#   and 400 for the fine-tuned (exploration) network (Fig. 5)
# - 200 epochs during the RL training (Fig. 8) with
#   the fine-tuned network as exploration strategy (Gφ),
#   ε = 0.01 and β = 0.1 (based on Table 1)
# - batch size of 512 for all networks built

# data assembly and training
drugex dataset -d ./data/ -e A2AR_raw.txt
drugex environ -a RF
drugex pretrainer -b 512 -p 300 -e 400
drugex agent.py -e 0.01 -b 0.1

# use the trained model to sample 1000 molecules
drugex designer -a e_0.01_0.1_512x10 -e output/RF_cls_ecfp6.pkg -o output/ -n 10000