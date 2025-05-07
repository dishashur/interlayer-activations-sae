"""
SAEs are tools to interpret LLM activations, to disentangle the superposition observed in LMs activation space. 

Here's what we will do: 1. NN on SAE representations layer i, NN on corresponding LM's representation layer i,
2. Observe correlation of a distance metric on both NNs
3. Repeat procedure through layers i+2, i+5 e.t.c

Go from here 

different sae loss ---> interlayer structure
linear probles ---> interlayer structure
code help from https://github.com/JoshEngels/SAE-Dark-Matter
future work ----> quantify  monotsemanticity/superposition through sparsity
"""

import gpt2
import torch
import SAEtrainedongpt2


#to load a pretrained SAE ---from SAE_probes
def load_gemma_2_9b_sae(sae_id):
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release = "gemma-scope-9b-pt-res",
        sae_id = sae_id,
        device = "cpu",
    )
    return sae

#to load a pretraibedSAE --- from SAE_dark_matter
gemma-scope-2b-pt-res:
  repo_id: google/gemma-scope-2b-pt-res
  model: gemma-2-2b
  conversion_func: gemma_2
  saes:
  - id: layer_0/width_16k/average_l0_105
    path: layer_0/width_16k/average_l0_105
    l0: 105
#to save a pickle file for gemma layer wise autoencoder width and l0
size = "2b"
extension = "res" #"mlp", "att"
sae = abovedict[f"gemma-scope-{size}-pt-{extension}"]["saes"]
 # Parse id of form 'layer_12/width_262k/average_l0_67'
layer, width, l0 = sae["id"].split("/")
layer = int(layer.split("_")[1])
width = width.split("_")[1]
l0 = int(l0.split("_")[2])
if layer not in layer_to_width_l0_pairs:
    layer_to_width_l0_pairs[layer] = []
layer_to_width_l0_pairs[layer].append((width, l0))
with open(f"gemma_sae_pickles/gemma_sae_dict_{size}_{extension}.pkl", "wb") as f:
        pickle.dump(layer_to_width_l0_pairs, f)

#from scrips/create_run_all_for_gemma_single_layer.py
layer = 20
layer_type = "att" #takes similar value as extension
gemma_dict = pickle.load(open(f"gemma_sae_pickles/gemma_sae_dict_9b_{layer_type}.pkl", "rb"))
# Sort by width, which is of the form 16k or 1m or 256k etc.
values = []
for width, l0 in gemma_dict[layer]:
    if width[-1] == "k":
        width = int(width[:-1]) * 1000
    elif width[-1] == "m":
        width = int(width[:-1]) * 1000000
    values.append(width)
hyperparams = [x for _, x in sorted(zip(values, hyperparams))]
command = f"python save_info_gemma.py --layer {layer} --sae_width {width} --sae_l0 {l0} --size 9b --layer_type {layer_type}"

#in SAE-Dark-Matter/encoder_pursuit.py
sae_name = f"layer_{layer}/width_{sae_width}/average_l0_{sae_l0}"
dictionary_sae = SAE.from_pretrained(
    release=f"gemma-scope-{size}-pt-{}",
    sae_id=sae_name,
    device="cpu",
)[0] 


#get the SAE activations



