# NETR
The codes for the coursework project "NETR: Nested Transformer for Medical Image Segmentation" of PURDUE UNIVERSITY ECE 69500: ML In Bioinfo And Healthcare. 

## 1. Download pre-trained swin transformer model (Swin-T)
* [Get pre-trained model in this link](https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. Please go to ["./datasets/README.md"](datasets/README.md).
## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

```bash
pip install -r requirements.txt
```

## 4. Train, Test, visualization

- Run the train script on synapse dataset.

- Train and test

```bash
sh train_test.sh 
```

- Visualization
```bash
sh visualization.sh 
```

## Reproducibility
The reported model weight is [here](https://drive.google.com/drive/folders/1pvYnY9pefHoLDeCrXaRvB8hx8XTHExTz?usp=sharing)

## References
* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [TransUnet](https://github.com/Beckschen/TransUNet)
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)


