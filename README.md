# Description
NABind is a novel structure-based method to predict DNA/RNA-binding residues by leveraging deep learning and template approaches. 

# Third-party software

# Important python packages

# Usage
## 1. Configuration
Download and install the standard alone software listed above.  
Change the paths of these softwares and related databases at config/config.json
## 2. Prediction
Run the following command:  

    python predict.py --pdb ./demo/6chv_D.pdb --outdir ./demo/ --type DNA --structure native

Type -h for help information:

    python predict.py -h
   
   
