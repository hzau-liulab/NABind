# Description
NABind is a novel structure-based method to predict DNA/RNA-binding residues by leveraging deep learning and template approaches.  
![image](img/img.png)  

# Third-party software
PSI-BLAST https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/  
HH-suite https://github.com/soedinglab/hh-suite  
GHECOM https://pdbj.org/ghecom/  
TM-align https://zhanggroup.org/TM-align/  
NW-align https://zhanggroup.org/NW-align/  
CD-HIT https://github.com/weizhongli/cdhit/releases  
DSSP https://swift.cmbi.umcn.nl/gv/dssp/DSSP_5.html  
NACCESS http://www.bioinf.manchester.ac.uk/naccess/  

# Database requirement
UniRef90 https://www.uniprot.org/help/downloads  
Uniclust30 https://uniclust.mmseqs.com/  
Manually created template library [Google Drive](https://drive.google.com/file/d/1hbQjtnSdU1I8TpVpdwHGx54xWhZheoGs/view?usp=share_link) or http://liulab.hzau.edu.cn/NABind  

# Important python packages
Numpy  1.25.0
Pandas  1.2.0
Biopython  1.76
Scipy  1.10.1
cdhit-reader  0.1.1
fair-esm  2.0.0
pytorch  2.0.1
DGL  0.9.0
GraphRicciCurvature  0.5.3.1

# Usage
## 1. Download pre-trained models
The pre-trained models can be found at [Google Drive](https://drive.google.com/drive/folders/1TOp5xAqd5Wf_RpubCyrhouU_sX4FXLov?usp=sharing) or http://liulab.hzau.edu.cn/NABind    
## 2. Configuration
Creat NABind environment (conda env create -f environment.yaml).  
Manually download and install the third-party software listed above.  
Change the paths of these softwares and related databases at config/config.json  
Activate NABind environment (conda activate NABind).  
## 3. Prediction
Run the following command:  

    python predict.py --pdb ./demo/6chv_D.pdb --outdir ./demo/ --type DNA --structure native

Type -h for help information:

    python predict.py -h
## OR use docker image
Download image

    docker load -i nabind_image.tar
    docker run -it -v your/uniref90_for_blast:/app/db/u90 -v your/uniclust30_for_hhblits:/app/db/u30 -v downloaded/templatedb/BioLip:/app/db/template/BioLip -v download/esm/model:/app/db/esm -v downloaded/pre-trained/model:/app/model nabindv1hzau
    python predict.py
    
# Citation
Structure-based prediction of nucleic acid binding residues by merging deep learning- and template-based approaches. *Submitted*, 2023.
