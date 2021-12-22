# LSBATCH: User input
#!/bin/sh

### -- set the job Name --
#BSUB -J RLtraining

### -- specify queue --
#BSUB -q gpuv100

### -- ask for number of cores (default: 1) --
#BSUB -n 4

### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"

### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- set walltime limit: hh:mm --
#BSUB -W 24:00

### -- specify that we need 5GB of memory per core/slot --
#BSUB -R "rusage[mem=5GB]"

### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot --
#BSUB -M 8GB

### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s154674@student.dtu.dk

### -- send notification at start --
#BSUB -B

### -- send notification at completion--
#BSUB -N

### -- Specify the output and error file. %J is the job-id %I is the job-array index --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o Output_%J_%I.out
#BSUB -e Error_%J_%I.err

### here follow the module to be loaded
module load cuda/10.0
### module load gcc/7.4.0
### module load mpi/3.1.3-gcc-7.4.0
### module load steno-amber/19.17

/zhome/6b/c/109916/env/bin/python3 test.py "$1" "$2"
