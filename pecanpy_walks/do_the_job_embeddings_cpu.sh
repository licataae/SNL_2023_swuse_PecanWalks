#!/bin/sh
#SBATCH --job-name fr_parse
#SBATCH --output pecanpy.o%j
#SBATCH --ntasks 3 #how many CPUs?
#SBATCH --partition public-cpu #which partition to use? run "sinfo" to see which ones you have access to...
#SBATCH --time 24:00:00 #set wall time for 24 hours, try to be as accurate as possible (max 4 days)

#source the virtual environment                                                                                                                                                         
source ~/baobab_python_env2/bin/activate
python_path=/home/users/l/licata/baobab_python_env2/bin/python
$python_path /home/users/l/licata/projects/wiki_matrix/redo/final/ppmi_embeddings_pecanpy.py

