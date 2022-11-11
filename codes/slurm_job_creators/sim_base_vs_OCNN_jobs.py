import os 

SNR = [-10,-5,0,5,10,15,20]
normalize =1
if normalize==1:
    normalizeString="normalize"
else:
    normalizeString="notNormalize"

#cluster= 'puma'
cluster= 'ocelote'

directory="./"

for snr in SNR:
    for seed in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
        the_name = directory+"jobs/base_vs_OCNN/OCNN"+str(snr)+"_sim_seed"+str(seed)+".slurm"
        print(the_name)
        f= open(the_name,"w+")
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=OCNN%d_sim_%s_seed%d\n" % (snr,normalizeString,seed))
        f.write("#SBATCH --account=yazdaniabyaneh\n")
        f.write("#SBATCH --partition=standard\n")
        f.write("#SBATCH --ntasks-per-node=28\n")
        f.write("#SBATCH --nodes=1\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write("#SBATCH --time=12:00:00\n")
        f.write("#SBATCH -o ./logs/base_vs_OCNN/%x_%j.out\n")
        f.write("pwd; hostname; date\nmodule load anaconda\nconda init bash\nsource ~/.bashrc\nconda activate cnn9\n")
        f.write("python ./codes/python_codes/sim/sim_base_vs_OCNN.py %d %d %d" % (snr,seed,normalize))
        f.close()
        os.system("sbatch "+the_name)


