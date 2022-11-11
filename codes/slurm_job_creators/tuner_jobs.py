import os 

SNR = [-5,0,5,10,20]

directory="./jobs/"


for snr in SNR:
    the_name = directory+"tuner_sim_"+str(snr)+".slurm"
    print(the_name)
    f= open(the_name,"w+")
    f.write("#!/bin/bash\n")
    f.write("#SBATCH --job-name=tuner_sim_snr%d_\n" % (snr))
    f.write("#SBATCH --account=yazdaniabyaneh\n")
    f.write("#SBATCH --partition=standard\n")
    f.write("#SBATCH --ntasks-per-node=28\n")
    f.write("#SBATCH --nodes=1\n")
    f.write("#SBATCH --gres=gpu:1\n")
    f.write("#SBATCH --time=24:00:00\n")
    f.write("#SBATCH -o ./logs/%x_%j.out\n")
    f.write("pwd; hostname; date\nmodule load anaconda\nconda init bash\nsource ~/.bashrc\nconda activate cnn9\n")
    f.write("python ./codes/python_codes/tuner_all_snrs.py %d" % (snr))
    f.close()
    os.system("sbatch "+the_name)


