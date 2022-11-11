import os 

SNR = [-10,-5, 0, 5, 10, 15, 20]
types=["g","w","l"] ##waveform types: 5G, Wi-Fi, or LTE

directory="./jobs/"

for snr in SNR:
    for type in types:
            slurmScript = directory+"snr"+str(snr)+"/"+type+"_SNR"+str(snr)+"_rf6.slurm"
            f= open(slurmScript,"w+")
            f.write("#!/bin/bash\n")
            f.write("#SBATCH --job-name=%s_snr%d\n" % (type, snr))
            f.write("#SBATCH --account=yazdaniabyaneh\n")
            f.write("#SBATCH --partition=standard\n")
            f.write("#SBATCH --ntasks=30\n")
            f.write("#SBATCH --ntasks-per-node=30\n")
            f.write("#SBATCH --nodes=1\n")
            f.write("#SBATCH --time=00:30:00\n")
            #log directory
            f.write("#SBATCH -o ./logs/%x_%j_Model.out\n")
            #activating conda virtual environment "cnn9"
            f.write("pwd; hostname; date\nmodule load anaconda\nconda init bash\nsource ~/.bashrc\nconda activate cnn9\n")
            #executing
            f.write("python ./codes/python_codes/mat_to_hkl_SNR.py %s %d" % (type,snr))
            f.close()
            os.system("sbatch "+slurmScript)
