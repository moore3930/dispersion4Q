:<<!
sbatch reward.sh wmt-qe-2022.train.csv meta-llama/Llama-2-7b-hf scores/Llama2-7B/wmt-qe-22-train
sbatch reward.sh wmt-qe-2022.train.csv meta-llama/Llama-2-13b-hf scores/Llama2-13B/wmt-qe-22-train

sbatch reward.sh wmt-qe-2022.train.csv haoranxu/ALMA-7B-R scores/ALMA-r-7B/wmt-qe-22-train 
sbatch reward.sh wmt-qe-2022.train.csv haoranxu/ALMA-13B-R scores/ALMA-r-13B/wmt-qe-22-train

sbatch reward.sh wmt-qe-2022.train.csv haoranxu/ALMA-7B-Pretrain scores/ALMA-v1-7B/wmt-qe-22-train haoranxu/ALMA-7B-Pretrain-LoRA
sbatch reward.sh wmt-qe-2022.train.csv haoranxu/ALMA-13B-Pretrain scores/ALMA-v1-13B/wmt-qe-22-train haoranxu/ALMA-13B-Pretrain-LoRA


sbatch reward.sh wmt-qe-2022.test.csv meta-llama/Llama-2-7b-hf scores/Llama2-7B/wmt-qe-22-test
sbatch reward.sh wmt-qe-2022.test.csv meta-llama/Llama-2-13b-hf scores/Llama2-13B/wmt-qe-22-test

sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-7B-R scores/ALMA-r-7B/wmt-qe-22-test
sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-13B-R scores/ALMA-r-13B/wmt-qe-22-test

sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-7B-Pretrain scores/ALMA-v1-7B/wmt-qe-22-test haoranxu/ALMA-7B-Pretrain-LoRA
sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-13B-Pretrain scores/ALMA-v1-13B/wmt-qe-22-test haoranxu/ALMA-13B-Pretrain-LoRA
!

sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-7B-Pretrain scores/ALMA-Base-7B/wmt-qe-22-test
sbatch reward.sh wmt-qe-2022.test.csv haoranxu/ALMA-13B-Pretrain scores/ALMA-Base-13B/wmt-qe-22-test




#sbatch reward.sh wmt-qe-2022.test.csv Unbabel/TowerInstruct-7B-v0.1 scores/Tower-v1-7B/wmt-qe-22-test
#sbatch reward.sh wmt-qe-2022.test.csv Unbabel/TowerInstruct-13B-v0.1 scores/Tower-v1-13B/wmt-qe-22-test
#sbatch reward.sh wmt-qe-2022.test.csv Unbabel/TowerBase-7B-v0.1 scores/Tower-base-7B/wmt-qe-22-test
#sbatch reward.sh wmt-qe-2022.test.csv Unbabel/TowerBase-13B-v0.1 scores/Tower-base-13B/wmt-qe-22-test

:<<!
sbatch inference.sh en-de
sbatch inference.sh en-fr
sbatch inference.sh en-nl
sbatch inference.sh en-it
sbatch inference.sh en-es
sbatch inference.sh en-pt
sbatch inference.sh en-ko
sbatch inference.sh en-ru
sbatch inference.sh en-zh

sbatch inference.sh en-uk
sbatch inference.sh en-ja
sbatch inference.sh en-cs
sbatch inference.sh en-is
sbatch inference.sh en-es
sbatch inference.sh en-pt
!
