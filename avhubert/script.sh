OMP_NUM_THREADS=1 python -B infer_s2s.py --config-dir ./conf/ --config-name s2s_decode \
  dataset.gen_subset=test common_eval.path=/home/sungnyun/av-hubert-temp/avhubert/outputs/large_noise_pt_noise_ft_433h_xmodal/finetune/checkpoints/checkpoint_last.pt \
  common_eval.results_path=/home/sungnyun/av-hubert-temp/avhubert/outputs/large_noise_pt_noise_ft_433h_xmodal/decode/s2s/test \
  override.noise_wav=/home/sungnyun/dataset/musan/tsv/babble override.noise_prob=1 override.noise_snr=0 \
  common.user_dir=`pwd`
