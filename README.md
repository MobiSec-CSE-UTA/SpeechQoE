# SpeechQoE
A Prototype for SpeechQoE. SpeechQoE leverages speech signals to assessment Quality of Experience (QoE) for online voice services. Please refer paper for details.

## Get Started 
Once you download it to your local machine, you can run the following command in your root directory:
```bash
$python main.py --dataset voice --method voice_qoe --tgt dwhite --epoch 20 --log_suffix run_voice_dwhite --src rest --train True --model model_shallow --nshot 5 --ntask 1 --lr 0.03 --num_source 100 --num_aug_shot 30 --calibrate False --alpha 0.01 --k 2
```
Please look into main.py for explanations about parameters. 
