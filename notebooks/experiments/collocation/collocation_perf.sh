#0.3
#llama vs. others
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama_llama2_30_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama_llama2chat_30_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama_alpaca_30_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama_vicuna_30_perf.log
##python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama_koala_30_perf.log
#
##Llama2 vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/huggyllama/llama-7b/--mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2_llama_30_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2_alpaca_30_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2_vicuna_30_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2_koala_30_perf.log
#
##llama2chat vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2chat_llama_30_perf.log
python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_llama2_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/tatsu-lab/alpaca-7b/  --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2chat_alpaca_30_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2chat_vicuna_30_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/llama2chat_koala_30_perf.log
#
##alpaca vs. others
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/alpaca_llama_30_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/alpaca_llama2_30_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/alpaca_llama2chat_30_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/alpaca_vicuna_30_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/alpaca_koala_30_perf.log
#
##vicuna vs. others
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/vicuna_llama_30_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/vicuna_llama2_30_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/vicuna_llama2chat_30_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/vicuna_alpaca_30_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/vicuna_koala_30_perf.log
#
##koala vs. others
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/koala_llama_30_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/koala_llama2_30_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/koala_llama2chat_30_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/koala_alpaca_30_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.30 | tee ./log/koala_vicuna_30_perf.log
#
##0.6
##llama vs. others
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama_llama2_60_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama_llama2chat_60_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama_alpaca_60_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama_vicuna_60_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama_koala_60_perf.log
#
##Llama2 vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/huggyllama/llama-7b/--mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2_llama_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2_llama2chat_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2_alpaca_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2_vicuna_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2_koala_60_perf.log
#
##llama2chat vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2chat_llama_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2chat_llama2_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/tatsu-lab/alpaca-7b/  --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2chat_alpaca_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2chat_vicuna_60_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/llama2chat_koala_60_perf.log
#
##alpaca vs. others
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/alpaca_llama_60_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/alpaca_llama2_60_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/alpaca_llama2chat_60_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/alpaca_vicuna_60_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/alpaca_koala_60_perf.log
#
##vicuna vs. others
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/vicuna_llama_60_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/vicuna_llama2_60_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/vicuna_llama2chat_60_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/vicuna_alpaca_60_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/vicuna_koala_60_perf.log
#
##koala vs. others
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/koala_llama_60_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/koala_llama2_60_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/koala_llama2chat_60_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/koala_alpaca_60_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.60 | tee ./log/koala_vicuna_60_perf.log
#
##0.9
##llama vs. others
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama_llama2_90_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama_alpaca_90_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama_vicuna_90_perf.log
#python collocation.py --models /workspace/llms/huggyllama/llama-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama_koala_90_perf.log
#
##Llama2 vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/huggyllama/llama-7b/--mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_llama_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_alpaca_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_vicuna_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2_koala_90_perf.log
#
##llama2chat vs. others
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_llama_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_llama2_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/tatsu-lab/alpaca-7b/  --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_alpaca_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_vicuna_90_perf.log
#python collocation.py --models /workspace/llms/meta-llama/Llama-2-7b-chat-hf/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/llama2chat_koala_90_perf.log
#
##alpaca vs. others
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/alpaca_llama_90_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/alpaca_llama2_90_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/alpaca_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/alpaca_vicuna_90_perf.log
#python collocation.py --models /workspace/llms/tatsu-lab/alpaca-7b/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/alpaca_koala_90_perf.log
#
##vicuna vs. others
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/vicuna_llama_90_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/vicuna_llama2_90_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/vicuna_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/ /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/vicuna_alpaca_90_perf.log
#python collocation.py --models /workspace/llms/lmsys/vicuna-7b-v1.1/  /workspace/llms/TheBloke/koala-7B-HF/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/vicuna_koala_90_perf.log
#
##koala vs. others
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/huggyllama/llama-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/koala_llama_90_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/ /workspace/llms/meta-llama/Llama-2-7b-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/koala_llama2_90_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/meta-llama/Llama-2-7b-chat-hf/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/koala_llama2chat_90_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/tatsu-lab/alpaca-7b/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/koala_alpaca_90_perf.log
#python collocation.py --models /workspace/llms/TheBloke/koala-7B-HF/  /workspace/llms/lmsys/vicuna-7b-v1.1/ --mode collocate --profile_mode performance --collocate_percentage 0.90 | tee ./log/koala_vicuna_90_perf.log
