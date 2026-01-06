python /data2/ly/dataset_eval/code_apply/vendi_score_execution_records.py \
  --data-roots /data2/ly/dataset_eval/code_apply/your_dir1 /data2/ly/dataset_eval/code_apply/your_dir2 \
  --model-path /data2/Qwen/Qwen3-Embedding-0.6B \
  --batch-size 8


   python vendi_score_analysis.py --data-roots /your/data/path

      python vendi_score_execution_records.py \
       --data-roots /your/data/path \
       --include-values


screen -L -S swift_8b -Logfile /raid/data/ly/data/agentgym_data_new/swift.log bash /raid/data/ly/data/agentgym_data_new/swift_qwen3_8b.sh
