# Run the full Question 1 pipeline on K_{2,2} (PowerShell).
# Usage: ./scripts/run_question1.ps1
python experiments/question1/run_training.py --config configs/question1/k22_binary_iql.yaml
python experiments/question1/run_baselines.py
python experiments/question1/run_cross_play.py --config configs/question1/k22_binary_iql.yaml
python experiments/question1/run_extended_variant.py --config configs/question1/k22_ternary_iql.yaml
python experiments/question1/analyze_results.py
