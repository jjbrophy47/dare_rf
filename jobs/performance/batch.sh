# DARE (topd=0)
./jobs/performance/primer.sh 'surgical' 'dare' 1.0 'accuracy' 'gini' 3 1440 short
./jobs/performance/primer.sh 'vaccine' 'dare' 1.0 'accuracy' 'gini' 6 1440 short
./jobs/performance/primer.sh 'adult' 'dare' 1.0 'accuracy' 'gini' 3 1440 short
./jobs/performance/primer.sh 'bank_marketing' 'dare' 1.0 'roc_auc' 'gini' 3 1440 short
./jobs/performance/primer.sh 'flight_delays' 'dare' 1.0 'roc_auc' 'gini' 20 1440 short
./jobs/performance/primer.sh 'diabetes' 'dare' 1.0 'accuracy' 'gini' 20 1440 short
./jobs/performance/primer.sh 'no_show' 'dare' 1.0 'roc_auc' 'gini' 20 1440 short
./jobs/performance/primer.sh 'olympics' 'dare' 1.0 'roc_auc' 'gini' 20 1440 short
./jobs/performance/primer.sh 'census' 'dare' 1.0 'roc_auc' 'gini' 20 1440 short
./jobs/performance/primer.sh 'credit_card' 'dare' 1.0 'average_precision' 'gini' 6 1440 short
./jobs/performance/primer.sh 'twitter' 'dare' 0.5 'roc_auc' 'gini' 35 1440 short
./jobs/performance/primer.sh 'synthetic' 'dare' 0.5 'accuracy' 'gini' 60 1440 short
./jobs/performance/primer.sh 'higgs' 'dare' 0.05 'accuracy' 'gini' 60 1440 short
./jobs/performance/primer.sh 'ctr' 'dare' 0.005 'roc_auc' 'gini' 70 1440 short

./jobs/performance/notune_primer.sh 'credit_card' 'dare' 250 10 5 'average_precision' 'gini' 6 1440 short

# ExtraTrees
# ./jobs/performance/primer.sh 'surgical' 'extra_trees' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'vaccine' 'extra_trees' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'adult' 'extra_trees' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'bank_marketing' 'extra_trees' 1.0 'roc_auc' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'flight_delays' 'extra_trees' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'diabetes' 'extra_trees' 1.0 'accuracy' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'no_show' 'extra_trees' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'olympics' 'extra_trees' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'census' 'extra_trees' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'credit_card' 'extra_trees' 1.0 'average_precision' 'gini' 6 1440 short
# ./jobs/performance/primer.sh 'twitter' 'extra_trees' 0.5 'roc_auc' 'gini' 35 1440 short
# ./jobs/performance/primer.sh 'synthetic' 'extra_trees' 0.5 'accuracy' 'gini' 40 1440 short
./jobs/performance/primer.sh 'higgs' 'extra_trees' 0.05 'accuracy' 'gini' 45 1440 short
./jobs/performance/primer.sh 'ctr' 'extra_trees' 0.005 'roc_auc' 'gini' 60 1440 short

# ExtraTrees (k=1)
# ./jobs/performance/primer.sh 'surgical' 'extra_trees_k1' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'vaccine' 'extra_trees_k1' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'adult' 'extra_trees_k1' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'bank_marketing' 'extra_trees_k1' 1.0 'roc_auc' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'flight_delays' 'extra_trees_k1' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'diabetes' 'extra_trees_k1' 1.0 'accuracy' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'no_show' 'extra_trees_k1' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'olympics' 'extra_trees_k1' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'census' 'extra_trees_k1' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'credit_card' 'extra_trees_k1' 1.0 'average_precision' 'gini' 6 1440 short
# ./jobs/performance/primer.sh 'twitter' 'extra_trees_k1' 0.5 'roc_auc' 'gini' 35 1440 short
# ./jobs/performance/primer.sh 'synthetic' 'extra_trees_k1' 0.5 'accuracy' 'gini' 40 1440 short
./jobs/performance/primer.sh 'higgs' 'extra_trees_k1' 0.05 'accuracy' 'gini' 45 1440 short
./jobs/performance/primer.sh 'ctr' 'extra_trees_k1' 0.005 'roc_auc' 'gini' 60 1440 short

# Sklearn
# ./jobs/performance/primer.sh 'surgical' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'vaccine' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'adult' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'bank_marketing' 'sklearn' 1.0 'roc_auc' 'gini' 3 1440 short
# ./jobs/performance/primer.sh 'flight_delays' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'diabetes' 'sklearn' 1.0 'accuracy' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'no_show' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'olympics' 'sklearn' 1.0 'roc_auc' 'gini' 25 4320 long
# ./jobs/performance/primer.sh 'census' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/primer.sh 'credit_card' 'sklearn' 1.0 'average_precision' 'gini' 6 1440 short
# ./jobs/performance/primer.sh 'twitter' 'sklearn' 0.5 'roc_auc' 'gini' 35 1440 short
# ./jobs/performance/primer.sh 'synthetic' 'sklearn' 0.5 'accuracy' 'gini' 60 4320 long
./jobs/performance/primer.sh 'higgs' 'sklearn' 0.05 'accuracy' 'gini' 45 1440 short
./jobs/performance/primer.sh 'ctr' 'sklearn' 0.005 'roc_auc' 'gini' 60 1440 short

# Sklearn w/ bootstrap
# ./jobs/performance/bootstrap_primer.sh 'surgical' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/bootstrap_primer.sh 'vaccine' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/bootstrap_primer.sh 'adult' 'sklearn' 1.0 'accuracy' 'gini' 3 1440 short
# ./jobs/performance/bootstrap_primer.sh 'bank_marketing' 'sklearn' 1.0 'roc_auc' 'gini' 3 1440 short
# ./jobs/performance/bootstrap_primer.sh 'flight_delays' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/bootstrap_primer.sh 'no_show' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/bootstrap_primer.sh 'diabetes' 'sklearn' 1.0 'accuracy' 'gini' 20 1440 short
# ./jobs/performance/bootstrap_primer.sh 'olympics' 'sklearn' 1.0 'roc_auc' 'gini' 25 4320 long
# ./jobs/performance/bootstrap_primer.sh 'census' 'sklearn' 1.0 'roc_auc' 'gini' 20 1440 short
# ./jobs/performance/bootstrap_primer.sh 'credit_card' 'sklearn' 1.0 'average_precision' 'gini' 6 1440 short
# ./jobs/performance/bootstrap_primer.sh 'twitter' 'sklearn' 0.5 'roc_auc' 'gini' 35 1440 short
# ./jobs/performance/bootstrap_primer.sh 'synthetic' 'sklearn' 0.5 'accuracy' 'gini' 40 1440 short
./jobs/performance/bootstrap_primer.sh 'higgs' 'sklearn' 0.05 'accuracy' 'gini' 45 1440 short
./jobs/performance/bootstrap_primer.sh 'ctr' 'sklearn' 0.005 'roc_auc' 'gini' 60 1440 short
