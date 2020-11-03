criterion='gini'

# D-DART
# ./jobs/performance/primer.sh surgical 'exact' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh vaccine 'exact' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh adult 'exact' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh bank_marketing 'exact' 1.0 roc_auc $criterion 3 1440 short
# ./jobs/performance/primer.sh flight_delays 'exact' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh diabetes 'exact' 1.0 accuracy $criterion 20 1440 short
# ./jobs/performance/primer.sh olympics 'exact' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh skin 'exact' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh census 'exact' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh credit_card 'exact' 1.0 average_precision $criterion 6 1440 short
# ./jobs/performance/primer.sh twitter 'exact' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh gas_sensor 'exact' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh synthetic 'exact' 0.5 accuracy $criterion 40 1440 short
# ./jobs/performance/notune_primer.sh higgs 100 10 0.25 'exact' accuracy $criterion 45 1440 short

# Random
# ./jobs/performance/primer.sh surgical 'random' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh vaccine 'random' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh adult 'random' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh bank_marketing 'random' 1.0 roc_auc $criterion 3 1440 short
# ./jobs/performance/primer.sh flight_delays 'random' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh diabetes 'random' 1.0 accuracy $criterion 20 1440 short
# ./jobs/performance/primer.sh olympics 'random' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh skin 'random' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh census 'random' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh credit_card 'random' 1.0 average_precision $criterion 6 1440 short
# ./jobs/performance/primer.sh twitter 'random' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh gas_sensor 'random' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh synthetic 'random' 0.5 accuracy $criterion 40 1440 short
# ./jobs/performance/notune_primer.sh higgs 100 10 0.25 'random' accuracy $criterion 45 1440 short

# BORAT
# ./jobs/performance/primer.sh surgical 'borat' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh vaccine 'borat' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh adult 'borat' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/primer.sh bank_marketing 'borat' 1.0 roc_auc $criterion 3 1440 short
# ./jobs/performance/primer.sh flight_delays 'borat' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh diabetes 'borat' 1.0 accuracy $criterion 20 1440 short
# ./jobs/performance/primer.sh olympics 'borat' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh skin 'borat' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh census 'borat' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/primer.sh credit_card 'borat' 1.0 average_precision $criterion 6 1440 short
# ./jobs/performance/primer.sh twitter 'borat' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh gas_sensor 'borat' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh synthetic 'borat' 0.5 accuracy $criterion 60 4320 long
# ./jobs/performance/notune_primer.sh higgs 100 10 0.25 'borat' accuracy $criterion 45 1440 short

# Sklearn
./jobs/performance/primer.sh surgical 'sklearn' 1.0 accuracy $criterion 3 1440 short
./jobs/performance/primer.sh vaccine 'sklearn' 1.0 accuracy $criterion 3 1440 short
./jobs/performance/primer.sh adult 'sklearn' 1.0 accuracy $criterion 3 1440 short
./jobs/performance/primer.sh bank_marketing 'sklearn' 1.0 roc_auc $criterion 3 1440 short
./jobs/performance/primer.sh flight_delays 'sklearn' 1.0 roc_auc $criterion 20 1440 short
./jobs/performance/primer.sh diabetes 'sklearn' 1.0 accuracy $criterion 20 1440 short
./jobs/performance/primer.sh olympics 'sklearn' 1.0 roc_auc $criterion 25 4320 long
# ./jobs/performance/primer.sh skin 'sklearn' 1.0 roc_auc $criterion 25 4320 long
./jobs/performance/primer.sh census 'sklearn' 1.0 roc_auc $criterion 20 1440 short
./jobs/performance/primer.sh credit_card 'sklearn' 1.0 average_precision $criterion 6 1440 short
# ./jobs/performance/primer.sh twitter 'sklearn' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/primer.sh gas_sensor 'sklearn' 1.0 roc_auc $criterion 30 1440 short
./jobs/performance/primer.sh synthetic 'sklearn' 0.5 accuracy $criterion 60 4320 long
./jobs/performance/notune_primer.sh higgs 100 10 0.25 'sklearn' accuracy $criterion 45 1440 short

# Sklearn w/ bootstrap
# ./jobs/performance/bootstrap_primer.sh surgical 'sklearn' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/bootstrap_primer.sh vaccine 'sklearn' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/bootstrap_primer.sh adult 'sklearn' 1.0 accuracy $criterion 3 1440 short
# ./jobs/performance/bootstrap_primer.sh bank_marketing 'sklearn' 1.0 roc_auc $criterion 3 1440 short
# ./jobs/performance/bootstrap_primer.sh flight_delays 'sklearn' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/bootstrap_primer.sh diabetes 'sklearn' 1.0 accuracy $criterion 20 1440 short
# ./jobs/performance/bootstrap_primer.sh olympics 'sklearn' 1.0 roc_auc $criterion 25 4320 long
# ./jobs/performance/bootstrap_primer.sh skin 'sklearn' 1.0 roc_auc $criterion 25 4320 long
# ./jobs/performance/bootstrap_primer.sh census 'sklearn' 1.0 roc_auc $criterion 20 1440 short
# ./jobs/performance/bootstrap_primer.sh credit_card 'sklearn' 1.0 average_precision $criterion 6 1440 short
# ./jobs/performance/bootstrap_primer.sh twitter 'sklearn' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/bootstrap_primer.sh gas_sensor 'sklearn' 1.0 roc_auc $criterion 30 1440 short
# ./jobs/performance/bootstrap_primer.sh synthetic 'sklearn' 0.5 accuracy $criterion 40 1440 short
# ./jobs/performance/notune_bootstrap_primer.sh higgs 100 10 0.25 'sklearn' accuracy $criterion 45 1440 short
