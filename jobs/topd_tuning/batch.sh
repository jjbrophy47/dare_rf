criterion='gini'

# ./jobs/topd_tuning/primer.sh surgical 250 10 0.25 1.0 accuracy $criterion 3 1440 short
# ./jobs/topd_tuning/primer.sh vaccine 250 20 -1 1.0 accuracy $criterion 3 1440 short
# ./jobs/topd_tuning/primer.sh adult 250 20 -1 1.0 accuracy $criterion 3 1440 short
# ./jobs/topd_tuning/primer.sh bank_marketing 250 10 0.25 1.0 roc_auc $criterion 3 1440 short
# ./jobs/topd_tuning/primer.sh flight_delays 250 20 -1 1.0 roc_auc $criterion 18 1440 short
# ./jobs/topd_tuning/primer.sh diabetes 250 20 -1 1.0 accuracy $criterion 18 1440 short
# ./jobs/topd_tuning/primer.sh olympics 250 20 0.25 1.0 roc_auc $criterion 35 4320 long
# ./jobs/topd_tuning/primer.sh census 250 20 -1 1.0 roc_auc $criterion 18 1440 short
# ./jobs/topd_tuning/primer.sh credit_card 250 10 0.25 1.0 average_precision $criterion 9 1440 short
./jobs/topd_tuning/primer.sh synthetic 250 20 0.25 0.5 accuracy $criterion 30 4320 long
# ./jobs/topd_tuning/primer.sh higgs 100 10 0.25 0.05 accuracy $criterion 40 1440 short
