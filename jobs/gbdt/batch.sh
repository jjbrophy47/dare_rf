model='lgb'

./jobs/gbdt/primer.sh surgical $model 1.0 accuracy 3 1440 short
./jobs/gbdt/primer.sh vaccine $model 1.0 accuracy 3 1440 short
./jobs/gbdt/primer.sh adult $model 1.0 accuracy 3 1440 short
./jobs/gbdt/primer.sh bank_marketing $model 1.0 roc_auc 3 1440 short
./jobs/gbdt/primer.sh flight_delays $model 1.0 roc_auc 18 1440 short
./jobs/gbdt/primer.sh diabetes $model 1.0 accuracy 18 1440 short
./jobs/gbdt/primer.sh olympics $model 1.0 roc_auc 18 1440 short
./jobs/gbdt/primer.sh census $model 1.0 roc_auc 18 1440 short
./jobs/gbdt/primer.sh credit_card $model 1.0 average_precision 7 1440 short
./jobs/gbdt/primer.sh synthetic $model 0.5 accuracy 30 1440 short
./jobs/gbdt/primer.sh higgs $model 0.05 accuracy 40 1440 short
