criterion='gini'
operation='delete'

./jobs/update/primer.sh surgical 100 10 0.25 $criterion 3 1440 short
./jobs/update/primer.sh vaccine 250 20 -1 $criterion 3 1440 short
./jobs/update/primer.sh adult 250 20 -1 $criterion 3 1440 short
./jobs/update/primer.sh bank_marketing 100 10 0.25 $criterion 3 1440 short
./jobs/update/primer.sh flight_delays 250 20 -1 $criterion 18 1440 short
./jobs/update/primer.sh diabetes 250 20 -1 $criterion 18 1440 short
./jobs/update/primer.sh olympics 250 20 0.25 $criterion 35 1440 short
./jobs/update/primer.sh census 250 20 -1 $criterion 18 1440 short
./jobs/update/primer.sh credit_card 250 20 0.25 $criterion 9 1440 short
./jobs/update/primer.sh synthetic 250 20 0.25 $criterion 55 1440 short
./jobs/update/primer.sh higgs 100 10 0.25 $criterion 45 1440 short
