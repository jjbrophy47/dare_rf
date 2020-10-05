criterion='gini'

./jobs/cleaning/primer.sh surgical 250 10 0.25 $criterion 9 1440 short
./jobs/cleaning/primer.sh vaccine 250 20 -1 $criterion 9 1440 short
./jobs/cleaning/primer.sh adult 250 20 -1 $criterion 9 1440 short
./jobs/cleaning/primer.sh bank_marketing 250 10 0.25 $criterion 9 1440 short
./jobs/cleaning/primer.sh flight_delays 250 20 -1 $criterion 18 1440 short
./jobs/cleaning/primer.sh diabetes 250 20 -1 $criterion 18 1440 short
./jobs/cleaning/primer.sh olympics 250 20 0.25 $criterion 45 1440 short
./jobs/cleaning/primer.sh census 250 20 -1 $criterion 45 1440 short
./jobs/cleaning/primer.sh credit_card 250 20 0.25 $criterion 45 1440 short
./jobs/cleaning/primer.sh synthetic 250 20 0.25 $criterion 55 1440 short
./jobs/cleaning/primer.sh higgs 100 10 0.25 $criterion 45 1440 short
