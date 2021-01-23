
# DART (Gini, Deletion)
./jobs/delete/dart_primer.sh 'surgical' 'gini' 100 20 10 3 1440 short
./jobs/delete/dart_primer.sh 'vaccine' 'gini' 100 20 10 6 1440 short
./jobs/delete/dart_primer.sh 'adult' 'gini' 100 20 10 3 1440 short
./jobs/delete/dart_primer.sh 'bank_marketing' 'gini' 100 20 10 3 1440 short
./jobs/delete/dart_primer.sh 'flight_delays' 'gini' 250 20 100 18 1440 short
./jobs/delete/dart_primer.sh 'diabetes' 'gini' 100 20 10 18 1440 short
./jobs/delete/dart_primer.sh 'no_show' 'gini' 100 20 100 18 1440 short
./jobs/delete/dart_primer.sh 'olympics' 'gini' 250 20 10 35 1440 short
./jobs/delete/dart_primer.sh 'census' 'gini' 100 20 100 18 1440 short
./jobs/delete/dart_primer.sh 'credit_card' 'gini' 250 20 1 9 1440 short
./jobs/delete/dart_primer.sh 'twitter' 'gini' 100 20 100 35 1440 short
./jobs/delete/dart_primer.sh 'synthetic' 'gini' 50 20 100 55 1440 short
./jobs/delete/dart_primer.sh 'higgs' 'gini' 50 10 100 55 1440 short
# ./jobs/delete/dart_primer.sh 'ctr' 'gini' 50 10 100 55 1440 short

# CEDAR (Gini, Deletion)
# ./jobs/update/cedar_primer.sh surgical 250 10 0.25 'gini' 'deletion' 3 1440 short
# ./jobs/update/cedar_primer.sh vaccine 250 20 -1 'gini' 'deletion' 18 1440 short
# ./jobs/update/cedar_primer.sh adult 250 20 -1 'gini' 'deletion' 3 1440 short
# ./jobs/update/cedar_primer.sh bank_marketing 250 10 0.25 'gini' 'deletion' 18 1440 short
# ./jobs/update/cedar_primer.sh flight_delays 250 20 -1 'gini' 'deletion' 18 1440 short
# ./jobs/update/cedar_primer.sh diabetes 250 20 -1 'gini' 'deletion' 18 1440 short
# ./jobs/update/cedar_primer.sh olympics 250 20 0.25 'gini' 'deletion' 33 1440 short
# ./jobs/update/cedar_primer.sh census 250 20 -1 'gini' 'deletion' 18 1440 short
# ./jobs/update/cedar_primer.sh credit_card 250 10 0.25 'gini' 'deletion' 9 1440 short
# ./jobs/update/cedar_primer.sh synthetic 250 20 0.25 'gini' 'deletion' 75 1440 short
# ./jobs/update/cedar_primer.sh higgs 100 10 0.25 'gini' 'deletion' 75 1440 short

# DART (Entropy, Deletion)
# ./jobs/update/dart_primer.sh surgical 250 10 0.25 'entropy' 'deletion' 3 1440 short
# ./jobs/update/dart_primer.sh vaccine 250 20 -1 'entropy' 'deletion' 3 1440 short
# ./jobs/update/dart_primer.sh adult 250 20 -1 'entropy' 'deletion' 3 1440 short
# ./jobs/update/dart_primer.sh bank_marketing 250 10 0.25 'entropy' 'deletion' 3 1440 short
# ./jobs/update/dart_primer.sh flight_delays 250 20 -1 'entropy' 'deletion' 18 1440 short
# ./jobs/update/dart_primer.sh diabetes 250 20 -1 'entropy' 'deletion' 18 1440 short
# ./jobs/update/dart_primer.sh olympics 250 20 0.25 'entropy' 'deletion' 35 1440 short
# ./jobs/update/dart_primer.sh census 250 20 -1 'entropy' 'deletion' 18 1440 short
# ./jobs/update/dart_primer.sh credit_card 250 10 0.25 'entropy' 'deletion' 9 1440 short
# ./jobs/update/dart_primer.sh synthetic 250 20 0.25 'entropy' 'deletion' 55 1440 short
# ./jobs/update/dart_primer.sh higgs 100 10 0.25 'entropy' 'deletion' 45 1440 short

# D-DART (Alternates, Gini, Deletion)
# ./jobs/update/alternate_primer.sh surgical 250 5 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh surgical 250 3 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh surgical 500 5 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh surgical 500 3 0.25 $criterion $operation 3 1440 short

# ./jobs/update/alternate_primer.sh vaccine 250 10 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh vaccine 250 5 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh vaccine 250 3 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh vaccine 500 10 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh vaccine 500 5 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh vaccine 500 3 -1 $criterion $operation 3 1440 short

# ./jobs/update/alternate_primer.sh adult 250 10 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh adult 250 5 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh adult 250 3 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh adult 500 10 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh adult 500 5 -1 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh adult 500 3 -1 $criterion $operation 3 1440 short

# ./jobs/update/alternate_primer.sh bank_marketing 250 5 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh bank_marketing 250 3 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh bank_marketing 500 5 0.25 $criterion $operation 3 1440 short
# ./jobs/update/alternate_primer.sh bank_marketing 500 3 0.25 $criterion $operation 3 1440 short

# ./jobs/update/alternate_primer.sh flight_delays 250 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh flight_delays 250 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh flight_delays 250 3 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh flight_delays 500 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh flight_delays 500 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh flight_delays 500 3 -1 $criterion $operation 20 1440 short

# ./jobs/update/alternate_primer.sh diabetes 250 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh diabetes 250 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh diabetes 250 3 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh diabetes 500 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh diabetes 500 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh diabetes 500 3 -1 $criterion $operation 20 1440 short

# ./jobs/update/alternate_primer.sh olympics 250 10 0.25 $criterion $operation 30 1440 short
# ./jobs/update/alternate_primer.sh olympics 250 5 0.25 $criterion $operation 30 1440 short
# ./jobs/update/alternate_primer.sh olympics 250 3 0.25 $criterion $operation 30 1440 short
# ./jobs/update/alternate_primer.sh olympics 500 10 0.25 $criterion $operation 30 1440 short
# ./jobs/update/alternate_primer.sh olympics 500 5 0.25 $criterion $operation 30 1440 short
# ./jobs/update/alternate_primer.sh olympics 500 3 0.25 $criterion $operation 30 1440 short

# ./jobs/update/alternate_primer.sh census 250 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh census 250 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh census 250 3 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh census 500 10 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh census 500 5 -1 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh census 500 3 -1 $criterion $operation 20 1440 short

# ./jobs/update/alternate_primer.sh credit_card 250 5 0.25 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh credit_card 250 3 0.25 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh credit_card 500 5 0.25 $criterion $operation 20 1440 short
# ./jobs/update/alternate_primer.sh credit_card 500 3 0.25 $criterion $operation 20 1440 short

# ./jobs/update/alternate_primer.sh synthetic 250 10 0.25 $criterion $operation 55 1440 short
# ./jobs/update/alternate_primer.sh synthetic 250 5 0.25 $criterion $operation 55 1440 short
# ./jobs/update/alternate_primer.sh synthetic 250 3 0.25 $criterion $operation 55 1440 short
# ./jobs/update/alternate_primer.sh synthetic 500 10 0.25 $criterion $operation 55 1440 short
# ./jobs/update/alternate_primer.sh synthetic 500 5 0.25 $criterion $operation 55 1440 short
# ./jobs/update/alternate_primer.sh synthetic 500 3 0.25 $criterion $operation 55 1440 short

# ./jobs/update/alternate_primer.sh higgs 100 5 0.25 $criterion $operation 45 1440 short
# ./jobs/update/alternate_primer.sh higgs 100 3 0.25 $criterion $operation 45 1440 short
# ./jobs/update/alternate_primer.sh higgs 250 5 0.25 $criterion $operation 45 1440 short
# ./jobs/update/alternate_primer.sh higgs 250 3 0.25 $criterion $operation 45 1440 short
