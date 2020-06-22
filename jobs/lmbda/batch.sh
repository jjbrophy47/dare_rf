sbatch jobs/lmbda/surgical.sh forest gini 1.0 0.01 0.01
# sbatch jobs/lmbda/surgical.sh forest entropy 1.0 0.01 0.01

# sbatch jobs/lmbda/adult.sh forest gini 1.0 0.01 0.01
# sbatch jobs/lmbda/adult.sh forest entropy 1.0 0.01 0.01

# sbatch jobs/lmbda/bank_marketing.sh forest gini 1.0 0.001 0.001 10
# sbatch jobs/lmbda/bank_marketing.sh forest entropy 1.0 0.001 0.001 5

# sbatch jobs/lmbda/flight_delays.sh forest gini 1.0 0.0001 0.0002
# sbatch jobs/lmbda/flight_delays.sh forest entropy 1.0 0.0001 0.0002

# sbatch jobs/lmbda/diabetes.sh forest gini 1.0 0.0001 0.0002
# sbatch jobs/lmbda/diabetes.sh forest entropy 1.0 0.0001 0.0002

# sbatch jobs/lmbda/skin.sh forest gini 1.0 0.00001 0.00002
# sbatch jobs/lmbda/skin.sh forest entropy 1.0 0.00001 0.00002

# sbatch jobs/lmbda/census.sh forest gini 1.0 0.00001 0.00002 10
# sbatch jobs/lmbda/census.sh forest entropy 1.0 0.00001 0.00002 20

# sbatch jobs/lmbda/twitter.sh forest gini 1.0 0.000001 0.000002
# sbatch jobs/lmbda/twitter.sh forest entropy 1.0 0.000001 0.000002

# sbatch jobs/lmbda/gas_sensor.sh forest gini 1.0 0.000001 0.000002
# sbatch jobs/lmbda/gas_sensor.sh forest entropy 1.0 0.000001 0.000002

sbatch jobs/lmbda/higgs.sh forest gini 0.05 0.0000001 0.0000002
# sbatch jobs/lmbda/higgs.sh forest entropy 0.05 0.0000001 0.0000002
