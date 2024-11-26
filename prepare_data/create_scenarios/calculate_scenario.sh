#!/bin/bash

start_year=1984
end_year=2019

python create_scenarios.py --config_file config_files/Configuration_scenario_aragonite_no_effect.yaml --start_year ${start_year} --end_year ${end_year} &
python create_scenarios.py --config_file config_files/Configuration_scenario_aragonite_constant_no_extremes.yaml --start_year ${start_year} --end_year ${end_year} &
python create_scenarios.py --config_file config_files/Configuration_scenario_aragonite_hindcast_no_extremes.yaml --start_year ${start_year} --end_year ${end_year} &
python create_scenarios.py --config_file config_files/Configuration_scenario_temperature_no_effect.yaml --start_year ${start_year} --end_year ${end_year} &
python create_scenarios.py --config_file config_files/Configuration_scenario_temperature_hindcast_no_extremes.yaml --start_year ${start_year} --end_year ${end_year} &
