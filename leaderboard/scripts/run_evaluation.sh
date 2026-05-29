#!/bin/bash

export CARLA_ROOT=/home/carla1000/InterFuser/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner

export LEADERBOARD_ROOT=/home/carla1000/InterFuser/leaderboard
export SCENARIO_RUNNER_ROOT=/home/carla1000/InterFuser/scenario_runner
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=2000 # same as the carla server port
export TM_PORT=2500 # port for traffic manager, required when spawning multiple servers/clients
export DEBUG_CHALLENGE=0
export REPETITIONS=1 # multiple evaluation runs
#export ROUTES=/home/carla1000/InterFuser/leaderboard/data/training_routes/routes_town05_long.xml
export ROUTES=/home/carla1000/InterFuser/leaderboard/data/42routes/42routes.xml
export TEAM_AGENT=/home/carla1000/InterFuser/leaderboard/team_code/interfuser_agent.py # agent
#export TEAM_AGENT=/home/carla1000/InterFuser/leaderboard/team_code/auto_pilot.py # agent
export TEAM_CONFIG=/home/carla1000/InterFuser/leaderboard/team_code/interfuser_config.py # model checkpoint, not required for expert
export CHECKPOINT_ENDPOINT=/home/carla1000/InterFuser/results/sample_resultvideo100020000.json # results file
#export SCENARIOS=/home/carla1000/InterFuser/leaderboard/data/scenarios/town05_all_scenarios.json
export SCENARIOS=/home/carla1000/InterFuser/leaderboard/data/42routes/42scenarios.json
export SAVE_PATH=/home/carla1000/InterFuser/data/eval # path for saving episodes while evaluating
export RESUME=True
#export DIGITAL_TWIN_ENABLED=true
#export DIGITAL_TWIN_HOST=192.168.0.140  # Use Computer B's IP from step 2
#export DIGITAL_TWIN_HOST="localhost"
#export DIGITAL_TWIN_PORT=9999
#export MAP_STREAM_ENABLED=1
#export MAP_STREAM_HOST=192.168.0.140   # IP of Machine B
#export MAP_STREAM_PORT=5005
#export HD_MAP_DOWNSAMPLE=3
export DIGITAL_TWIN_HOST=192.168.0.140
export DIGITAL_TWIN_PORT=9999
export DIGITAL_TWIN_ENABLED=true


python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}
