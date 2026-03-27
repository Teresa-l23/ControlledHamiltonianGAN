#!/bin/bash
# Run SympNet experiments in batches across multiple GPUs.
# Expected workflow:
#   1) python set_up.py
#   2) bash run_sym.sh

# Configuration
BASE_DIR="/home/jiayinliu/projects/SympNet"
GPU_IDS=(0 1)  # Add more GPU IDs if you have more GPUs
MAX_PARALLEL=2  # Number of experiments to run in parallel
CONDA_SH="/home/jiayinliu/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="hgan"
PYTHON_BIN=""

# Color codes for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# System configurations
declare -A SYSTEMS
SYSTEMS=(
    # ["MS"]="mass_spring"
    ["P"]="pendulum"
    ["DP"]="double_pendulum"
    ["TB1"]="two_body"
    # ["THB2"]="three_body"
)

# Generator frames to test
GENERATOR_FRAMES=(30)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SympNet Parallel Experiments Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Base directory: ${BASE_DIR}"
echo -e "Available GPUs: ${GPU_IDS[@]}"
echo -e "Max parallel experiments: ${MAX_PARALLEL}"
echo ""

if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

if [ -z "${PYTHON_BIN}" ]; then
    PYTHON_BIN=$(which python3)
fi

echo -e "Python binary: ${PYTHON_BIN}"
echo ""

# Create main logs directory
mkdir -p "${BASE_DIR}/logs"

# Create array of all experiments
EXPERIMENTS=()
for system_suffix in "${!SYSTEMS[@]}"; do
    for gen_frames in "${GENERATOR_FRAMES[@]}"; do
        EXPERIMENTS+=("${system_suffix}:${gen_frames}")
    done
done

TOTAL_EXPERIMENTS=${#EXPERIMENTS[@]}
echo -e "${GREEN}Total experiments: ${TOTAL_EXPERIMENTS}${NC}"
echo -e "${YELLOW}Running ${MAX_PARALLEL} experiments in parallel${NC}"
echo ""
read -p "Press Enter to start, or Ctrl+C to cancel..."
echo ""

# Start time
START_TIME=$(date +%s)
echo -e "${GREEN}Starting experiments at $(date)${NC}" | tee "${BASE_DIR}/logs/run_sympnet_experiments.log"

# Function to run a single experiment in background
run_experiment_bg() {
    local system_suffix=$1
    local gen_frames=$2
    local gpu_id=$3
    local exp_num=$4
    
    local system_name="${SYSTEMS[$system_suffix]}"
    local folder_name="ControlledHGAN_${system_suffix}"
    local folder_path="${BASE_DIR}/${folder_name}"
    local config_file="hgan/configuration_${gen_frames}.ini"
    local log_dir="${BASE_DIR}/logs/${system_suffix}"
    local log_file="${log_dir}/sympnet_${system_suffix}_${gen_frames}.log"
    
    mkdir -p "${log_dir}"
    
    echo -e "${BLUE}[${exp_num}/${TOTAL_EXPERIMENTS}] Starting: ${system_name} (${gen_frames} frames, sympnet) on GPU ${gpu_id}${NC}"

    if [ ! -d "${folder_path}" ]; then
        echo -e "${RED}✗ Missing folder: ${folder_path}. Run: python set_up.py${NC}"
        return 1
    fi

    if [ ! -f "${folder_path}/src/${config_file}" ]; then
        echo -e "${RED}✗ Missing config: ${folder_path}/src/${config_file}. Run: python set_up.py${NC}"
        return 1
    fi
    
    # Log start
    {
        echo "========================================"
        echo "Experiment: ${system_name} (${gen_frames} frames, sympnet)"
        echo "Started at: $(date)"
        echo "GPU: ${gpu_id}"
        echo "========================================"
        echo ""
    } > "${log_file}"
    
    # Run experiment
    cd "${folder_path}/src" && \
    CUDA_VISIBLE_DEVICES=${gpu_id} "${PYTHON_BIN}" -m hgan run --config-path ${config_file} >> "${log_file}" 2>&1
    
    local exit_code=$?
    
    # Log completion
    {
        echo ""
        echo "========================================"
        echo "Completed at: $(date)"
        echo "Exit code: ${exit_code}"
        echo "========================================"
    } >> "${log_file}"
    
    if [ ${exit_code} -eq 0 ]; then
        echo -e "${GREEN}✓ [${exp_num}/${TOTAL_EXPERIMENTS}] SUCCESS: ${system_name} (${gen_frames} frames)${NC}"
    else
        echo -e "${RED}✗ [${exp_num}/${TOTAL_EXPERIMENTS}] FAILED: ${system_name} (${gen_frames} frames)${NC}"
    fi
    
    cd "${BASE_DIR}"
}

# Run experiments in parallel batches
exp_num=0
for ((i=0; i<${#EXPERIMENTS[@]}; i+=MAX_PARALLEL)); do
    # Start a batch of experiments
    pids=()
    
    for ((j=0; j<MAX_PARALLEL && i+j<${#EXPERIMENTS[@]}; j++)); do
        exp_num=$((exp_num + 1))
        exp="${EXPERIMENTS[$((i+j))]}"
        system_suffix="${exp%:*}"
        gen_frames="${exp#*:}"
        gpu_id="${GPU_IDS[$((j % ${#GPU_IDS[@]}))]}"
        
        run_experiment_bg "${system_suffix}" "${gen_frames}" "${gpu_id}" "${exp_num}" &
        pids+=($!)
    done
    
    # Wait for batch to complete
    echo -e "${YELLOW}Waiting for batch to complete...${NC}"
    for pid in "${pids[@]}"; do
        wait ${pid}
    done
    
    echo -e "${GREEN}Batch completed. Moving to next batch...${NC}"
    echo ""
done

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments completed!${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
