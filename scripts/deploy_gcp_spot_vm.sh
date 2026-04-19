#!/bin/bash
# Quick deploy script for GCP Spot VM training
# Usage: ./deploy_gcp_spot_vm.sh --bucket your-bucket --epochs 100

set -e

# Default config
ZONE="us-central1-a"
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"  # Cheapest: ~$0.10/hr
GPU_COUNT=1
VM_NAME="ssis-training"
COMPONENT="collision"
EPOCHS=100
CHECKPOINT_EVERY=10
GCS_BUCKET=""
GITHUB_REPO=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --bucket)
      GCS_BUCKET="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --checkpoint-every)
      CHECKPOINT_EVERY="$2"
      shift 2
      ;;
    --component)
      COMPONENT="$2"
      shift 2
      ;;
      --gpu)
      GPU_TYPE="$2"
      shift 2
      ;;
    --vm-name)
      VM_NAME="$2"
      shift 2
      ;;
    --zone)
      ZONE="$2"
      shift 2
      ;;
    --repo)
      GITHUB_REPO="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --bucket BUCKET          GCS bucket for checkpoints (required)"
      echo "  --epochs N               Training epochs (default: 100)"
      echo "  --checkpoint-every N     Checkpoint every N epochs (default: 10)"
      echo "  --component NAME         Component to train: collision|navic|reentry (default: collision)"
      echo "  --gpu TYPE               GPU type: nvidia-tesla-t4|v100|a100 (default: t4)"
      echo "  --vm-name NAME           VM instance name (default: ssis-training)"
      echo "  --zone ZONE              GCP zone (default: us-central1-a)"
      echo "  --repo URL               GitHub repo URL to clone"
      echo "  --help                   Show this help"
      echo ""
      echo "Examples:"
      echo "  $0 --bucket my-ssis-checkpoints --epochs 100"
      echo "  $0 --bucket my-bucket --gpu nvidia-tesla-v100 --epochs 200"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate
if [ -z "$GCS_BUCKET" ]; then
  echo "Error: --bucket is required"
  echo "Run with --help for usage"
  exit 1
fi

# Check if bucket exists
echo "Checking GCS bucket..."
if ! gsutil ls "gs://$GCS_BUCKET" &>/dev/null; then
  echo "Creating bucket: gs://$GCS_BUCKET"
  gsutil mb -l "${ZONE%-*}" "gs://$GCS_BUCKET" || true
fi

echo ""
echo "========================================"
echo "Deploying SSIS Spot VM"
echo "========================================"
echo "VM Name: $VM_NAME"
echo "Zone: $ZONE"
echo "Machine: $MACHINE_TYPE + $GPU_TYPE x $GPU_COUNT"
echo "Component: $COMPONENT"
echo "Epochs: $EPOCHS"
echo "GCS Bucket: gs://$GCS_BUCKET"
echo "Cost: ~\$0.10-0.75/hr (spot pricing)"
echo "========================================"
echo ""

# Check if VM already exists
if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" &>/dev/null; then
  echo "⚠️  VM '$VM_NAME' already exists"
  read -p "Delete and recreate? (y/N): " -n 1 -r
  echo
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting existing VM..."
    gcloud compute instances delete "$VM_NAME" --zone="$ZONE" --quiet
  else
    echo "Using existing VM. To connect: gcloud compute ssh $VM_NAME --zone=$ZONE"
    exit 0
  fi
fi

# Create startup script with embedded config
STARTUP_SCRIPT=$(mktemp)
cat > "$STARTUP_SCRIPT" <<EOF
#!/bin/bash
export GCS_BUCKET="$GCS_BUCKET"
export COMPONENT="$COMPONENT"
export EPOCHS="$EPOCHS"
export CHECKPOINT_EVERY="$CHECKPOINT_EVERY"
export GITHUB_REPO="$GITHUB_REPO"

$(cat scripts/gcp_spot_startup.sh)
EOF

echo "Creating spot VM..."
gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
  --image-family="ubuntu-2204-lts" \
  --image-project="ubuntu-os-cloud" \
  --boot-disk-size="100GB" \
  --boot-disk-type="pd-ssd" \
  --metadata-from-file="startup-script=$STARTUP_SCRIPT" \
  --provisioning-model="SPOT" \
  --maintenance-policy="TERMINATE" \
  --scopes="cloud-platform" \
  --service-account="default" \
  --quiet

rm "$STARTUP_SCRIPT"

echo ""
echo "✅ Spot VM created: $VM_NAME"
echo ""
echo "Next steps:"
echo "  1. Wait 3-5 minutes for setup (GPU driver install + reboot)"
echo "  2. Monitor logs: gcloud compute tail-serial-port-output $VM_NAME --zone=$ZONE"
echo "  3. SSH to VM: gcloud compute ssh $VM_NAME --zone=$ZONE"
echo "  4. Watch training: tail -f /var/log/ssis-training.log"
echo ""
echo "Checkpoint storage: gs://$GCS_BUCKET/checkpoints/"
echo "Models output: gs://$GCS_BUCKET/models/"
echo ""
echo "If VM is preempted, restart it:"
echo "  gcloud compute instances start $VM_NAME --zone=$ZONE"
echo "It will automatically resume from last checkpoint."
echo ""
echo "Cost monitoring:"
echo "  Approximate cost: ~\$0.10-0.75/hr depending on GPU type"
echo "  100 epochs on T4: ~2-3 hours = ~\$0.20-0.30"
