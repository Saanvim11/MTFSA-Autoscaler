#!/bin/bash
# deploy.sh — DEPLOY MTFSA TO AWS

echo "Starting MTFSA Deployment..."

# 1. Update system
sudo apt update -y && sudo apt upgrade -y

# 2. Install Python + pip
sudo apt install python3-pip python3-venv -y

# 3. Create project dir
mkdir -p /home/ubuntu/mtfsa
cd /home/ubuntu/mtfsa

# 4. Copy your code (you'll upload via SCP or Git)
# (We'll do this manually below)

# 5. Create venv
python3 -m venv venv
source venv/bin/activate

# 6. Install packages
pip install --upgrade pip
pip install -r requirements.txt

# 7. Run Streamlit on port 8501
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

echo "DEPLOYMENT COMPLETE!"
echo "Access at: http://$(curl -s ifconfig.me):8501"