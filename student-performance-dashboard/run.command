#!/bin/bash
cd "$(dirname "$0")"

# Free up port 8501 if previously occupied
pkill -f "streamlit run app.py" 2>/dev/null || true
sleep 0.5

export MPLCONFIGDIR=/tmp/matplotlib
export PATH="/Library/Frameworks/Python.framework/Versions/3.15/bin:$PATH"

echo "=========================================================="
echo " 🎓 Starting Student Performance AI Dashboard..."
echo "=========================================================="
echo " Opening http://localhost:8501 in your default browser..."
echo ""

(sleep 2 && open "http://localhost:8501") &

# Pipe empty string so it automatically passes any onboarding prompts
echo "" | /Library/Frameworks/Python.framework/Versions/3.15/bin/python3 -m streamlit run app.py --server.port 8501 --server.headless false --browser.gatherUsageStats false
