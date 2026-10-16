#!/usr/bin/env bash
set -e
echo "Pushing ATB (QuantPulse AI v5.2) to https://github.com/Anonymous-hss/ATB.git ..."
git branch -M main
git remote set-url origin https://github.com/Anonymous-hss/ATB.git || git remote add origin https://github.com/Anonymous-hss/ATB.git
git push -u origin main
echo "Successfully pushed to https://github.com/Anonymous-hss/ATB.git!"
