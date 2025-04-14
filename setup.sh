#!/bin/bash

apt-get update
apt-get install -y unzip curl

# Instala o Google Chrome
curl -sSL https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -o chrome.deb
apt-get install -y ./chrome.deb

# Instala o ChromeDriver compatível
CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+')
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_VERSION")
curl -sSL "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip" -o chromedriver.zip
unzip chromedriver.zip
mv chromedriver /usr/bin/chromedriver
chmod +x /usr/bin/chromedriver
