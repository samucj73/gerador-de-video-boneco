#!/bin/bash

# Atualiza repositórios e instala dependências
apt-get update
apt-get install -y unzip curl

# Instalar o Google Chrome
mkdir -p .render/chrome
cd .render/chrome
curl -O https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install -y ./google-chrome-stable_current_amd64.deb || true
cd ../..

# Descobrir a versão principal do Chrome
CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+')
CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d '.' -f 1)

# Instalar o Chromedriver compatível
mkdir -p .render/chromedriver
cd .render/chromedriver
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_MAJOR")
curl -O "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
unzip chromedriver_linux64.zip
chmod +x chromedriver
cd ../..
