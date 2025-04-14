#!/bin/bash

set -e

apt-get update
apt-get install -y wget curl unzip gnupg

# Instalar Google Chrome
wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt install -y ./google-chrome-stable_current_amd64.deb || true

# Criar link simbólico
ln -fs /usr/bin/google-chrome /usr/local/bin/chrome

# Descobrir versão do Chrome instalada
CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+')
CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d '.' -f 1)

# Baixar Chromedriver compatível com a versão
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_MAJOR")
wget -q -O chromedriver.zip "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
unzip -o chromedriver.zip
chmod +x chromedriver
mv -f chromedriver /usr/local/bin/chromedriver

# Confirmar instalação
which chromedriver
chromedriver --version
