#!/bin/bash

# Atualiza o sistema e instala dependências
apt-get update
apt-get install -y wget curl unzip gnupg

# Instala o Google Chrome
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
apt-get install -y ./google-chrome-stable_current_amd64.deb || true

# Cria link simbólico para facilitar a localização do Chrome
ln -s /usr/bin/google-chrome /usr/local/bin/chrome

# Descobre a versão do Chrome instalada
CHROME_VERSION=$(google-chrome --version | grep -oP '\d+\.\d+\.\d+')
CHROME_MAJOR=$(echo $CHROME_VERSION | cut -d '.' -f 1)

# Baixa e instala o Chromedriver compatível
CHROMEDRIVER_VERSION=$(curl -s "https://chromedriver.storage.googleapis.com/LATEST_RELEASE_$CHROME_MAJOR")
wget -O chromedriver.zip "https://chromedriver.storage.googleapis.com/$CHROMEDRIVER_VERSION/chromedriver_linux64.zip"
unzip chromedriver.zip
chmod +x chromedriver
mv chromedriver /usr/local/bin/
