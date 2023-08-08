#!/bin/bash
# Install SC2 and add the custom maps

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR/pymarl

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
        unzip -P iagreetotheeula SC2.4.10.zip
        rm -rf SC2.4.10.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding SMAC maps.'
MAP_DIR="$SC2PATH/Maps/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip SMAC_Maps.zip
mkdir "$MAP_DIR/SMAC_Maps"
mv *.SC2Map "$MAP_DIR/SMAC_Maps"
rm -rf SMAC_Maps.zip

echo 'StarCraft II and SMAC are installed.'

