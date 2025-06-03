@echo off

:: Activate the conda environment for ScaleX-CLI
CALL "C:\ProgramData\<your-anaconda-distributio-name>\Scripts\activate.bat" ScaleX-CLI

:: Navigate to the ScaleX directory (Change path according to yours)
cd /D <path-to-your-ScaleC-CLI>

:: Run ScaleX-CLI
python inference_scalex.py -i Input -o Output -f v1.4 -b x4 -s 4