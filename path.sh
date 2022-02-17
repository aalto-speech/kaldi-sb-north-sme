module load kaldi-strawberry
source venv/bin/activate
PATH=$PATH:$PWD/utils
export LC_ALL=C # NOTE: Not compatible with Sami characters. Use temporary LC_ALL="fi_FI.utf8" for those
