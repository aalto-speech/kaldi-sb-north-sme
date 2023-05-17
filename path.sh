export PYTHONIOENCODING="utf-8:replace"
module load kaldi-strawberry
source venv/bin/activate
export PATH=$PATH:$PWD/utils
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:pychain/openfst/lib:$PWD
export LC_ALL=C # NOTE: Not compatible with Sami characters. Use temporary LC_ALL="fi_FI.utf8" for those
export PYTHONPATH=$PYTHONPATH:"$PWD"/pychain:"$PWD"/local/lm
module load variKN
module load sentencepiece
module load flac
