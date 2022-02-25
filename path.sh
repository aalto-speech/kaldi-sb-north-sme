export PYTHONIOENCODING="utf-8"
module load kaldi-strawberry
source venv/bin/activate
export PATH=$PATH:$PWD/utils
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:pychain/openfst/lib
export LC_ALL=C # NOTE: Not compatible with Sami characters. Use temporary LC_ALL="fi_FI.utf8" for those
export PYTHONPATH=$PYTHONPATH:"$PWD"/pychain
