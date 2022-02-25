# North sami models (Kaldi+SpeechBrain+Wav2Vec2)

## PyChain
To install pychain on Aalto Triton cluster, clone PyChain, check local/chain/pychain-install-env.sh to get the correct env,
start a GPU node job, make openfst, then cd openfst_binding and pip install . , and finally cd ../pychain_binding and pip install .

Also remember to add pychain to PYTHONPATH and the openfst libs to LD_LIBRARY_PATH
