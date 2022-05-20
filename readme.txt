Train on the LJSpeech Dataset

Preprocess the data by running preprocess.py (accepts the LJSpeech Dataset is downloaded, followed by the location where the processed data is supposed to be outputted to)

Then, run the train.py module (Accepts the location of the directory containing the data to train the model, the directory where the trained checkpoints shall be saved, and the directory where the partially trained checkpoint is saved)

Then, run generate.py (Accepts the location of the checkpoint, location of the spectrograms to fed to the vocoder, and finally, the location where the output is to be saved)

NOTE: Running via cmd, run as positional arguments.

References:
- https://github.com/mkotha/WaveRNN
- https://github.com/r9y9/wavenet_vocoder
- https://github.com/keithito/tacotron