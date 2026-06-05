this backup serves as an archive.

feature extraction via CQT in this directory is not converted to decibel, meaning CQT features use raw amplitude values.

what was done
- cqt extraction
- np.abs

what SHOULD be done
- cqt extraction
- np.abs
- librosa.amplitude_to_db

this makes for standard in most of MIR researches with logarithmic distribution
