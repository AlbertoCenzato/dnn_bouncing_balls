import sys

import numpy as np
import imageio


if __name__ == '__main__':
    sample_array = np.load(sys.argv[1])
    seq_len = sample_array.shape[0]
    sample_array = np.reshape(sample_array, (seq_len, 60, 60, 1))
    border = np.ones((60,1,1))
    frames = [np.concatenate([np.squeeze(x, axis=0), border], axis=1) for x in np.split(sample_array, seq_len)]
    unrolled_sequence = np.concatenate(frames, axis=1)

    outfile = sys.argv[2] if len(sys.argv) > 2 else 'outfile.jpg'
    imageio.imwrite(outfile, unrolled_sequence)
