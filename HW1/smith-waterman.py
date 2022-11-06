

# Importing Python packages
from enum import IntEnum
import numpy as np


def smith_waterman(seq1, seq2):
    # Creating the matrix
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    # Filling the matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1] + 2
            else:
                matrix[i, j] = max(
                    matrix[i - 1, j] - 1, matrix[i, j - 1] - 1, matrix[i - 1, j - 1] - 2, 0)

    print(matrix)


if __name__ == "__main__":

    smith_waterman("GCACGCTG", "GACGCGCG")
