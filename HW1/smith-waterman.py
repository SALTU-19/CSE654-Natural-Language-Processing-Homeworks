

# Importing Python packages
from enum import IntEnum
import numpy as np

# enum for the match, mismatch and gap scores


class Score(IntEnum):
    MATCH = 2
    MISMATCH = -2
    GAP = -1


def smith_waterman(seq1, seq2):
    # Creating the matrix
    matrix = np.zeros((len(seq1) + 1, len(seq2) + 1), dtype=int)
    # Filling the matrix
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            if seq1[i - 1] == seq2[j - 1]:
                matrix[i, j] = matrix[i - 1, j - 1] + Score.MATCH
            else:
                matrix[i, j] = max(
                    matrix[i - 1, j] + Score.GAP, matrix[i, j - 1] + Score.GAP, matrix[i - 1, j - 1] + Score.MISMATCH, 0)

    # print(matrix)
    # Finding the maximum value in the matrix
    max_value = np.amax(matrix)
    # traceback to find the alignment
    i, j = np.where(matrix == max_value)
    i = i[0]
    j = j[0]
    alignment1 = ""
    alignment2 = ""
    while matrix[i, j] != 0:
        if matrix[i - 1, j - 1] + 2 == matrix[i, j] and seq1[i - 1] == seq2[j - 1]:
            alignment1 = seq1[i - 1] + alignment1
            alignment2 = seq2[j - 1] + alignment2
            i -= 1
            j -= 1
        elif matrix[i - 1, j] - 1 == matrix[i, j]:
            alignment1 = seq1[i - 1] + alignment1
            alignment2 = "-" + alignment2
            i -= 1
        elif matrix[i, j - 1] - 1 == matrix[i, j]:
            alignment1 = "-" + alignment1
            alignment2 = seq2[j - 1] + alignment2
            j -= 1
        else:
            alignment1 = seq1[i - 1] + alignment1
            alignment2 = seq2[j - 1] + alignment2
            i -= 1
            j -= 1
    if alignment1 == alignment2 and len(alignment1) == len(seq1):
        print("Common sentence: " + alignment1)
    # print(matrix)


def compareTwoFiles(file1, file2):
    # read file1 line by line
    with open(file1, 'r', encoding="utf-8") as f:
        linesOfFile1 = f.readlines()
    # read file2 line by line
    with open(file2, 'r', encoding="utf-8") as f:
        linesOfFile2 = f.readlines()
    # compare each line of file1 with each line of file2
    for line1 in linesOfFile1:
        for line2 in linesOfFile2:
            # print("line1: " + line1)
            # print("line2: " + line2)
            if line1 != " " and line2 != " ":
                smith_waterman(line1, line2)


if __name__ == "__main__":
    # enter the file name input
    file1 = input("Enter the first file name(e.g. 1.txt): ")
    file2 = input("Enter the second file name: ")
    # compare two files
    compareTwoFiles("txts/" + file1, "txts/" + file2)
    # smith_waterman("gidiyor", "geliyor")
