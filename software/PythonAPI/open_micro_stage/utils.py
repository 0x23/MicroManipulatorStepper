type LocalArray = list[float]
type LocalMatrix = list[LocalArray]


def generate_eye(size: int) -> LocalMatrix:
    """Generate a 2D list representing an eye pattern."""
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


def matrix_multiply(A: LocalMatrix, B: LocalMatrix) -> LocalMatrix:
    """Multiply two matrices A and B."""
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must equal number of rows in B.")

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


def matrix_dot_product(A: LocalMatrix, B: LocalArray) -> LocalArray:
    """Calculate the dot product of two matrices A and B."""
    if len(A[0]) != len(B):
        raise ValueError("Number of columns in A must equal number of elements in B.")

    result = [0 for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B)):
            result[i] += A[i][j] * B[j]

    return result
