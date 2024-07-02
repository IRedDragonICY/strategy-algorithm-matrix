class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        self.data = data if data else [[0 for _ in range(cols)] for _ in range(rows)]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for addition.")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions for subtraction.")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __mul__(self, other):
        if self.cols != other.rows:
            raise ValueError("Incompatible dimensions for matrix multiplication.")

        n = max(self.rows, self.cols, other.rows, other.cols)
        next_power_of_two = 2**((n-1).bit_length())

        a_padded = Matrix(next_power_of_two, next_power_of_two)
        b_padded = Matrix(next_power_of_two, next_power_of_two)

        for i in range(self.rows):
            for j in range(self.cols):
                a_padded.data[i][j] = self.data[i][j]

        for i in range(other.rows):
            for j in range(other.cols):
                b_padded.data[i][j] = other.data[i][j]

        result_padded = strassen(a_padded, b_padded)

        result = Matrix(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                result.data[i][j] = result_padded.data[i][j]

        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    def determinant(self):
        if self.rows != self.cols:
            raise ValueError("Determinant is only defined for square matrices.")
        if self.rows == 1:
            return self.data[0][0]
        if self.rows == 2:
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            sub_matrix = Matrix(self.rows - 1, self.cols - 1)
            for i in range(1, self.rows):
                k = 0
                for l in range(self.cols):
                    if l != j:
                        sub_matrix.data[i - 1][k] = self.data[i][l]
                        k += 1
            det += ((-1) ** j) * self.data[0][j] * sub_matrix.determinant()
        return det

    def inverse(self):
        if self.rows != self.cols:
            raise ValueError("Inverse is only defined for square matrices.")
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is not invertible (determinant is zero).")
        adj = self.adjoint()
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = adj.data[i][j] / det
        return result

    def adjoint(self):
        if self.rows != self.cols:
            raise ValueError("Adjoint is only defined for square matrices.")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                sub_matrix = Matrix(self.rows - 1, self.cols - 1)
                for k in range(self.rows):
                    if k != i:
                        for l in range(self.cols):
                            if l != j:
                                sub_matrix.data[k if k < i else k - 1][l if l < j else l - 1] = self.data[k][l]
                result.data[i][j] = ((-1) ** (i + j)) * sub_matrix.determinant()
        return result.transpose()

    def __str__(self):
        return '\n'.join([' '.join([str(item) for item in row]) for row in self.data])

def split(matrix):
    row2, col2 = matrix.rows // 2, matrix.cols // 2
    a = Matrix(row2, col2)
    b = Matrix(row2, col2)
    c = Matrix(row2, col2)
    d = Matrix(row2, col2)

    for i in range(row2):
        for j in range(col2):
            a.data[i][j] = matrix.data[i][j]
            b.data[i][j] = matrix.data[i][j + col2]
            c.data[i][j] = matrix.data[i + row2][j]
            d.data[i][j] = matrix.data[i + row2][j + col2]

    return a, b, c, d

def strassen(x, y):
    if x.rows == 1:
        return Matrix(1, 1, [[x.data[0][0] * y.data[0][0]]])

    a, b, c, d = split(x)
    e, f, g, h = split(y)

    p1 = strassen(a, f - h)
    p2 = strassen(a + b, h)
    p3 = strassen(c + d, e)
    p4 = strassen(d, g - e)
    p5 = strassen(a + d, e + h)
    p6 = strassen(b - d, g + h)
    p7 = strassen(a - c, e + f)

    c11 = p5 + p4 - p2 + p6
    c12 = p1 + p2
    c21 = p3 + p4
    c22 = p1 + p5 - p3 - p7

    result = Matrix(x.rows, x.cols)
    for i in range(x.rows // 2):
        for j in range(x.cols // 2):
            result.data[i][j] = c11.data[i][j]
            result.data[i][j + x.cols // 2] = c12.data[i][j]
            result.data[i + x.rows // 2][j] = c21.data[i][j]
            result.data[i + x.rows // 2][j + x.cols // 2] = c22.data[i][j]

    return result

def get_matrix_input(rows, cols):
    print(f"Enter the elements for the {rows}x{cols} matrix (row-wise, separated by spaces):")
    data = []
    for i in range(rows):
        row = list(map(int, input().split()))
        if len(row) != cols:
            raise ValueError(f"Incorrect number of elements entered for row {i+1}")
        data.append(row)
    return Matrix(rows, cols, data)

if __name__ == "__main__":
    while True:
        print("\nMatrix Operations:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication (Strassen's algorithm)")
        print("4. Transpose")
        print("5. Determinant")
        print("6. Inverse")
        print("7. Adjoint")
        print("8. Exit")

        choice = input("Enter your choice (1-8): ")

        if choice in ('1', '2', '3'):
            try:
                rows1 = int(input("Enter number of rows for matrix 1: "))
                cols1 = int(input("Enter number of columns for matrix 1: "))
                matrix1 = get_matrix_input(rows1, cols1)

                rows2 = int(input("Enter number of rows for matrix 2: "))
                cols2 = int(input("Enter number of columns for matrix 2: "))
                matrix2 = get_matrix_input(rows2, cols2)

                if choice == '1':
                    result = matrix1 + matrix2
                elif choice == '2':
                    result = matrix1 - matrix2
                else:
                    result = matrix1 * matrix2
                print("Resultant Matrix:")
                print(result)
            except ValueError as e:
                print(f"Error: {e}")

        elif choice in ('4', '5', '6', '7'):
            try:
                rows = int(input("Enter number of rows for the matrix: "))
                cols = int(input("Enter number of columns for the matrix: "))
                matrix = get_matrix_input(rows, cols)

                if choice == '4':
                    result = matrix.transpose()
                elif choice == '5':
                    result = matrix.determinant()
                elif choice == '6':
                    result = matrix.inverse()
                else:
                    result = matrix.adjoint()
                print("Result:")
                print(result)
            except ValueError as e:
                print(f"Error: {e}")
        elif choice == '8':
            break
        else:
            print("Invalid choice. Please try again.")