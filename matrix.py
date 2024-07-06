class Matrix:
    def __init__(self, nums):
        self.mat = nums

        self.width = len(nums[0])
        self.height = len(nums)

    get_row = lambda self, height_inx: self.mat[height_inx]
    get_col = lambda self, width_inx: [self.mat[i][width_inx] for i in range(self.height)]

    def transpose(self):
        new_mat = []

        for i in range(self.width):
            col = [self.mat[x][i] for x in range(self.height)]
            new_mat.append(col)

        self.mat = new_mat
        self.width, self.height = self.height, self.width

    def scalar_mult(self, scalar):
        for i in range(self.width):
            for x in range(self.height):
                self.mat[i][x] *= scalar

    def __div__(self, scalar):
        pass

    def __repr__(self) -> str:
        output = ""
        for i in self.mat:
            output += f"[ {" ".join([str(x) for x in i])} ]\n"

        return output[:-1]

    @staticmethod
    def mat_mul(a, b):
        width = b.width
        height = a.height

        new_mat = [[0 for _ in range(width)] for _ in range(height)]

        for i in range(width):
            for x in range(height):
                row = a.get_row(x)
                col = b.get_col(i)

                weighted_sum = sum([j * k for j, k in zip(row, col)])

                new_mat[x][i] = weighted_sum

        return Matrix(new_mat) 
    

class SquareMatrix(Matrix):
    def __init__(self, nums):
        super().__init__(nums)

        if self.width != self.height:
            raise Exception("Width and height must be same for square matrix")

    def get_minor(self, width_inx, height_inx):
        minor_size = self.width - 1
        minor = [[0 for _ in range(minor_size)] for _ in range(minor_size)]

        num_added = 0
        for i in range(self.height):
            for x in range(self.width):
                if x != width_inx and i != height_inx:
                    place_width_inx = num_added % minor_size
                    place_height_inx = num_added // minor_size

                    minor[place_height_inx][place_width_inx] = self.mat[i][x]

                    num_added += 1

        minor_mat = SquareMatrix(minor)

        return minor_mat.det()
    
    def det(self):
        if self.width == 2:
            #base case
            return self.mat[0][0] * self.mat[1][1] - self.mat[1][0] * self.mat[0][1]
        
        determinant = 0
        for i in range(self.width):
            minor = self.get_minor(i, 0)
            add = self.mat[0][i] * minor
            
            if i % 2 == 0:
                determinant += add
            else:
                determinant -= add

        return determinant
    
    def inverse(self):
        if self.width == 2:
            #special case
            new_mat = [
                [self.mat[1][1], -self.mat[0][1]],
                [-self.mat[1][0], self.mat[0][0]]
            ]

            inv_mat = SquareMatrix(new_mat)
            inv_mat.scalar_mult(1 / self.det())

            return inv_mat

        minor_mat = [[self.get_minor(i, x) for i in range(self.width)] for x in range(self.height)]

        cofactor_mat = []
        for i in range(self.height):
            cofactor_mat.append([])

            for x in range(self.width):
                mult = 1 if (i + x) % 2 == 0 else -1

                cofactor_mat[i].append(minor_mat[i][x] * mult)

        inv_mat = SquareMatrix(cofactor_mat)

        inv_mat.transpose()
        inv_mat.scalar_mult(1 / self.det())

        return inv_mat
    

class Vector(Matrix):
    def __init__(self, nums):
        super().__init__(nums)

        self.transpose()  #convert to column vector

        if len(nums) >= 2:
            self.x = nums[0]
            self.y = nums[1]
        if len(nums) >= 3:
            self.z = nums[2]