# Large-Matrix-Multiplication-Using-OpenMP

# Assignment
In this assignment you are to develop an efficient large matrix multiplication algorithm in OpenMP. A prime criterion in the assessment of your assignment will be the efficiency of your implementation and the evidence you present to substantiate your claim that your implementations are efficient.

# Approach 
The purpose of the matrix multiplication is to take a matrix A of dimension m*n and another matrix B of dimension n*p and generate a matrix C of dimension m*p such that C = A * B.
I have implemented Strassen algorithm in OpenMP for the large matrix multiplication and compared it with sequential matrix multiplication algorithm to check the efficiency of the Strassen algorithm. A threshold value for matrix size is implemented in the code and the matrix size equal or above the threshold will compute Strassen algorithm for the matrix multiplication, otherwise, normal matrix multiplication is used. 

![alt text](http://www.brainkart.com/media/extra/fm3moQv.jpg)
