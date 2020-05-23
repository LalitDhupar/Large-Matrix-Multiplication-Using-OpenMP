# Large-Matrix-Multiplication-Using-OpenMP

# Assignment
In this assignment you are to develop an efficient large matrix multiplication algorithm in OpenMP. A prime criterion in the assessment of your assignment will be the efficiency of your implementation and the evidence you present to substantiate your claim that your implementations are efficient.

# Approach 
The purpose of the matrix multiplication is to take a matrix A of dimension m*n and another matrix B of dimension n*p and generate a matrix C of dimension m*p such that C = A * B.
I have implemented Strassen algorithm in OpenMP for the large matrix multiplication and compared it with sequential matrix multiplication algorithm to check the efficiency of the Strassen algorithm. A threshold value for matrix size is implemented in the code and the matrix size equal or above the threshold will compute Strassen algorithm for the matrix multiplication, otherwise, normal matrix multiplication is used. 

Let A, B be two square matrices over a ring R. We want to calculate the matrix product C as

{\displaystyle \mathbf {C} =\mathbf {A} \mathbf {B} \qquad \mathbf {A} ,\mathbf {B} ,\mathbf {C} \in R^{2^{n}\times 2^{n}}}\mathbf {C} =\mathbf {A} \mathbf {B} \qquad \mathbf {A} ,\mathbf {B} ,\mathbf {C} \in R^{2^{n}\times 2^{n}}
If the matrices A, B are not of type 2n × 2n we fill the missing rows and columns with zeros.

We partition A, B and C into equally sized block matrices

{\displaystyle \mathbf {A} ={\begin{bmatrix}\mathbf {A} _{1,1}&\mathbf {A} _{1,2}\\\mathbf {A} _{2,1}&\mathbf {A} _{2,2}\end{bmatrix}}{\mbox{ , }}\mathbf {B} ={\begin{bmatrix}\mathbf {B} _{1,1}&\mathbf {B} _{1,2}\\\mathbf {B} _{2,1}&\mathbf {B} _{2,2}\end{bmatrix}}{\mbox{ , }}\mathbf {C} ={\begin{bmatrix}\mathbf {C} _{1,1}&\mathbf {C} _{1,2}\\\mathbf {C} _{2,1}&\mathbf {C} _{2,2}\end{bmatrix}}}\mathbf {A} ={\begin{bmatrix}\mathbf {A} _{1,1}&\mathbf {A} _{1,2}\\\mathbf {A} _{2,1}&\mathbf {A} _{2,2}\end{bmatrix}}{\mbox{ , }}\mathbf {B} ={\begin{bmatrix}\mathbf {B} _{1,1}&\mathbf {B} _{1,2}\\\mathbf {B} _{2,1}&\mathbf {B} _{2,2}\end{bmatrix}}{\mbox{ , }}\mathbf {C} ={\begin{bmatrix}\mathbf {C} _{1,1}&\mathbf {C} _{1,2}\\\mathbf {C} _{2,1}&\mathbf {C} _{2,2}\end{bmatrix}}
with

{\displaystyle \mathbf {A} _{i,j},\mathbf {B} _{i,j},\mathbf {C} _{i,j}\in R^{2^{n-1}\times 2^{n-1}}}\mathbf {A} _{i,j},\mathbf {B} _{i,j},\mathbf {C} _{i,j}\in R^{2^{n-1}\times 2^{n-1}}.
The naive algorithm would be:

{\displaystyle \mathbf {C} _{1,1}=\mathbf {A} _{1,1}\mathbf {B} _{1,1}+\mathbf {A} _{1,2}\mathbf {B} _{2,1}}\mathbf {C} _{1,1}=\mathbf {A} _{1,1}\mathbf {B} _{1,1}+\mathbf {A} _{1,2}\mathbf {B} _{2,1}
{\displaystyle \mathbf {C} _{1,2}=\mathbf {A} _{1,1}\mathbf {B} _{1,2}+\mathbf {A} _{1,2}\mathbf {B} _{2,2}}\mathbf {C} _{1,2}=\mathbf {A} _{1,1}\mathbf {B} _{1,2}+\mathbf {A} _{1,2}\mathbf {B} _{2,2}
{\displaystyle \mathbf {C} _{2,1}=\mathbf {A} _{2,1}\mathbf {B} _{1,1}+\mathbf {A} _{2,2}\mathbf {B} _{2,1}}\mathbf {C} _{2,1}=\mathbf {A} _{2,1}\mathbf {B} _{1,1}+\mathbf {A} _{2,2}\mathbf {B} _{2,1}
{\displaystyle \mathbf {C} _{2,2}=\mathbf {A} _{2,1}\mathbf {B} _{1,2}+\mathbf {A} _{2,2}\mathbf {B} _{2,2}}\mathbf {C} _{2,2}=\mathbf {A} _{2,1}\mathbf {B} _{1,2}+\mathbf {A} _{2,2}\mathbf {B} _{2,2}
With this construction we have not reduced the number of multiplications. We still need 8 multiplications to calculate the Ci,j matrices, the same number of multiplications we need when using standard matrix multiplication.

The Strassen algorithm defines instead new matrices:

{\displaystyle \mathbf {M} _{1}:=(\mathbf {A} _{1,1}+\mathbf {A} _{2,2})(\mathbf {B} _{1,1}+\mathbf {B} _{2,2})}\mathbf {M} _{1}:=(\mathbf {A} _{1,1}+\mathbf {A} _{2,2})(\mathbf {B} _{1,1}+\mathbf {B} _{2,2})
{\displaystyle \mathbf {M} _{2}:=(\mathbf {A} _{2,1}+\mathbf {A} _{2,2})\mathbf {B} _{1,1}}\mathbf {M} _{2}:=(\mathbf {A} _{2,1}+\mathbf {A} _{2,2})\mathbf {B} _{1,1}
{\displaystyle \mathbf {M} _{3}:=\mathbf {A} _{1,1}(\mathbf {B} _{1,2}-\mathbf {B} _{2,2})}\mathbf {M} _{3}:=\mathbf {A} _{1,1}(\mathbf {B} _{1,2}-\mathbf {B} _{2,2})
{\displaystyle \mathbf {M} _{4}:=\mathbf {A} _{2,2}(\mathbf {B} _{2,1}-\mathbf {B} _{1,1})}\mathbf {M} _{4}:=\mathbf {A} _{2,2}(\mathbf {B} _{2,1}-\mathbf {B} _{1,1})
{\displaystyle \mathbf {M} _{5}:=(\mathbf {A} _{1,1}+\mathbf {A} _{1,2})\mathbf {B} _{2,2}}\mathbf {M} _{5}:=(\mathbf {A} _{1,1}+\mathbf {A} _{1,2})\mathbf {B} _{2,2}
{\displaystyle \mathbf {M} _{6}:=(\mathbf {A} _{2,1}-\mathbf {A} _{1,1})(\mathbf {B} _{1,1}+\mathbf {B} _{1,2})}\mathbf {M} _{6}:=(\mathbf {A} _{2,1}-\mathbf {A} _{1,1})(\mathbf {B} _{1,1}+\mathbf {B} _{1,2})
{\displaystyle \mathbf {M} _{7}:=(\mathbf {A} _{1,2}-\mathbf {A} _{2,2})(\mathbf {B} _{2,1}+\mathbf {B} _{2,2})}\mathbf {M} _{7}:=(\mathbf {A} _{1,2}-\mathbf {A} _{2,2})(\mathbf {B} _{2,1}+\mathbf {B} _{2,2})
only using 7 multiplications (one for each Mk) instead of 8. We may now express the Ci,j in terms of Mk:

{\displaystyle \mathbf {C} _{1,1}=\mathbf {M} _{1}+\mathbf {M} _{4}-\mathbf {M} _{5}+\mathbf {M} _{7}}\mathbf {C} _{1,1}=\mathbf {M} _{1}+\mathbf {M} _{4}-\mathbf {M} _{5}+\mathbf {M} _{7}
{\displaystyle \mathbf {C} _{1,2}=\mathbf {M} _{3}+\mathbf {M} _{5}}\mathbf {C} _{1,2}=\mathbf {M} _{3}+\mathbf {M} _{5}
{\displaystyle \mathbf {C} _{2,1}=\mathbf {M} _{2}+\mathbf {M} _{4}}\mathbf {C} _{2,1}=\mathbf {M} _{2}+\mathbf {M} _{4}
{\displaystyle \mathbf {C} _{2,2}=\mathbf {M} _{1}-\mathbf {M} _{2}+\mathbf {M} _{3}+\mathbf {M} _{6}}\mathbf {C} _{2,2}=\mathbf {M} _{1}-\mathbf {M} _{2}+\mathbf {M} _{3}+\mathbf {M} _{6}
