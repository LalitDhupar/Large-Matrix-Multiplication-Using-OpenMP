# Large-Matrix-Multiplication-Using-OpenMP

# Assignment
In this assignment you are to develop an efficient large matrix multiplication algorithm in OpenMP. A prime criterion in the assessment of your assignment will be the efficiency of your implementation and the evidence you present to substantiate your claim that your implementations are efficient.

# Approach 
The purpose of the matrix multiplication is to take a matrix A of dimension m*n and another matrix B of dimension n*p and generate a matrix C of dimension m*p such that C = A * B.
I have implemented Strassen algorithm in OpenMP for the large matrix multiplication and compared it with sequential matrix multiplication algorithm to check the efficiency of the Strassen algorithm. A threshold value for matrix size is implemented in the code and the matrix size equal or above the threshold will compute Strassen algorithm for the matrix multiplication, otherwise, normal matrix multiplication is used. 

![alt text](http://www.brainkart.com/media/extra/fm3moQv.jpg)

# Results and Performance matrix:
When the matrix size was small both the Sequential and Strassen algorithms took almost similar time to multiply the matrices. However, when the matrix size increased Strassen algorithm outperformed the sequential algorithm significantly.

<table class="tg">
  <tr>
    <th class="tg-yw4l"><b>Matrix Size</b></th>
    <th class="tg-yw4l"><b>Sequential Time(sec)</b></th>
    <th class="tg-yw4l"><b>Strassen Time(sec)</b></th>
  </tr>
  <tr>
    <td class="tg-yw4l">128</td>
    <td class="tg-yw4l">0.035</td>
    <td class="tg-yw4l">0.029</td>
  </tr>
  <tr>
    <td class="tg-yw4l">256</td>
    <td class="tg-yw4l">0.142</td>
    <td class="tg-yw4l">0.095</td>
  </tr>
   <tr>
    <td class="tg-yw4l">512</td>
    <td class="tg-yw4l">0.627</td>
    <td class="tg-yw4l">0.31</td>
  </tr>
   <tr>
    <td class="tg-yw4l">1024</td>
    <td class="tg-yw4l">4.22</td>
    <td class="tg-yw4l">1.933</td>
  </tr>
   <tr>
    <td class="tg-yw4l">2048</td>
    <td class="tg-yw4l">33.757</td>
    <td class="tg-yw4l">13.453</td>
  </tr>
</table>




