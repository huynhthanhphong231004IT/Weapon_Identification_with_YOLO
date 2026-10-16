<h2 align="center">
  Author: Huynh Thanh Phong (ReoRioll)
</h2>

<p align="center">
   Computer Science of College of Information and Communication Technology of Can Tho University (Course 48)<br>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Researchs:</b> Artificial Intelligence in Education - Mathematics in Deep Learning and Machine Learning<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Name Project:</b></b> </mark> YOLO-WeaponID: Hệ thống giám sát và nhận diện mối đe doạ an ninh thời gian thực<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Link Data:</b></b> </mark> https://www.kaggle.com/datasets/reorioll/data-weapon<br>

</p>
<p align="center">
   <b>Presional link Information</b>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Facbook: https://www.facebook.com/huynh.thanh.phong.561667 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Kaggle: https://www.kaggle.com/reorioll <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Youtobe: https://www.youtube.com/@ReoRioll-2304CICTCTU <br>
</p>
<br>

```python
!git clone https://github.com/huynhthanhphong231004IT/Weapon_Identification_with_YOLO.git
```


<div align="center">
<p align="center">
  <img src="Charts/Flow_2.jpg" width="800">
  <br><br>
  <i>Kiến trúc phân loại vũ khí hai nhánh: Kết hợp mô hình phát hiện YOLO, trích xuất đặc trưng hình thái học PCA và dung hợp mượt hóa theo thời gian</i>
</p>
</div>

<div align="center">
  <table>
    <tr>
      <td width="50%"><img src="Images/KQ_01.jpg"></td>
      <td width="50%"><img src="Images/KQ_02.jpg"></td>
    </tr>
  </table>
  <p><i>Kết quả nhận diện vũ khí và phân tích chỉ số rủi ro trên giao diện hệ thống giám sát an ninh thông minh.</i></p>
</div>

### Cơ sở lý thuyết thuật toán, cơ sở toán học và phương pháp xác minh đặc trưng hình học - màu sắc kết hợp mô hình học sâu.
---
Hệ thống xác minh và theo dõi mối đe dọa được đề xuất dựa trên kiến trúc kết hợp (hybrid framework), dung hợp mô hình phát hiện học sâu (Deep Learning) với hệ thống xác minh logic mờ và hình học định tính. Cơ sở toán học của các thuật toán cốt lõi được mô tả chi tiết.

### 1. Độ trùng lặp khung bao và ghép nối mục tiêu

Để theo dõi các mục tiêu phát hiện qua các khung hình liên tiếp và thực hiện phân nhóm (clustering), chỉ số Giao trên Hợp (IoU) giữa hai khung bao A và B được tính toán theo công thức 

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{IoU}(A,B)=\frac{\text{Area}(A\cap&space;B)}{\text{Area}(A\cup&space;B)}=\frac{\max(0,x_{i2}-x_{i1})\times\max(0,y_{i2}-y_{i1})}{\text{Area}(A)&plus;\text{Area}(B)-\text{Area}(A\cap&space;B)}" title="IoU Formula" />
</p>

<p>
  <img src="https://latex.codecogs.com/svg.image?\text{Trong&space;đó&space;}(x_{i1},y_{i1})\text{&space;và&space;}(x_{i2},y_{i2})\text{&space;lần&space;lượt&space;là&space;tọa&space;độ&space;của&space;hình&space;chữ&space;nhật&space;giao&space;nhau.}" title="Explanation" />
</p>

### 2. Hàm thuộc tính Logic mờ dạng hình thang/tam giác

Để đánh giá các đặc trưng hình học biến thiên liên tục mà không gặp phải hiện tượng chia ngưỡng cứng (hard thresholding), hàm thuộc tính dạng tam giác/hình thang f(x; a, b, c, d) được định nghĩa
<p align="center">
  <img src="https://latex.codecogs.com/svg.image?f(x;a,b,c,d)=\begin{cases}0,&\text{x}\le&space;a\text{&space;hoặc&space;x}\ge&space;d\\\frac{x-a}{b-a&plus;\epsilon},&a<x<b\\1.0,&b\le&space;x\le&space;c\\\frac{d-x}{d-c&plus;\epsilon},&c<x<d\end{cases}" title="Membership Function" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{Trong&space;đó&space;}\epsilon=10^{-9}\text{&space;là&space;hằng&space;số&space;ổn&space;định&space;tránh&space;lỗi&space;chia&space;cho&space;0.}" title="Epsilon Explanation" />
</p>

### 3. Trích xuất trục chính bằng phân rã ma trận hiệp biến (PCA)

<p>
Để đo lường độ đối xứng không gian và phân bố chiều rộng của đối tượng, các đường biên tiền cảnh thu được sau phân đoạn được chuyển về hệ tọa độ chuẩn hóa thông qua Phân tích Thành phần Chính (PCA). Cho tập hợp <img src="https://latex.codecogs.com/svg.image?N" title="N" /> điểm ảnh tiền cảnh đã trừ đi trọng tâm <img src="https://latex.codecogs.com/svg.image?\mathbf{P}=\{(x_i,y_i)\}_{i=1}^N" title="P" />
</p>


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{\Sigma}=\frac{1}{N}\mathbf{P}^T\mathbf{P}&plus;\epsilon\mathbf{I}_2" title="Covariance Matrix" />
</p>

<p>
  Trục hướng chính của vật thể được xác định bằng cách giải phương trình đặc trưng tìm giá trị riêng <img src="https://latex.codecogs.com/svg.image?\lambda" title="lambda" /> và vecto riêng <img src="https://latex.codecogs.com/svg.image?\mathbf{v}" title="v" />:
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{\Sigma}\mathbf{v}=\lambda\mathbf{v}" title="Eigenvalue Equation" />
</p>
<p>
  Các tọa độ không gian khi chiếu lên vecto riêng chính <img src="https://latex.codecogs.com/svg.image?\mathbf{v}_{\text{major}}" title="v_major" /> thể hiện chiều dài dọc của đối tượng. Từ đó, thuật toán chia trục thành <img src="https://latex.codecogs.com/svg.image?B" title="B" /> khoảng (bins) để lấy mẫu chiều rộng <img src="https://latex.codecogs.com/svg.image?w(b)" title="w(b)" />, phục vụ việc đánh giá độ bất đối xứng <img src="https://latex.codecogs.com/svg.image?(1-S)" title="1-S" /> và độ nhọn của vật thể:
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?S=\frac{\sum(G\cap&space;G_{\text{flip}})}{\max(1,\sum(G\cup&space;G_{\text{flip}}))}" title="Symmetry Score" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{Trong&space;đó&space;}G\text{&space;đại&space;diện&space;cho&space;lưới&space;nhị&space;phân&space;hóa&space;(rasterized&space;grid)&space;của&space;hình&space;dạng&space;đối&space;tượng&space;đã&space;căn&space;chỉnh&space;theo&space;trục&space;chính.}" title="Grid Explanation" />
</p>

### 4. Kết hợp quyết định thích ứng và làm mượt theo thời gian
<p>
  Thay vì chỉ dựa vào xác suất đầu ra (logits) của mạng nơ-ron sâu, vecto phân loại cuối cùng <img src="https://latex.codecogs.com/svg.image?\mathbf{P}_{\text{final}}" title="P_final" /> thực hiện kết hợp vecto xác suất <img src="https://latex.codecogs.com/svg.image?\mathbf{Y}" title="Y" /> từ mô hình YOLO với vecto điểm hình học <img src="https://latex.codecogs.com/svg.image?\mathbf{G}" title="G" /> theo trọng số chất lượng phân đoạn <img src="https://latex.codecogs.com/svg.image?Q_{\text{seg}}" title="Q_seg" />:
</p>


<p align="center">
  <img src="https://latex.codecogs.com/svg.image?w_{\text{geo}}=\left(0.55&plus;0.30\cdot\frac{\max(0,0.65-c_{\text{det}})}{0.65}\right)\cdot&space;Q_{\text{seg}}" title="w_geo Formula" />
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\mathbf{P}_{\text{fused}}=(1-w_{\text{geo}})\mathbf{Y}&plus;w_{\text{geo}}\mathbf{G}" title="P_fused Formula" />
</p>

<p>
  Trong đó <img src="https://latex.codecogs.com/svg.image?c_{\text{det}}" title="c_det" /> là độ tin cậy của mô hình phát hiện và <img src="https://latex.codecogs.com/svg.image?Q_{\text{seg}}\in[0,1]" title="Q_seg in [0,1]" /> là điểm chất lượng phân đoạn hình ảnh.
</p>

<p>
  Để loại bỏ hiện tượng nhấp nháy (flickering) và dao động dự đoán giữa các khung hình, trung bình động lũy thừa (EMA) với hệ số làm mượt <img src="https://latex.codecogs.com/svg.image?\alpha=0.06" title="alpha=0.06" /> được duy trì trên một cửa sổ thời gian cố định <img src="https://latex.codecogs.com/svg.image?W_{\text{smooth}}" title="W_smooth" />:
</p>

<p align="center">
  <img src="https://latex.codecogs.com/svg.image?\text{Trong&space;đó&space;}c_{\text{det}}\text{&space;là&space;độ&space;tin&space;cậy&space;của&space;mô&space;hình&space;phát&space;hiện&space;và&space;}Q_{\text{seg}}\in[0,1]\text{&space;là&space;điểm&space;chất&space;lượng&space;phân&space;đoạn&space;hình&space;ảnh.}" title="Variables Explanation" />
</p>


## Kết quả huấn luyện trên mô hình Yolo11n trên tập dữ liệu data-weapon
<div align="center">
<p align="center">
  <img src="Charts/group_figure_1.png" width="800">
  <hr width="100%">
  <img src="Charts/group_figure_2.png" width="800">
  <hr width="100%">
  <img src="Charts/group_figure_3.png" width="800">
  <hr width="100%">
  <img src="Charts/group_figure_4.png" width="800">
  <hr width="100%">
  <img src="Charts/group_figure_5.png" width="800">
  <br><br>
  <i>Khung phân tích đa chiều về động lực học quá trình huấn luyện, phân phối thống kê và tương quan chỉ số đánh giá</i>
</p>
</div>







