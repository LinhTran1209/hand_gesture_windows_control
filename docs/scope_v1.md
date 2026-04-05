# SCOPE V1 - Đồ án nhận diện cử chỉ tay điều khiển máy tính

## 1. Tên đề tài
Nhận diện cử chỉ tay cho hệ thống điều khiển máy tính không chạm trên Windows.

## 2. Mục tiêu
Xây dựng một ứng dụng desktop trên Windows cho phép người dùng sử dụng webcam để nhận diện cử chỉ tay theo thời gian thực và điều khiển các thao tác cơ bản trên máy tính mà không cần chạm vào chuột hoặc bàn phím.

## 3. Phạm vi thực hiện
- Hệ điều hành: Windows
- Thiết bị đầu vào: Webcam RGB
- Số tay hỗ trợ: 1 tay
- Tài nguyên tính toán: CPU
- Hình thức nhận diện:
  - Cử chỉ tĩnh
  - Cử chỉ động
- Kết quả đầu ra:
  - Điều khiển chuột
  - Click trái / click phải
  - Cuộn lên / cuộn xuống
  - Chuyển lùi / chuyển tới
  - Tạm dừng / kích hoạt điều khiển

## 4. Phạm vi không thực hiện
- Không hỗ trợ Android, Linux, macOS ở bản đầu
- Không hỗ trợ 2 tay đồng thời
- Không dùng camera chiều sâu
- Không xử lý chuỗi cử chỉ phức tạp nhiều bước
- Không tối ưu cho game
- Không điều khiển toàn bộ hệ điều hành theo mức thương mại hoàn chỉnh

## 5. Bộ cử chỉ bản 1
### Cử chỉ tĩnh
1. open_palm
2. fist
3. point
4. pinch
5. two_fingers
6. thumbs_up

### Cử chỉ động
7. swipe_left
8. swipe_right

### Dữ liệu âm tính / không lệnh
9. no_gesture
10. transition_motion
11. non_command_motion

## 6. Ánh xạ cử chỉ sang thao tác
- point -> di chuột
- pinch -> click trái
- two_fingers -> click phải hoặc vào chế độ scroll
- open_palm -> kích hoạt / sẵn sàng nhận lệnh
- fist -> khóa / tạm dừng điều khiển
- thumbs_up -> xác nhận / enter / play-pause
- swipe_left -> back / previous
- swipe_right -> next / forward

## 7. Hướng kỹ thuật
Hệ thống gồm 3 khối chính:

### Khối A - Phát hiện bàn tay và trích xuất điểm mốc
- Nhận ảnh từ webcam
- Sử dụng MediaPipe để phát hiện bàn tay
- Trích xuất 21 landmarks bàn tay theo từng frame

### Khối B - Xây dựng mô hình và nhận diện cử chỉ tay
- Thu thập dữ liệu từ dataset công khai và dữ liệu tự quay
- Tiền xử lý dữ liệu landmark
- Huấn luyện mô hình nhận diện cử chỉ tĩnh
- Huấn luyện mô hình nhận diện cử chỉ động
- Đánh giá mô hình bằng accuracy, precision, recall, F1-score

### Khối C - Điều khiển máy tính và giao diện ứng dụng
- Nhận kết quả suy luận từ mô hình
- Làm mượt kết quả theo thời gian
- Ánh xạ sang thao tác chuột / bàn phím
- Hiển thị webcam, trạng thái cử chỉ, chế độ điều khiển

## 8. Hướng dữ liệu
### Dataset công khai
- HaGRID: dùng cho cử chỉ tĩnh
- IPN Hand: dùng cho cử chỉ động

### Dataset tự thu thập
- Quay bằng webcam trên máy Windows
- Có dữ liệu cử chỉ tĩnh, động và không lệnh
- Nhiều điều kiện ánh sáng, nền, khoảng cách, vị trí tay

## 9. Hướng mô hình
### Mô hình cử chỉ tĩnh
- Input: 21 landmarks của 1 frame
- Output: nhãn cử chỉ tĩnh

### Mô hình cử chỉ động
- Input: chuỗi landmarks của nhiều frame liên tiếp
- Output: nhãn cử chỉ động

## 10. Tiêu chí hoàn thành bản 1
- Mở được webcam và nhận diện landmarks theo thời gian thực
- Huấn luyện được mô hình tĩnh và động
- Ứng dụng nhận diện được tối thiểu 6-8 cử chỉ
- Điều khiển được các thao tác cơ bản trên Windows
- Chạy ổn định trên CPU
- Có giao diện desktop để demo

## 11. Tiêu chí đánh giá
### Đánh giá mô hình
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

### Đánh giá hệ thống
- FPS
- Độ trễ
- Tỷ lệ kích hoạt nhầm
- Độ ổn định khi demo thực tế

## 12. Kế hoạch tổng quát
- Giai đoạn 1: dựng pipeline webcam + landmarks
- Giai đoạn 2: thu thập và xử lý dữ liệu
- Giai đoạn 3: huấn luyện mô hình tĩnh
- Giai đoạn 4: huấn luyện mô hình động
- Giai đoạn 5: tích hợp vào app Windows
- Giai đoạn 6: kiểm thử, đóng gói, hoàn thiện báo cáo