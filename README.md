# Mô tả chức năng các Function

- **split_data**: Chia dữ liệu thành tập huấn luyện và kiểm tra.
- **get_model_name**: Lấy tên của mô hình từ tên lớp của nó.
- **sigmoid**: Tính toán hàm kích hoạt sigmoid.
- **softmax**: Tính toán hàm softmax để xác suất hóa đa lớp.
- **mse_loss**: Tính toán hàm lỗi bình phương trung bình (MSE).
- **cross_entropy_loss**: Tính toán hàm lỗi cross-entropy cho bài toán phân loại.
- **scaler_data**: Tải và chuẩn bị dữ liệu để đánh giá bộ chuẩn hóa.
- **scaler_evaluate**: So sánh hiệu suất và kết quả chuẩn hóa giữa bộ chuẩn hóa tự xây dựng và sklearn trên một tập dữ liệu.
- **scaler_evaluates**: Đánh giá và so sánh các bộ chuẩn hóa trên nhiều tập dữ liệu.
- **classification_data**: Chuẩn bị và chuẩn hóa dữ liệu phân loại để đánh giá mô hình.
- **classification_evaluate**: So sánh thời gian huấn luyện và các chỉ số hiệu suất giữa mô hình phân loại tự xây dựng và sklearn trên một tập dữ liệu.
- **classification_evaluates**: Đánh giá các mô hình phân loại trên nhiều tập dữ liệu phân loại.
- **classification_predictions**: Tạo dự đoán trên dữ liệu mới bằng mô hình phân loại tự xây dựng.
- **regression_data**: Chuẩn bị và chuẩn hóa dữ liệu hồi quy để đánh giá mô hình.
- **regression_evaluate**: So sánh thời gian huấn luyện và các chỉ số hiệu suất giữa mô hình hồi quy tự xây dựng và sklearn trên một tập dữ liệu.
- **regression_evaluates**: Đánh giá các mô hình hồi quy trên nhiều tập dữ liệu hồi quy.

# Mô tả các cột trong Dataset Regression

- **DayOfWeek**: Số thứ tự ngày trong tuần (1 = Thứ Hai, ..., 7 = Chủ Nhật).
- **DepTime**: Giờ khởi hành thực tế, được mã hóa dưới dạng số (HHMM).
- **ArrTime**: Giờ hạ cánh thực tế, được mã hóa dưới dạng số (HHMM).
- **CRSArrTime**: Giờ hạ cánh dự kiến, mã hóa dưới dạng số (HHMM).
- **FlightNum**: Số hiệu chuyến bay, được lưu dưới dạng số nguyên.
- **ActualElapsedTime**: Thời gian thực tế của chuyến bay (phút).
- **CRSElapsedTime**: Thời gian bay dự kiến theo lịch trình (phút).
- **AirTime**: Thời gian bay thực tế khi ở trên không (phút).
- **DepDelay**: Số phút trễ khởi hành (âm nếu sớm hơn lịch trình).
- **Distance**: Khoảng cách giữa hai sân bay (dặm).
- **TaxiIn**: Thời gian di chuyển trên đường băng sau khi hạ cánh (phút).
- **TaxiOut**: Thời gian di chuyển trên đường băng trước khi cất cánh (phút).
- **Cancelled**: Trạng thái chuyến bay bị hủy (0 = Không, 1 = Có).
- **Diverted**: Trạng thái chuyến bay chuyển hướng (0 = Không, 1 = Có).
- **CarrierDelay**: Độ trễ do hãng hàng không (phút).
- **WeatherDelay**: Độ trễ do thời tiết (phút).
- **NASDelay**: Độ trễ do Hệ thống Hàng không Quốc gia (phút).
- **SecurityDelay**: Độ trễ do kiểm tra an ninh (phút).
- **LateAircraftDelay**: Độ trễ do máy bay đến muộn (phút).
- **ArrDelay**: Số phút trễ khi hạ cánh (âm nếu sớm hơn lịch trình).

# Mô tả các cột trong Dataset Classification

- **HighBP**: Huyết áp cao (0: Không, 1: Có).
- **HighChol**: Cholesterol cao (0: Không, 1: Có).
- **CholCheck**: Đã kiểm tra cholesterol trong 5 năm qua (0: Không, 1: Có).
- **BMI**: Chỉ số khối cơ thể (Body Mass Index).
- **Smoker**: Tình trạng hút thuốc lá (0: Không, 1: Có).
- **Stroke**: Từng bị đột quỵ (0: Không, 1: Có).
- **HeartDiseaseorAttack**: Từng bị bệnh tim hoặc đau tim (0: Không, 1: Có).
- **PhysActivity**: Tham gia hoạt động thể chất thường xuyên (0: Không, 1: Có).
- **Fruits**: Ăn trái cây ít nhất 1 lần/ngày (0: Không, 1: Có).
- **Veggies**: Ăn rau ít nhất 1 lần/ngày (0: Không, 1: Có).
- **AnyHealthcare**: Đã nhận dịch vụ chăm sóc sức khỏe trong năm qua (0: Không, 1: Có).
- **GenHlth**: Đánh giá sức khỏe tổng quát (1: Tốt nhất, 5: Tệ nhất).
- **MentHlth**: Số ngày có vấn đề về sức khỏe tinh thần trong tháng qua.
- **PhysHlth**: Số ngày có vấn đề về sức khỏe thể chất trong tháng qua.
- **DiffWalk**: Khó khăn khi đi bộ hoặc vận động (0: Không, 1: Có).
- **Sex**: Giới tính (0: Nữ, 1: Nam).
- **Age**: Nhóm tuổi (1: 18-24, 2: 25-29, ..., 13: 80 trở lên).
- **Education**: Trình độ học vấn (1: Chưa tốt nghiệp trung học, ..., 6: Sau đại học).
- **Income**: Nhóm thu nhập (1: Dưới $10k, ..., 8: Trên $75k).
- **HasDiabetes**: Đang bị tiểu đường (0: Không, 1: Có).

