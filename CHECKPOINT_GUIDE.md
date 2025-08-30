# Hướng dẫn sử dụng Checkpoint trong iML AutoML Pipeline

## Tổng quan

Tính năng checkpoint cho phép bạn:
1. **Dừng pipeline** tại một điểm cụ thể (ví dụ: sau khi tạo guideline)
2. **Chỉnh sửa thủ công** các kết quả trung gian 
3. **Tiếp tục pipeline** từ điểm đã dừng

## Cách sử dụng

### 1. Chạy pipeline đến bước tạo guideline (Khuyên dùng)

```bash
# Sử dụng script helper (dễ dàng nhất)
python run_checkpoint.py -i ./path/to/your/data --stop-at-guideline

# Hoặc sử dụng script gốc
python run.py -i ./path/to/your/data --checkpoint-mode partial --checkpoint-action guideline
```

### 2. Chỉnh sửa guideline thủ công

Sau khi pipeline dừng, bạn sẽ thấy thông báo:
```
Pipeline stopped after guideline generation.
You can now manually edit the guideline in the states folder.
```

**Tìm file guideline:**
- Vào thư mục output (ví dụ: `runs/run_20240101_120000_abcd1234/`)
- Mở file: `states/guideline_agent_parsed_response.json`
- Chỉnh sửa nội dung guideline theo ý muốn
- Lưu file

### 3. Tiếp tục pipeline từ bước preprocessing

```bash
# Sử dụng script helper
python run_checkpoint.py -i ./path/to/your/data --resume-from-preprocessing -o ./runs/run_20240101_120000_abcd1234

# Hoặc sử dụng script gốc  
python run.py -i ./path/to/your/data --checkpoint-mode resume --checkpoint-action preprocessing -o ./runs/run_20240101_120000_abcd1234
```

## Các điểm dừng khả dụng

### Chế độ Partial (dừng tại):
- `description`: Sau phân tích mô tả bài toán
- `profiling`: Sau phân tích dữ liệu  
- `guideline`: Sau tạo guideline (thường dùng nhất)

### Chế độ Resume (tiếp tục từ):
- `preprocessing`: Từ bước tạo code preprocessing
- `modeling`: Từ bước tạo code modeling
- `assembler`: Từ bước assemble và execute

## Ví dụ workflow hoàn chỉnh

```bash
# Bước 1: Chạy đến guideline
python run_checkpoint.py -i ./datasets/my_data --stop-at-guideline
# Output: runs/run_20240101_120000_abcd1234/

# Bước 2: Chỉnh sửa guideline
# Mở và sửa file: runs/run_20240101_120000_abcd1234/states/guideline_agent_parsed_response.json

# Bước 3: Tiếp tục pipeline
python run_checkpoint.py -i ./datasets/my_data --resume-from-preprocessing -o ./runs/run_20240101_120000_abcd1234
```

## Các tùy chọn khác

### Dừng tại các bước khác:
```bash
# Dừng sau phân tích dữ liệu
python run_checkpoint.py -i ./data --stop-at-profiling

# Dừng sau phân tích mô tả
python run_checkpoint.py -i ./data --stop-at-description
```

### Tiếp tục từ các bước khác:
```bash
# Tiếp tục từ modeling (bỏ qua preprocessing)
python run_checkpoint.py -i ./data --resume-from-modeling -o ./existing_run

# Tiếp tục từ assembler (chỉ assemble và execute)
python run_checkpoint.py -i ./data --resume-from-assembler -o ./existing_run
```

### Chạy pipeline hoàn chỉnh (như trước):
```bash
python run_checkpoint.py -i ./data --full
# Hoặc đơn giản:
python run.py -i ./data
```

## Files quan trọng trong thư mục states/

- `description_analyzer_parsed_response.json`: Kết quả phân tích mô tả bài toán
- `profiling_result.json`: Kết quả phân tích dữ liệu chi tiết
- `profiling_summarizer_parsed_response.json`: Tóm tắt phân tích dữ liệu  
- `model_retrieval.json`: Đề xuất models từ HuggingFace
- `guideline_agent_parsed_response.json`: **File guideline có thể chỉnh sửa**

## Lưu ý quan trọng

1. **Đường dẫn output**: Khi resume, bạn PHẢI chỉ định đường dẫn chính xác đến thư mục run đã tồn tại
2. **Backup**: Nên backup file guideline gốc trước khi chỉnh sửa
3. **Format JSON**: Đảm bảo file JSON vẫn đúng format sau khi chỉnh sửa
4. **Dependencies**: Đảm bảo tất cả dependencies đã được cài đặt (`pip install -r requirements.txt`)

## Troubleshooting

### Lỗi "Cannot resume: guideline not found"
- Kiểm tra đường dẫn output directory có đúng không
- Đảm bảo file `guideline_agent_parsed_response.json` tồn tại trong `states/`

### Lỗi JSON parsing  
- Kiểm tra syntax JSON sau khi chỉnh sửa guideline
- Sử dụng JSON validator online nếu cần

### Pipeline không tìm thấy checkpoint data
- Đảm bảo bạn đang chỉ định đúng thư mục output từ lần chạy trước
- Kiểm tra các file states/ có đầy đủ không
