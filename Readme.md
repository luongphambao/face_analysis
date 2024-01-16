

# Face Analysis Challenge
## 1. Setup môi trường:
Đầu tiên ta cần cài đặt torch với cuda11.7
```
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117  -f https://download.pytorch.org/whl/torch_stable.html
```


Install các thư viện cần thiết khác
```
pip install -r requirements.txt
```

## 2. Training
Chuẩn bị sẵn folder crop chứa các ảnh khuôn mặt đã crop và 3 file labels_train.csv.labels_val.csv, labels_add.csv(có thể ko cần file này) trong folder data
```
```
Để tải mô hình đã được huấn luyện ta chỉ cần chạy lệnh sau
```
python3 train.py
```
## 3. Chạy Inference



```
python3 predict.py --json_path <path_to_json>  --data_folder_path <path_to_data_folder> --data_img2id_path <path_to_data_img2id>
--csv_path <path_to_csv> 
```
Sau đó ta sẽ tạo file để nộp từ folder submision
```
zip -r predicted.zip answer.csv
```
sẽ lưu thành file predicted.zip



