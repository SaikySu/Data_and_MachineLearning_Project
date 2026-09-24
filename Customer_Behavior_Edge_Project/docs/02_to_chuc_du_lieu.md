# Bước 2 – Chia dữ liệu thành các mục phù hợp

## 1. Nguyên tắc chia

1. **Chia theo bài toán (task)** trước, theo tập (split) sau. Mỗi tầng của hệ thống có một thư mục `data/processed/<task>/` với định dạng thống nhất, để module nào cũng có thể được huấn luyện/kiểm thử độc lập.
2. **Chia theo sequence, không chia theo frame.** Các frame trong một video gần như giống nhau, chia ngẫu nhiên theo frame sẽ gây rò rỉ (data leakage) và làm điểm val ảo cao.
3. **Giữ nguyên tập test của benchmark** (MOT17/MOT20 test không có nhãn public) để so sánh với bài báo; tập val được cắt từ tập train của benchmark.
4. **Tách theo "miền" khi có thể**: val/test ưu tiên các sequence có góc camera CCTV trong nhà (gần nhất với cửa hàng), train lấy phần còn lại.
5. Mọi phép chia được ghi trong `configs/split.yaml` và tái tạo được bằng `scripts/prepare_data.py` với `seed` cố định.

## 2. Năm mục dữ liệu

### 2.1 `detection/` – phát hiện người (YOLO format)

| Nguồn | Train | Val | Test | Ghi chú |
|---|---|---|---|---|
| CrowdHuman | `train` (15 000 ảnh) | `val` (4 370 ảnh) | – | Dùng box **full-body** (`fbox`), lọc `ignore`; class duy nhất `0 = person` |
| MOT17 (frame) | Các sequence train trừ val | MOT17-09, MOT17-11 (cảnh trong nhà/TTTM) | – | Lấy mẫu 1 frame / 5 frame để giảm trùng lặp |
| Roboflow retail (tuỳ chọn) | 80 % | 20 % | – | Domain cửa hàng |
| WiderPerson (tuỳ chọn) | `train` | `val` | – | |

Định dạng: `images/{split}/*.jpg`, `labels/{split}/*.txt` với dòng `0 cx cy w h` (chuẩn hoá 0–1). File `detection.yaml` dùng trực tiếp cho Ultralytics.

### 2.2 `tracking/` – theo dõi đa đối tượng (MOT format)

| Nguồn | Train | Val | Test |
|---|---|---|---|
| MOT17 | 02, 04, 05, 10, 13 | 09, 11 | 7 sequence test chính thức (submit server) |
| MOT20 | 01, 02 | 03, 05 | test chính thức |
| CAVIAR Shopping Centre | 18 sequence | 4 | 4 |

Giữ nguyên MOT format (`img1/`, `gt/gt.txt`, `seqinfo.ini`) vì TrackEval/ByteTrack/BoT-SORT đọc trực tiếp. Chỉ lọc `class == 1` (pedestrian) và `visibility >= 0.25` khi tạo nhãn train cho detector.

### 2.3 `action/` – hành động trước kệ (MERL Shopping)

* Chia **theo người** (subject) đúng như protocol gốc của MERL: train 60 video / val 9 / test 37, để cùng một người không xuất hiện ở cả train lẫn test.
* Mỗi mẫu là một clip 16–32 frame quanh một khoảng thời gian có nhãn, lưu `action/clips/{split}/<video>_<start>_<end>.mp4` + `action/{split}.csv` (`clip, label, video, start, end`).
* 5 lớp gộp về 3 nhóm cho Edge nếu cần đơn giản: `interact_shelf` (Reach/Retract/Hand-in), `inspect_product`, `inspect_shelf`.

### 2.4 `trajectory/` – quỹ đạo cho analytics

* ATC: mỗi ngày là một file CSV → chia **theo ngày**: 70 % ngày train / 15 % val / 15 % test. Cột chuẩn hoá: `t, track_id, x, y` (mét) + `zone_id` sau khi áp bản đồ vùng.
* Quỹ đạo sinh từ CAVIAR/MOT17 ground truth (tâm đáy box → homography nếu có, không thì dùng pixel): cùng schema để module analytics chạy được trên cả hai.
* Dùng để **kiểm định các chỉ số nghiệp vụ** (đếm vào/ra, dwell time, heatmap, ma trận chuyển vùng) bằng nhãn ground truth thay vì output tracker.

### 2.5 `anomaly/` – hành vi bất thường (RetailS, tuỳ chọn)

* Chuỗi keypoint theo `person_id` cắt thành cửa sổ 30 frame, nhãn 0/1. Chia theo **ngày quay** để tránh rò rỉ.
* Chỉ dùng khi dự án mở rộng sang phát hiện trộm cắp; mặc định tắt trong `configs/datasets.yaml`.

## 3. Sơ đồ luồng dữ liệu

```
raw/MOT17 ──┐
raw/MOT20 ──┼─► prepare_data.py --task tracking  ─► processed/tracking/{train,val,test}
raw/CAVIAR ─┘                 │
                              └─ --task detection ─► processed/detection (YOLO) ◄── raw/CrowdHuman
raw/MERL ─────► --task action     ─► processed/action
raw/ATC  ─────► --task trajectory ─► processed/trajectory
```

## 4. Kiểm tra chất lượng sau khi chia

`prepare_data.py` in ra bảng thống kê cho từng split: số sequence, số frame, số track, số box, mật độ người/frame, tỉ lệ box bị che (`vis < 0.5`). Kiểm tra bắt buộc:

* Không có sequence/subject/ngày nào xuất hiện ở hai split.
* Phân phối mật độ người/frame giữa train và val không lệch quá 30 %.
* Tất cả box nằm trong ảnh, `w, h > 0`.

## 5. Kích thước ước tính

| Mục | Dung lượng raw | Sau xử lý |
|---|---|---|
| MOT17 | ~5.5 GB | ~2 GB (chỉ giữ ảnh + gt) |
| MOT20 | ~5 GB | ~2 GB |
| CrowdHuman | ~3.5 GB | ~3.5 GB |
| MERL Shopping | ~10 GB | ~1 GB clip |
| CAVIAR | ~0.5 GB | ~0.5 GB |
| ATC (mẫu) | ~1 GB CSV | ~0.2 GB |
