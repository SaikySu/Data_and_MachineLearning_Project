# Bước 1 – Khảo sát dataset cho bài toán theo dõi chuyển động & phân tích hành vi khách hàng trên Edge

> Ngày khảo sát: 24/09/2026. Các link được kiểm tra qua tìm kiếm web; một số host (MOTChallenge, MERL, Kaggle, Roboflow, HuggingFace) **không truy cập được từ môi trường cloud** này nên việc tải phải thực hiện trên máy local (xem `scripts/download_datasets.py`).

## 1. Yêu cầu bài toán → yêu cầu dữ liệu

Hệ thống mục tiêu chạy **trên thiết bị Edge** (Raspberry Pi 5 + AI HAT / Jetson Orin Nano / RK3588) đặt tại cửa hàng, gồm 4 tầng xử lý:

| Tầng | Chức năng | Loại dữ liệu cần | Nhãn cần |
|---|---|---|---|
| T1. Phát hiện người (Detection) | Tìm khách hàng trong từng frame | Ảnh/video CCTV góc cao, đông người, che khuất | Bounding box `person` |
| T2. Theo dõi đa đối tượng (Tracking) | Gán ID ổn định qua thời gian, sinh quỹ đạo | Video liên tục có ID xuyên frame | Box + track ID theo frame (MOT format) |
| T3. Phân tích hành vi mức cửa hàng (Analytics) | Đếm ra/vào, heatmap, dwell time, luồng di chuyển giữa các khu | Quỹ đạo (x, y, t, id) trong không gian cửa hàng | Toạ độ mặt sàn hoặc pixel + vùng (zone) |
| T4. Nhận diện hành động/bất thường (Action) | Với tay lên kệ, xem sản phẩm, hành vi đáng ngờ | Clip camera trên cao trong cửa hàng | Nhãn hành động theo đoạn thời gian, keypoint |

Tiêu chí chọn dataset: (1) góc camera giống CCTV cửa hàng; (2) có nhãn track ID hoặc quỹ đạo; (3) giấy phép cho phép nghiên cứu; (4) kích thước hợp lý để train model nhỏ (YOLO-n/s) chạy được trên Edge.

## 2. Bảng dataset đề xuất

### 2.1 Nhóm A – Detection & Tracking người (bắt buộc)

| Dataset | Mô tả | Quy mô | Định dạng nhãn | Giấy phép | Vai trò trong dự án |
|---|---|---|---|---|---|
| **MOT17** ([motchallenge.net](https://motchallenge.net/data/MOT17/)) | 14 sequence người đi bộ, camera tĩnh + di chuyển, trong nhà & ngoài trời (có cảnh trung tâm thương mại) | 7 train (5 136 frame) + 7 test (5 919 frame) | MOT: `frame,id,x,y,w,h,conf,class,vis` | CC BY-NC-SA 3.0 (nghiên cứu) | **Dataset chính** để huấn luyện/đánh giá tracker (HOTA, MOTA, IDF1) |
| **MOT20** ([motchallenge.net](https://motchallenge.net/)) | 8 video cảnh rất đông (tới 246 người/frame), camera cao | 4 train + 4 test, 13 410 frame | MOT | CC BY-NC-SA 3.0 | Kiểm tra độ bền trong giờ cao điểm |
| **CrowdHuman** ([arXiv](https://arxiv.org/abs/1805.00123), [HF mirror](https://huggingface.co/datasets/sshao0516/CrowdHuman)) | Ảnh người trong đám đông, nhãn full-body / visible / head | 15 000 train, 4 370 val, ~470K instance, ~23 người/ảnh | ODGT (JSON lines) | Nghiên cứu phi thương mại | **Fine-tune detector** YOLO-n/s cho cảnh che khuất |
| **WiderPerson** ([trang chủ](http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/), [Kaggle mirror](https://www.kaggle.com/datasets/petermushemi/widerperson-dataset-for-pedestrian-detection)) | Người đi bộ đa cảnh, không giới hạn giao thông | 13 382 ảnh, ~400K nhãn | txt (x1,y1,x2,y2,class) | Nghiên cứu | Bổ sung đa dạng cảnh cho detector |
| **Roboflow – Retail-Surveillance** ([link](https://universe.roboflow.com/project-ebze1/retail-surveillance)) / **CCTV People** ([link](https://universe.roboflow.com/project-wk4fq/cctv-people)) | Ảnh CCTV cửa hàng, xuất sẵn YOLO format | 5 513 / 700 ảnh | YOLO txt | Theo từng project (đa số CC BY 4.0) | Domain-adapt detector sang góc CCTV cửa hàng |

### 2.2 Nhóm B – Hành vi khách hàng trong cửa hàng (đặc thù bán lẻ)

| Dataset | Mô tả | Quy mô | Định dạng nhãn | Giấy phép | Vai trò |
|---|---|---|---|---|---|
| **MERL Shopping** ([MERL](https://www.merl.com/research/downloads/MERL_Shopping_Dataset), [FiftyOne card](https://docs.voxel51.com/dataset_zoo/datasets_hf/merl_shopping_dataset.html)) | Camera cố định nhìn từ trên xuống, người mua sắm trước kệ | 106 video × ~2 phút, 5 hành động: *Reach To Shelf, Retract From Shelf, Hand In Shelf, Inspect Product, Inspect Shelf* | MAT/CSV: khoảng thời gian (start, end, label) | Research-only (đồng ý điều khoản MERL) | **Dataset chính cho T4** – nhận diện tương tác với kệ |
| **CAVIAR – Shopping Centre (Lisbon)** ([Edinburgh](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)) | 26 sequence trong trung tâm thương mại, 2 góc nhìn (hành lang + trước cửa hàng): đi một mình, gặp nhau, *window shopping*, vào/ra cửa hàng | 384×288, 25 fps, ~90K frame | XML: box, hướng đi, nhãn hoạt động ngắn/dài hạn | Miễn phí cho nghiên cứu | Tracking + nhãn hành vi mức cảnh (dừng lại, xem hàng, vào cửa hàng) |
| **RetailS** ([GitHub](https://github.com/TeCSAR-UNCC/RetailS)) | Cửa hàng bán lẻ thật ở Mỹ, 6 camera, 10 ngày, có trộm cắp thật và dàn dựng | ~20M frame bình thường, test 2 432 frame thật + 20 578 frame dàn dựng | JSON keypoint (person ID, frame ID, XYC) + nhãn 0/1 | Chưa ghi rõ (liên hệ tác giả) | Phát hiện bất thường dựa trên pose, bảo mật riêng tư |
| **Kaggle – CCTV Shoplifting (YOLO & VLM)** ([link](https://www.kaggle.com/datasets/simuletic/cctv-shoplifting-detection-dataset-yolo-and-vlm)) | Dữ liệu tổng hợp góc CCTV | Vừa | YOLO + mô tả VLM | Theo Kaggle | Tuỳ chọn, tăng cường cho lớp bất thường |

### 2.3 Nhóm C – Quỹ đạo & luồng di chuyển (cho analytics và mô phỏng)

| Dataset | Mô tả | Quy mô | Định dạng | Giấy phép | Vai trò |
|---|---|---|---|---|---|
| **ATC Shopping Center** ([ATR](https://dil.atr.jp/crest2010_HRI/ATC_dataset/)) | Quỹ đạo người trong TTTM Osaka từ cảm biến 3D, ~900 m², 92 ngày | CSV mỗi ngày: `time, person_id, x, y, z, velocity, angle, facing` | CSV | Research-only | Xây/kiểm định thuật toán heatmap, dwell time, luồng khu vực **không cần video** |
| **Grand Central Station** | 17 000 quỹ đạo trong nhà, camera trên cao, 720×480, 25 fps, 33 phút | Quỹ đạo pixel | txt | Nghiên cứu | Kiểm tra phân cụm luồng đi |
| **EPFL Multi-camera (Laboratory/Terrace/Campus)** ([CVLab](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/)) | Nhiều camera đồng bộ, có calibration | 4 camera, 2.5 phút/seq | Ground truth + calibration | Nghiên cứu (CVLab) | Mở rộng đa camera / chiếu xuống mặt sàn (homography) |

## 3. Đánh giá & lựa chọn cuối cùng

| Ưu tiên | Dataset | Lý do |
|---|---|---|
| ★★★ | **MOT17** | Chuẩn benchmark tracking, có sẵn 3 bộ detection, nhãn visibility giúp lọc che khuất; cảnh MOT17-09/11 là trung tâm thương mại. |
| ★★★ | **CrowdHuman** | Detector nhỏ (YOLO-n) cần dữ liệu che khuất dày đặc để không mất người ở góc CCTV. |
| ★★★ | **MERL Shopping** | Dataset duy nhất có nhãn hành động tương tác kệ hàng, góc trên cao giống camera cửa hàng. |
| ★★ | **CAVIAR Shopping Centre** | Nhỏ, nhẹ, có sẵn nhãn hành vi (window shopping, vào/ra) – phù hợp demo Edge nhanh. |
| ★★ | **ATC** | Cho phép phát triển và kiểm thử module analytics (heatmap/dwell/flow) độc lập với model thị giác. |
| ★ | MOT20, WiderPerson, Roboflow retail, RetailS | Tăng cường/ mở rộng ở giai đoạn sau. |

## 4. Lưu ý về giấy phép & quyền riêng tư

* Toàn bộ dataset nhóm A/B/C đều **chỉ dùng cho nghiên cứu, phi thương mại**. Triển khai thương mại phải thu thập dữ liệu riêng và tuân thủ luật bảo vệ dữ liệu cá nhân (tại Việt Nam: Nghị định 13/2023/NĐ-CP).
* Thiết kế hệ thống theo hướng **không lưu video, không nhận diện danh tính**: chỉ giữ box/keypoint/quỹ đạo ẩn danh trên Edge, đúng tinh thần RetailS.
* MERL yêu cầu chấp nhận điều khoản trước khi tải; RetailS cần liên hệ tác giả để xác nhận giấy phép.

## 5. Nguồn tham khảo

* MOTChallenge: <https://motchallenge.net/>
* CrowdHuman: <https://arxiv.org/abs/1805.00123>
* WiderPerson: <https://arxiv.org/abs/1909.12118>
* MERL Shopping: <https://www.merl.com/research/highlights/merl-shopping-dataset>
* CAVIAR: <https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/>
* RetailS: <https://github.com/TeCSAR-UNCC/RetailS>
* ATC dataset: <https://dil.atr.jp/crest2010_HRI/ATC_dataset/>
* EPFL multi-camera: <https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/>
* Roboflow Retail-Surveillance: <https://universe.roboflow.com/project-ebze1/retail-surveillance>
* Tổng quan hệ thống tương tự: "Retail store customer behavior analysis system" <https://arxiv.org/pdf/2309.03232>; "Advanced Customer Behavior Tracking and Heatmap Analysis with YOLOv5 and DeepSORT" <https://www.mdpi.com/2079-9292/13/23/4730>; "Real-time Embedded Person Detection and Tracking for Shopping Behaviour Analysis" <https://arxiv.org/pdf/2007.04942>
