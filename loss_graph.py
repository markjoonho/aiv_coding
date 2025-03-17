import re
import matplotlib.pyplot as plt

# 로그 파일 읽기
log_file = "./ckpt_final/20250318_050935/train.log"  # 로그 파일 경로
with open(log_file, "r", encoding="utf-8") as f:
    log_text = f.read()  # 전체 파일을 한 문자열로 읽기

# 정규 표현식으로 에포크 데이터 추출
pattern = re.compile(
    r"Epoch (\d+)/\d+ - Train Loss: ([\d.]+)\n"
    r".*?Train - ce: ([\d.]+), bbox: ([\d.]+), giou: ([\d.]+)\n"
    r".*?Validation Loss: ([\d.]+)\n"
    r".*?Validation - ce: ([\d.]+), bbox: ([\d.]+), giou: ([\d.]+)",
    re.MULTILINE
)


matches = pattern.findall(log_text)

# 데이터 저장
epochs = []
train_loss, val_loss = [], []
train_ce, val_ce = [], []
train_giou, val_giou = [], []
train_bbox, val_bbox = [], []

for match in matches:
    epochs.append(int(match[0]))
    train_loss.append(float(match[1]))
    train_ce.append(float(match[2]))
    train_bbox.append(float(match[3]))
    train_giou.append(float(match[4]))
    val_loss.append(float(match[5]))
    val_ce.append(float(match[6]))
    val_bbox.append(float(match[7]))
    val_giou.append(float(match[8]))
import ipdb; ipdb.set_trace()
# 데이터 확인
if not epochs:
    print("데이터가 추출되지 않았습니다. 로그 파일의 형식을 확인하세요.")
    exit()

# 그래프 생성
plt.figure(figsize=(12, 8))

# 1. Train Loss & Validation Loss
plt.subplot(2, 2, 1)
plt.plot(epochs, train_loss, label="Train Loss", marker='o')
plt.plot(epochs, val_loss, label="Validation Loss", marker='s')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train Loss & Validation Loss")
plt.legend()
plt.grid()

# 2. Train ce & Validation ce
plt.subplot(2, 2, 2)
plt.plot(epochs, train_ce, label="Train CE", marker='o')
plt.plot(epochs, val_ce, label="Validation CE", marker='s')
plt.xlabel("Epoch")
plt.ylabel("CE Loss")
plt.title("Train CE & Validation CE")
plt.legend()
plt.grid()

# 3. Train giou & Validation giou
plt.subplot(2, 2, 3)
plt.plot(epochs, train_giou, label="Train GIoU", marker='o')
plt.plot(epochs, val_giou, label="Validation GIoU", marker='s')
plt.xlabel("Epoch")
plt.ylabel("GIoU Loss")
plt.title("Train GIoU & Validation GIoU")
plt.legend()
plt.grid()

# 4. Train bbox & Validation bbox
plt.subplot(2, 2, 4)
plt.plot(epochs, train_bbox, label="Train BBox", marker='o')
plt.plot(epochs, val_bbox, label="Validation BBox", marker='s')
plt.xlabel("Epoch")
plt.ylabel("BBox Loss")
plt.title("Train BBox & Validation BBox")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig(log_file.split('/')[2] + '.jpeg')
