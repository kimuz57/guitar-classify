# 🎸 Guitar Classify — ระบบจำแนกคอร์ดกีตาร์แบบเรียลไทม์

โปรเจกต์จำแนกคอร์ดกีตาร์จากเสียง โดยใช้เทคนิค Deep Learning (CNN)
บนข้อมูล **Log-Mel Spectrogram** รองรับการอัดเสียงผ่านไมโครโฟนและทำนายคอร์ดแบบทันที
ผ่าน REST API และมีเครื่องมือเก็บ Dataset ในตัว

---

## ภาพรวมโปรเจกต์

โปรเจกต์แบ่งออกเป็น 3 ส่วนหลัก (แต่ละส่วนแยกกันทำงาน):

| โฟลเดอร์ | บทบาท | เทคโนโลยี |
| --- | --- | --- |
| `guitar-classify-api` | แกนหลัก: เทรนโมเดล + API ทำนายคอร์ด + เก็บ Dataset | Python, FastAPI, TensorFlow/Keras, librosa |
| `guitar-classify-backend` | เซิร์ฟเวอร์ Go (โครงเริ่มต้น สำหรับต่อยอด) | Go, Fiber v2 |
| `guitar-classify-frontend` | เว็บแอปพลิเคชันฝั่งผู้ใช้ | Next.js 16, React 19, Tailwind CSS v4, TypeScript |

> จุดทำงานจริงของโปรเจกต์อยู่ที่ `guitar-classify-api` เป็นหลัก
> ส่วน `backend` และ `frontend` ยังเป็นโครงเริ่มต้น (scaffold) สำหรับพัฒนาต่อ

---

## ฟีเจอร์หลัก

- **ทำนายคอร์ดแบบเรียลไทม์** — อัดเสียงจากไมโครโฟนเป็นช่วง ๆ (ทุก 2 วินาที) ส่งไปทำนายที่ `/predict_stream`
- **เลือกขนาดโมเดลได้** — 8 ขนาด ตั้งแต่ `pico` (เล็กสุด/เร็วสุด) ถึง `large` (แม่นสุด)
- **เก็บ Dataset ผ่าน WebSocket** — เลือกคอร์ดที่หน้าเว็บ อัดเสียงต่อเนื่อง แล้วเซิร์ฟเวอร์ตัดเป็นไฟล์ 2–6 วินาทีให้อัตโนมัติ
- **Data Augmentation** — สร้างข้อมูลเพิ่ม (ลด/เพิ่มเสียง, ใส่ Noise, Time Stretch, Pitch Shift, EQ, Echo)
- **Training Pipeline ครบวงจร** — Feature Extraction แบบ cache, เทรนต่อจาก checkpoint ได้, Early Stopping, Mixed Precision (GPU)
- **ตรวจ Data Leakage** — สคริปต์ตรวจไม่ให้เสียงอัดครั้งเดียวกันไปอยู่ทั้งชุด Train และ Test

---

## โครงสร้างโปรเจกต์

```
Guitar_Classify/
├── README.md                      # เอกสารนี้
├── .gitignore                     # ไฟล์ที่ Git ละเว้นทั้งโปรเจกต์
│
├── guitar-classify-api/           # แกนหลัก (Python / FastAPI)
│   ├── app.py                     # API ทำนายคอร์ดแบบเรียลไทม์ (REST)
│   ├── main.py                    # API เก็บ Dataset ผ่าน WebSocket + สถิติ
│   ├── train.py                   # สคริปต์เทรนโมเดล (8 ขนาด) + Feature Extraction
│   ├── augment_advanced.py        # Data Augmentation ชุดรวม
│   ├── data.py                    # ตรวจสอบ Data Leakage (ช่วยวิเคราะห์)
│   ├── clean_models.py            # ทำความสะอาดไฟล์ .keras ข้ามเวอร์ชัน Keras
│   ├── requirements.txt           # ไลบรารี Python
│   ├── templates/
│   │   ├── index.html             # หน้า UI ทำนายคอร์ด (คู่กับ app.py)
│   │   └── index2.html            # หน้า UI อัดเสียงเก็บ Dataset (คู่กับ main.py)
│   ├── dataset/                   # (สร้างอัตโนมัติ) ไฟล์เสียง .wav แยกตามคอร์ด
│   ├── features_cache/            # (สร้างอัตโนมัติ) Feature cache + label encoder
│   ├── checkpoints/               # (สร้างอัตโนมัติ) Checkpoint ระหว่างเทรน
│   └── model/<variant>/           # (สร้างอัตโนมัติ) โมเดล best_model.keras แยกขนาด
│
├── guitar-classify-backend/       # เซิร์ฟเวอร์ Go (โครงเริ่มต้น)
│   ├── main.go
│   ├── go.mod
│   └── go.sum
│
└── guitar-classify-frontend/      # เว็บแอป (Next.js)
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx
    │   └── globals.css
    ├── public/
    ├── package.json
    └── ...
```

---

## เริ่มต้นใช้งาน

### 1. ส่วน API / โมเดล (`guitar-classify-api`)

ต้องมี Python 3.10+ และติดตั้ง `ffmpeg` (ใช้แปลงไฟล์เสียง webm ผ่าน pydub)

```bash
cd guitar-classify-api
pip install -r requirements.txt
```

**เทรนโมเดล** (เลือกขนาด เช่น `micro`):

```bash
python train.py --variant micro
# สกัด Feature อย่างเดียว (ไม่เทรน)
python train.py --extract_only
```

**รัน API ทำนายคอร์ด** (พอร์ต 8000):

```bash
python app.py          # หรือ uvicorn app:app --host 127.0.0.1 --port 8000
```

**รัน API เก็บ Dataset** (พอร์ต 8000):

```bash
python main.py
```

### 2. ส่วน Backend Go (`guitar-classify-backend`)

```bash
cd guitar-classify-backend
go mod download
go run .
# เปิด http://localhost:3000
```

### 3. ส่วน Frontend (`guitar-classify-frontend`)

```bash
cd guitar-classify-frontend
npm install
npm run dev
# เปิด http://localhost:3000
```

---

## โมเดลและ Pipeline การเทรน

- **Input**: Log-Mel Spectrogram ขนาด `128 (mel) × 190 (time) × 1` ที่ Sample Rate `16000 Hz`
- **สถาปัตยกรรม**: CNN (Conv2D + BatchNorm + MaxPool หลายชั้น → GlobalAveragePooling → Dense + Dropout → Softmax)
- **ขนาดโมเดลที่รองรับ** (`--variant`):
  `pico`, `nano`, `micro`, `small_fast`, `fast`, `small`, `medium`, `large`
- **ลูกเล่นระหว่างเทรน**: Mixed Precision (`mixed_float16`) เมื่อมี GPU, cache Feature ทีละคลาส,
  บันทึก checkpoint ทุก epoch และเทรนต่อได้, EarlyStopping + ReduceLROnPlateau

ดูรายละเอียดทั้งหมดได้ที่ [`guitar-classify-api/README.md`](guitar-classify-api/README.md)

---

## API Endpoints หลัก (`guitar-classify-api`)

| App | Method | Endpoint | คำอธิบาย |
| --- | --- | --- | --- |
| `app.py` | `GET` | `/` | หน้า UI ทำนายคอร์ด (`templates/index.html`) |
| `app.py` | `POST` | `/predict_stream?variant=<variant>` | รับไฟล์เสียง (webm) แล้วคืน `{predicted_chord, confidence}` |
| `main.py` | `GET` | `/` | หน้า UI อัดเสียงเก็บ Dataset |
| `main.py` | `WS` | `/ws/record` | รับชื่อคอร์ด แล้วรับ Packet เสียงต่อเนื่อง เพื่อตัดเป็นไฟล์ Dataset |
| `main.py` | `GET` | `/stats` | สถิติจำนวนไฟล์แยกตามคอร์ดและความยาว |

---

## หมายเหตุ

- โฟลเดอร์ `dataset/`, `features_cache/`, `checkpoints/`, `model/` ถูกสร้างอัตโนมัติเมื่อรันสคริปต์ และไม่ถูก commit
- ไฟล์เสียงที่เก็บได้จะถูกตั้งชื่อเป็น `<คอร์ด>_<ความยาว>s_<timestamp>_<จุดเริ่ม>s.wav`
  เพื่อให้ตรวจ Data Leakage ได้จากการแบ่งกลุ่มตาม timestamp
