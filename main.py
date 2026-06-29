import os
import io
import json

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY chưa được thiết lập. "
        "Nếu chạy local hãy kiểm tra file .env, "
        "nếu chạy trên Render hãy thêm Environment Variable GEMINI_API_KEY."
    )

genai.configure(api_key=API_KEY)

app = FastAPI(
    title="GeoVision AI Backend",
    version="1.0.1",
    description="API phân tích ảnh tư liệu Địa lí bằng AI tạo sinh"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = genai.GenerativeModel("gemini-2.5-flash")


@app.get("/")
async def home():
    return {
        "success": True,
        "message": "GeoVision AI Backend is running"
    }


@app.get("/health")
async def health():
    return {
        "success": True,
        "status": "ok"
    }


@app.post("/analyze-geography-image")
async def analyze_geography_image(
    file: UploadFile = File(...)
):
    grade = "7"
    topic = "Địa lí 7"

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="File ảnh đang bị rỗng."
            )

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="File gửi lên không phải ảnh hợp lệ."
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi đọc ảnh: {str(e)}"
        )

    prompt = f"""
Bạn là giáo viên Địa lí lớp {grade}.
Hãy phân tích ảnh tư liệu Địa lí được học sinh hoặc giáo viên tải lên.

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Ngắn gọn, dễ hiểu, phù hợp học sinh lớp {grade}.
- Nội dung phục vụ dạy học môn Địa lí.
- Chỉ trả về JSON hợp lệ, không thêm giải thích bên ngoài JSON.
- Không dùng markdown.
- Không dùng ```json.
- Không thêm chữ trước hoặc sau JSON.

Chủ đề người dùng chọn: {topic}

Trả về đúng cấu trúc JSON sau:

{{
  "chu_de": "Chủ đề địa lí của ảnh",
  "loai_tu_lieu": "Bản đồ / biểu đồ / cảnh quan / ảnh vệ tinh / dân cư / kinh tế / khí hậu / địa hình",
  "mo_ta_anh": "Mô tả ngắn gọn nội dung ảnh",
  "kien_thuc_trong_tam": [
    "Ý chính 1",
    "Ý chính 2",
    "Ý chính 3"
  ],
  "tu_khoa": [
    "từ khóa 1",
    "từ khóa 2",
    "từ khóa 3"
  ],
  "cau_hoi_trac_nghiem": [
    {{
      "cau_hoi": "Câu hỏi",
      "A": "Đáp án A",
      "B": "Đáp án B",
      "C": "Đáp án C",
      "D": "Đáp án D",
      "dap_an": "A"
    }}
  ],
  "cau_hoi_tu_luan": [
    "Câu hỏi tự luận 1",
    "Câu hỏi tự luận 2"
  ],
  "goi_y_hoat_dong": "Gợi ý hoạt động dạy học từ ảnh này",
  "ghi_nho": "Một câu ghi nhớ ngắn gọn"
}}
"""

    try:
        response = model.generate_content([prompt, image])

        if not response or not getattr(response, "text", None):
            raise HTTPException(
                status_code=500,
                detail="AI không trả về nội dung."
            )

        text = response.text.strip()

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

        try:
            result = json.loads(text)

        except json.JSONDecodeError:
            result = {
                "chu_de": topic,
                "loai_tu_lieu": "Không xác định",
                "mo_ta_anh": text,
                "kien_thuc_trong_tam": [],
                "tu_khoa": [],
                "cau_hoi_trac_nghiem": [],
                "cau_hoi_tu_luan": [],
                "goi_y_hoat_dong": "",
                "ghi_nho": ""
            }

        return {
            "success": True,
            "result": result,
            "note": "Kết quả do AI tạo sinh hỗ trợ, giáo viên cần kiểm tra trước khi sử dụng chính thức."
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi AI: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000))
    )