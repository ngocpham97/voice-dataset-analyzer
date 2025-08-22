import re
from collections import defaultdict


class L0Classifier:
    def __init__(self):
        self.keywords = {
            "hội thoại đời thường": ["khẩu ngữ", "đệm", "đi ăn", "đi học"],
            "dịch vụ/CSKH": ["tài khoản", "hỗ trợ", "khiếu nại", "đơn hàng"],
            "tin tức/thời sự": ["bản tin", "ngày", "địa danh", "tên cơ quan"],
            "giáo dục/giảng giải": ["khái niệm", "ví dụ", "bài học"],
            "doanh nghiệp/họp": ["agenda", "kế hoạch", "OKR", "KPI"],
            "podcast/tự sự": ["trải nghiệm", "suy ngẫm"],
            "giải trí/show": ["trò chuyện", "gameshow"],
            "hướng dẫn/kỹ thuật": ["bước", "cài đặt", "tham số"],
            "tài chính/kinh tế": ["lãi suất", "lạm phát"],
            "y tế": ["triệu chứng", "chẩn đoán"],
            "pháp lý": ["điều khoản", "hợp đồng"],
            "trẻ em": ["vốn từ", "học"],
            "tôn giáo/văn hóa": ["kinh", "nghi lễ"],
            "thể thao": ["tỷ số", "trận đấu"]
        }

    def classify(self, transcript, metadata):
        scores = defaultdict(int)

        # Normalize transcript
        normalized_transcript = self.normalize_transcript(transcript)

        # Count keyword occurrences
        for category, keywords in self.keywords.items():
            for keyword in keywords:
                if re.search(r'\b' + re.escape(keyword) + r'\b', normalized_transcript):
                    scores[category] += 1
        print(f"Classification scores: {scores}")
        # Determine the highest scoring category
        if scores:
            return max(scores, key=scores.get)
        return "unknown"

    def normalize_transcript(self, transcript):
        # Basic normalization: lowercasing and removing extra spaces
        return re.sub(r'\s+', ' ', transcript.lower()).strip()
